"""
Adapted from:
- https://github.com/microsoft/spacy-ann-linker/blob/master/spacy_ann/ann_kb.py
- https://github.com/explosion/spaCy/blob/master/spacy/kb/kb_in_memory.pyx
"""

from dataclasses import dataclass
from typing import Iterable

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import CrossEncoderReranker
from pydantic import create_model
from spacy.tokens import Span

from spacy_lancedb_linker.types import Alias, Entity

FAST_AND_SMALL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class AnnKnowledgeBase:
    uri: str
    # Parameters for the mention candidate generation
    top_k: int = 10
    max_distance: float = 0.5

    def __post_init__(self) -> None:
        """Create an AnnKnowledgeBase."""
        # connect to the LanceDB
        self.db = lancedb.connect(self.uri)

        # Embedding model for aliases and entities
        # any LanceDB-compatible model is available as a drop-in replacement
        # ref: https://lancedb.github.io/lancedb/embeddings/default_embedding_functions/#text-embedding-functions
        self.encoder = (
            get_registry()
            .get("sentence-transformers")
            .create(name=FAST_AND_SMALL, device="cpu")
        )

        # we need pydantic classes to define the Arrow schemas of LanceDB tables.
        # those tables will contain the embedding from our encoder to do the ANN search.
        # because we want to use the local self.encoder, we can't use the traditional pydantic syntax.
        # instead, we create the pydantic classes dynamically using pydantic.create_model
        # ref: https://docs.pydantic.dev/latest/api/base_model/#pydantic.create_model
        self.LanceAlias = create_model(
            "LanceAlias",
            __base__=LanceModel,
            alias=(Alias, ...),
            vector=(Vector(self.encoder.ndims()), self.encoder.VectorField()),
        )
        self.LanceEntity = create_model(
            "LanceEntity",
            __base__=LanceModel,
            entity=(Entity, ...),
            vector=(Vector(self.encoder.ndims()), self.encoder.VectorField()),
        )

        self._initialize_db()

    def _embed(self, text: str) -> list[int]:
        return self.encoder.generate_embeddings([text])[0]

    def _initialize_db(self) -> None:
        # TODO: do better, use mode from params: e.g. the table might already exists
        self.db.create_table("aliases", schema=self.LanceAlias, mode="overwrite")
        self.db.create_table("entities", schema=self.LanceEntity, mode="overwrite")

    def add_aliases(self, aliases: list[Alias]) -> None:
        """Build the ANN index of aliases in LanceDB."""
        table = self.db.open_table("aliases")
        table.add(
            [
                self.LanceAlias(alias=alias, vector=self._embed(alias.alias))
                for alias in aliases
            ]
        )

    def get_candidates_batch(
        self, mentions: Iterable[Span]
    ) -> Iterable[list[tuple[Alias, float]]]:
        return [self.get_candidates(span) for span in mentions]

    def get_candidates(self, mention: Span) -> list[tuple[Alias, float]]:
        return self.get_alias_candidates(mention.text)

    def get_alias_candidates(self, query: str) -> list[tuple[Alias, float]]:
        """Embed a mention query, search ANN neighbours against the aliases index."""
        table = self.db.open_table("aliases")
        results = (
            table.search(self._embed(query))
            .metric("cosine")
            .limit(self.top_k)
            .select(["alias"])
            .to_list()
        )
        filtered_results = [
            r for r in results if abs(r["_distance"]) < self.max_distance
        ]
        return [
            (Alias(**result["alias"]), abs(result["_distance"]))
            for result in filtered_results
        ]

    def _aliases_to_entities(self, aliases: list[tuple[Alias, float]]) -> list[str]:
        return list(
            set(entity_id for alias, _score in aliases for entity_id in alias.entities)
        )

    def get_entity_candidates(self, query: str) -> list[str]:
        """Get the entity IDs corresponding to a mention."""
        return self._aliases_to_entities(aliases=self.get_alias_candidates(query))

    def add_entities(self, entities: list[Entity]) -> None:
        """Build the ANN index of entities in LanceDB."""
        table = self.db.open_table("entities")
        table.add(
            [
                # TODO: add option for when the entity description is not available
                self.LanceEntity(entity=entity, vector=self._embed(entity.description))
                for entity in entities
            ]
        )
        # Create a full-text-search index, ref: https://lancedb.github.io/lancedb/fts/
        table.create_fts_index("entity.name", replace=True)

    def disambiguate(
        self,
        candidate_entities: list[str],
        context_embedding: list[int],
        text_query: str,
    ) -> list[tuple[Entity, float]]:
        """Disambiguate candidate entities by getting the most similar to the context in the doc."""
        table = self.db.open_table("entities")
        # we do a sort of hybrid search between:
        #   - full-text-search on entity names
        #   - vector search on the ANN index by the embedding of the context in the doc
        # ref: https://lancedb.github.io/lancedb/hybrid_search/hybrid_search/
        # ref: https://lancedb.github.io/lancedb/reranking/
        entities_results = (
            table.search(text_query, query_type="fts").select(["entity"]).to_list()
        )
        if len(entities_results):
            cosine_score = 0
            return [
                (Entity(**result["entity"]), cosine_score)
                for result in entities_results
            ]
        else:
            entities_results = (
                table.search(context_embedding)
                .metric("cosine")
                # prefilter for only the candidate entities
                # here we use a DataFusion function: https://lancedb.github.io/lancedb/sql/#sql-filters
                # list_has: https://datafusion.apache.org/user-guide/sql/scalar_functions.html#list-has
                .where(
                    f"list_has({candidate_entities}, entity.entity_id)", prefilter=True
                )
                # get the top_k
                .limit(self.top_k)
                # serialize
                .select(["entity"])
                .to_list()
            )
            return [
                (Entity(**result["entity"]), abs(result["_distance"]))
                for result in entities_results
            ]

    def _disambiguate(
        self,
        candidate_entities: list[str],
        context_embedding: list[int],
        text_query: str,
    ) -> list[tuple[Entity, float]]:
        """We do a multi-vector reranking.

        ref: https://lancedb.github.io/lancedb/reranking/#multi-vector-reranking
        """
        table = self.db.open_table("entities")

        direct = (
            table.search(self._embed(text_query))
            # prefilter for only the candidate entities
            # here we use a DataFusion function: https://lancedb.github.io/lancedb/sql/#sql-filters
            # list_has: https://datafusion.apache.org/user-guide/sql/scalar_functions.html#list-has
            .where(f"list_has({candidate_entities}, entity.entity_id)", prefilter=True)
            # get the top_k
            .limit(self.top_k)
        )

        context = (
            table.search(context_embedding)
            .metric("cosine")
            # prefilter for only the candidate entities
            # here we use a DataFusion function: https://lancedb.github.io/lancedb/sql/#sql-filters
            # list_has: https://datafusion.apache.org/user-guide/sql/scalar_functions.html#list-has
            .where(f"list_has({candidate_entities}, entity.entity_id)", prefilter=True)
            # get the top_k
            .limit(self.top_k)
        )

        # entity.name doesn't work because pyarrow changed pa.Table logic from pyarrow.Schema.field_by_name to pyarrow.Schema.field which doesn't support struct fields
        reranker = CrossEncoderReranker(column="entity.name")
        entities_results = (
            reranker.rerank_multivector(
                vector_results=[direct, context], query=None, deduplicate=True
            )
            # serialize
            .select(["entity"])
            .to_list()
        )
        return [
            (Entity(**result["entity"]), abs(result["_distance"]))
            for result in entities_results
        ]

    """
    We looked at the spacy abstractions closely:
        - spacy.kb.KnowledgeBase
        - spacy.kb.Candidate
    We decided to implement a class that does not use them.
    The reason is that we felt we were shoehorning too much of spacy classes.

    Still, the native spacy classes have the following useful methods that
    were not implemented here:
        - to_bytes
        - from_bytes
        - to_disk
        - from_disk
    """
