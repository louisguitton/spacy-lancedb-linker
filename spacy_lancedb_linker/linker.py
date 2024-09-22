from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from spacy.language import Language
from spacy.tokens import Doc, Span

from spacy_lancedb_linker.kb import AnnKnowledgeBase


@dataclass
class AnnLinker:
    """The AnnLinker adds Entity Linking capabilities
    to map NER mentions to KnowledgeBase Aliases or directly to KnowledgeBase Ids
    """

    kb: AnnKnowledgeBase | None = None
    use_disambiguation_threshold: bool = False

    def __post_init__(self) -> None:
        """Initialize the AnnLinker

        nlp (Language): spaCy Language object
        """
        Span.set_extension("alias_candidates", default=[], force=True)
        Span.set_extension("kb_candidates", default=[], force=True)

    def __call__(self, doc: Doc) -> Doc:
        """Annotate spaCy doc.ents with candidate info.
        If disambiguate is True, use entity vectors and doc context
        to pick the most likely Candidate

        doc (Doc): spaCy Doc

        RETURNS (Doc): spaCy Doc with updated annotations
        """
        if not self.kb:
            raise ValueError("KnowledgeBase `kb` required for AnnLinker")

        # TODO: have a configurable set on mentions
        mentions = doc.ents
        batch_candidates = self.kb.get_candidates_batch(mentions)

        for ent, alias_candidates in zip(doc.ents, batch_candidates):
            ent._.alias_candidates = alias_candidates

            if len(alias_candidates) == 0:
                continue
            else:
                candidate_entities = self.kb._aliases_to_entities(alias_candidates)

                # TODO: have a configurable context (e.g. -1/+1 sentence)
                context_embedding = self.kb._embed(ent.sent)

                kb_candidates = self.kb.disambiguate(
                    candidate_entities, context_embedding, ent.text
                )

                ent._.kb_candidates = kb_candidates

                if self.use_disambiguation_threshold:
                    filtered_results = [
                        (entity, cosine_score)
                        for entity, cosine_score in kb_candidates
                        if cosine_score < self.kb.max_distance
                    ]
                else:
                    filtered_results = kb_candidates

                if len(filtered_results):
                    best_candidate = filtered_results[0][0]
                    for token in ent:
                        token.ent_kb_id_ = best_candidate.entity_id
        return doc

    def set_kb(self, kb: AnnKnowledgeBase) -> None:
        """Set the KnowledgeBase."""
        self.kb = kb

    def from_disk(self, path: Path, **kwargs) -> None:
        """Deserialize saved AnnLinker from disk."""
        raise NotImplementedError("This is not available at this time.")

    def to_disk(self, path: Path, exclude: Tuple = tuple(), **kwargs) -> None:
        """Serialize AnnLinker to disk."""
        raise NotImplementedError("This is not available at this time.")


@Language.factory("ann_linker")
def create_ann_linker(nlp: Language, name: str) -> AnnLinker:
    return AnnLinker()
    return AnnLinker()
    return AnnLinker()
