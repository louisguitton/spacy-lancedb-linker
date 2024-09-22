import pytest
import spacy
import srsly
from spacy.language import Language

from spacy_lancedb_linker.kb import AnnKnowledgeBase
from spacy_lancedb_linker.linker import AnnLinker  # noqa
from spacy_lancedb_linker.types import Alias, Entity


@pytest.fixture(scope="session")
def entities() -> list[Entity]:
    return [
        Entity(**entity) for entity in srsly.read_jsonl("data/sample/entities.jsonl")
    ]


@pytest.fixture(scope="session")
def aliases() -> list[Alias]:
    return [Alias(**alias) for alias in srsly.read_jsonl("data/sample/aliases.jsonl")]


@pytest.fixture(scope="session")
def nlp() -> Language:
    return spacy.load("en_core_web_md")


@pytest.fixture(scope="session")
def kb(entities: list[Entity], aliases: list[Alias]) -> AnnKnowledgeBase:
    uri = "data/sample-lancedb"
    ann_kb = AnnKnowledgeBase(uri=uri)
    ann_kb.add_entities(entities)
    ann_kb.add_aliases(aliases)
    return ann_kb


def test_kb(kb: AnnKnowledgeBase):
    assert kb.get_alias_candidates("ML") == [
        (
            Alias(alias="ML", entities=["a1", "a2"], probabilities=[0.5, 0.5]),
            1.1920928955078125e-07,
        )
    ]

    candidate_entities = kb.get_entity_candidates("ML")
    assert sorted(candidate_entities) == ["a1", "a2"]

    doc_embedding = kb._embed(
        "Linear regression is one of the first statistical models used by students of ML"
    )
    assert kb.disambiguate(candidate_entities, doc_embedding, "ML") == [
        (
            Entity(
                entity_id="a1",
                name="Machine learning",
                description="Machine learning (ML) is the scientific study of algorithms and statistical models...",
                label=None,
            ),
            0.4473797082901001,
        ),
        (
            Entity(
                entity_id="a2",
                name="Meta Language",
                description='ML ("Meta Language") is a general-purpose functional programming language. It has roots in Lisp, and has been characterized as "Lisp with types".',
                label=None,
            ),
            0.6272382736206055,
        ),
    ]

    assert kb.get_alias_candidates("learning") == [
        (
            Alias(alias="Machine learning", entities=["a1"], probabilities=[1.0]),
            0.3687775135040283,
        )
    ]
    assert kb.get_alias_candidates("Machine") == []


def test_linker(nlp: Language, kb: AnnKnowledgeBase, aliases: list[Alias]):
    # given
    ruler = nlp.add_pipe("entity_ruler")
    patterns = [
        {"label": "SKILL", "pattern": alias}
        for alias in [a.alias for a in aliases] + ["machine learn"]
    ]
    ruler.add_patterns(patterns)

    ann_linker = nlp.add_pipe("ann_linker", last=True)
    ann_linker.set_kb(kb)

    # when
    doc = nlp("NLP is a subset of machine learn.")
    ent_nlp = doc.ents[0]
    ent_ml = doc.ents[1]

    # then
    assert ent_nlp.kb_id_ == "a3"
    assert ent_nlp._.alias_candidates == [
        (Alias(alias="NLP", entities=["a3", "a4"], probabilities=[0.5, 0.5]), 0.0),
        (
            Alias(
                alias="Natural language processing",
                entities=["a3"],
                probabilities=[1.0],
            ),
            0.39776819944381714,
        ),
    ]
    assert ent_nlp._.kb_candidates == [
        (
            Entity(
                entity_id="a3",
                name="Natural language processing",
                description="Natural language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data.",
                label=None,
            ),
            0.3475921154022217,
        ),
        (
            Entity(
                entity_id="a4",
                name="Neuro-linguistic programming",
                description="Neuro-linguistic programming (NLP) is a pseudoscientific approach to communication, personal development, and psychotherapy created by Richard Bandler and John Grinder in California, United States in the 1970s.",
                label=None,
            ),
            0.3534255027770996,
        ),
    ]

    assert ent_ml._.alias_candidates == [
        (
            Alias(alias="Machine learning", entities=["a1"], probabilities=[1.0]),
            0.15205204486846924,
        )
    ]
    assert ent_ml._.kb_candidates == [
        (
            Entity(
                entity_id="a1",
                name="Machine learning",
                description="Machine learning (ML) is the scientific study of algorithms and statistical models...",
                label=None,
            ),
            0,
        )
    ]
