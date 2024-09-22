# spacy-lancedb-linker

> spaCy pipeline component for ANN Entity Linking using LanceDB

## Installation

```sh
pip install spacy-lancedb-linker
```

## Usage

```python
import spacy

from spacy_lancedb_linker.kb import AnnKnowledgeBase
from spacy_lancedb_linker.linker import AnnLinker  # noqa
from spacy_lancedb_linker.types import Alias, Entity

nlp = spacy.load("en_core_web_md")
kb = AnnKnowledgeBase(uri="data/sample-lancedb")
kb.add_entities(
    [
        Entity(**entity)
        for entity in [
            {
                "entity_id": "a3",
                "name": "Natural language processing",
                "description": "Natural language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data.",
            },
            {
                "entity_id": "a4",
                "name": "Neuro-linguistic programming",
                "description": "Neuro-linguistic programming (NLP) is a pseudoscientific approach to communication, personal development, and psychotherapy created by Richard Bandler and John Grinder in California, United States in the 1970s.",
            },
        ]
    ]
)
kb.add_aliases(
    [
        Alias(**alias)
        for alias in [
            {"alias": "NLP", "entities": ["a3", "a4"], "probabilities": [0.5, 0.5]},
            {
                "alias": "Natural language processing",
                "entities": ["a3"],
                "probabilities": [1.0],
            },
            {
                "alias": "Neuro-linguistic programming",
                "entities": ["a4"],
                "probabilities": [1.0],
            },
        ]
    ]
)

ann_linker = nlp.add_pipe("ann_linker", last=True)
ann_linker.set_kb(kb)

doc = nlp("NLP is a subset of machine learn.")

print(doc.ents[0].kb_id_)
print(doc.ents[0]._.alias_candidates)
print(doc.ents[0]._.kb_candidates)
```

## Test

```sh
poetry install
poetry shell
poetry run pytest
```
