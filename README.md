# Project DEFUSE

Our goal is to improve LLM's ability to respond to confusing questions in RAG setting. A question is confusing if it has no good answer because it is based on a false assumption or interpretation of the document. We want the LLM to respond by pointing out the false premise in the question ("defusing" the confusion), rather than by playing along and creating even more confusion.

This is work in progress.

## LLM API Keys

File [`llmlib.py`](llmlib.py) has the code that interacts with remote LLMs. You need to provide your API Keys and other relevant environment variables, for example, in an `.env` file, like so:
```
OPENAI_API_KEY=...
OPENAI_ORGANIZATION_ID=...
OPENAI_PROJECT_ID=...
```

## Generating Confusing Questions

Use notebook [`datagen.ipynb`](datagen.ipynb) to generate confusing questions for a collection of documents. The notebook functions are implemented in [`datagen.py`](datagen.py), while specific LLM calls with prompts and few-shot examples are prepared in [`promptlib.py`](promptlib.py).
