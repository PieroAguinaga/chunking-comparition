# RAG Chunking Strategy Benchmark

An experimental framework that compares four document-chunking strategies for
Retrieval-Augmented Generation (RAG) on academic paper content.

The benchmark ingests a Markdown-formatted research paper, applies each
chunking strategy to produce a set of vector-indexed chunks, and then
evaluates retrieval quality against a curated Q&A dataset using an LLM judge.

---

## Motivation

Chunking is one of the most impactful — and least studied — decisions in a RAG
pipeline.  Splitting a document poorly means the retriever will either miss
relevant information or return overly broad, noisy context.  This project
systematically tests four different splitting approaches on the same corpus
and measures which one produces the most useful retrieved context.

---

## Chunking Strategies

| Strategy | Key Idea |
|---|---|
| **Markdown Header** | Splits at H1 / H2 / H3 heading boundaries. Fast and structure-aware; works best when the source document uses consistent headings. |
| **Page-Based** | Splits at `---PAGE---` delimiters inserted during PDF conversion. Preserves the original page layout; useful when page-level granularity is meaningful. |
| **Semantic** | Uses embedding cosine similarity between consecutive sentences to find natural topic transitions (LangChain `SemanticChunker`). Adapts to content density. |
| **Agentic** | A LangGraph workflow loops over document sections and calls an LLM with `with_structured_output` to identify chunk boundaries. The LLM returns typed `SectionChunks` (Pydantic) — no JSON parsing needed. Produces the richest metadata (title + key concepts) and handles irregular academic structure well. |

---

## Architecture

```
converted_files/          Raw Markdown files (one per paper)
        │
        ▼
chunking_strategy_comparison.py
        │   applies all 4 strategies
        ▼
chunkers.py               markdown_header_chunker
                          page_chunker
                          semantic_chunker
                          agentic_chunker  ◄── LLM-driven
        │
        │   embed (text-embedding-3-small, 1536-dim)
        ▼
Supabase / pgvector       table: documents
  ┌──────────────────────────────────────────┐
  │ id │ paper_id │ method │ content │ embedding │ metadata │
  └──────────────────────────────────────────┘
        │
        ▼
retrieval_chain.py        match_documents RPC  (cosine IVFFlat)
        │
        ▼
test_retrieval.py         Q&A dataset  ──►  LLM judge  ──►  accuracy per method
```

---

## Setup

### Prerequisites

- Python 3.10+
- A [Supabase](https://supabase.com) project (local or cloud) with the
  `pgvector` extension enabled
- An Azure OpenAI deployment with:
  - An embeddings model (`text-embedding-3-small`)
  - A chat model (e.g. `gpt-4o`) for the agentic chunker and judge

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```env
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_KEY=<your-service-role-key>

AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_CHAT_DEPLOYMENT=<your-chat-deployment-name>
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

### Initialise the database schema

Run the SQL file once against your Supabase project (via the SQL Editor or
the Supabase CLI):

```bash
supabase db reset          # local only — or paste semantic_database.sql in the cloud editor
```

The schema creates:
- The `documents` table with a `vector(1536)` column
- An IVFFlat index for cosine similarity search
- The `match_documents` RPC function used by the retrieval layer

---

## Usage

### Step 1 — Ingest and embed all chunks

```bash
python ingest_chunks.py
```

This reads every `.md` file in `converted_files/`, applies all four chunking
strategies, generates embeddings, and inserts the results into Supabase.
Expected output:

```
Loaded 1 document(s) from 'converted_files/'.

[1/4] Applying Markdown Header chunker...
[2/4] Applying Page chunker...
[3/4] Applying Semantic chunker...
[4/4] Applying Agentic chunker (LLM-driven)...

Total chunks generated: 142
  markdown_header     : 28 chunks
  page                : 12 chunks
  semantic            : 47 chunks
  agentic             : 55 chunks

Inserting chunks into Supabase...
Successfully inserted 142 chunks into 'documents'.
```

### Step 2 — Run the evaluation

```bash
python test_retrieval.py
```

For each method the script retrieves the top-5 most similar chunks for every
question in `dataset_rag.json` and asks the judge LLM for a binary verdict.

```
=== Evaluation Results ===
  agentic             : 13/25 correct (52.00%)
  markdown_header     : 12/25 correct (48.00%)
  semantic            :  1/25 correct  (4.00%)
  page                :  1/25 correct  (4.00%)
```

### Quick retrieval test

```bash
python -m utils.retrieval_chain.py
```

Runs a single sample query against the `page` method and prints the top-5
results with similarity scores.

---

## Evaluation Results

| Method | Correct | Accuracy |
|---|---|---|
| **Agentic** | 13/25 | 52.00% |
| **Markdown Header** | 12/25 | 48.00% |
| **Semantic** | 1/25 | 4.00% |
| **Page** | 1/25 | 4.00% |

---

## Conclusions

### 1. Agentic chunking leads, but no strategy performs well in absolute terms

The LLM-driven agentic chunker achieves the highest accuracy at 52 %, followed
closely by markdown header at 48 %. Both structure-aware strategies answer
roughly half the questions correctly. Semantic and page chunking essentially
fail (4 %), suggesting a fundamental retrieval breakdown for those methods.

### 2. Semantic and Page strategies show near-total retrieval failure

Both strategies fell to 4 % — just 1 correct answer out of 25. The most likely
cause is **embedding inconsistency between ingestion and query time**: if the
vectors stored in pgvector were generated with a different model version or
normalization than the query vectors, cosine similarity rankings become
essentially random. Fine-grained strategies are the most vulnerable because
their chunks must be extremely well-aligned to surface the right passage.

Possible root causes to investigate:
- Different `text-embedding-3-small` deployment or API version between ingestion and evaluation
- Database not re-indexed before this run (stale vectors from a prior session)
- Embedding normalization inconsistency in the retrieval helper

### 3. Structure-aware strategies are more robust

Markdown Header and Agentic chunking both survived far better than Semantic
and Page. Structure-aware chunks tend to be larger and semantically denser,
so even a slightly misaligned embedding still lands in the right neighbourhood.
Fine-grained or layout-based chunks carry no such safety net.

### 4. The 52 % ceiling suggests a systemic evaluation issue

A best-case accuracy of 52 % is lower than expected even for a moderate
chunking strategy. The long per-question latency observed for semantic and
page methods (~8 s/question vs ~1.5 s for agentic/markdown) points to
possible retrieval timeouts or judge LLM overload that could be artificially
suppressing scores across all methods.

### 5. Recommended next steps

- **Re-embed from scratch before every evaluation run** to guarantee vector
  consistency across ingestion and retrieval.
- **Log the embedding model version** alongside every ingestion job so
  mismatches can be detected immediately.
- **Add a latency budget** to the judge loop and profile the retrieval
  bottleneck observed in semantic/page.
- **Increase dataset size** beyond 25 questions to reduce variance; with n=25
  a single wrong answer shifts accuracy by 4 pp.

---

## Dataset

`dataset_rag.json` contains 25 question–answer pairs spanning five categories:

| Category | Count | Example question |
|---|---|---|
| methodology | 9 | *"What activation function does the tutorial use instead of ReLU?"* |
| dataset | 4 | *"At what sampling rate were acoustic signals recorded?"* |
| metrics | 4 | *"Which RL algorithm performed best?"* |
| limitations | 3 | *"What limitations of neural networks are mentioned?"* |
| comparison | 3 | *"How does a CNN differ from an MLP?"* |
| context | 4 | *"Why are power transformers considered critical assets?"* |

Questions range from easy (single fact lookup) to hard (multi-concept
synthesis), providing a balanced benchmark signal.

---

## Note on PDF Conversion

To ensure a fair comparison across all chunking strategies, the source PDF was
converted to Markdown using **LlamaIndex** prior to any chunking.

LlamaIndex was chosen for this step because, after evaluating several
alternatives, it consistently produced the cleanest Markdown output for
academic PDFs: headings, inline math, tables, and figure captions were all
preserved with high fidelity.  This matters because strategies like
`markdown_header` and `agentic` rely on the document structure remaining intact
after conversion — garbage-in would mean garbage-out regardless of how
sophisticated the chunker is.

The resulting file (`converted_files/2512.22190v1.md`) is the single source of
truth that all four strategies consume, so any difference in evaluation scores
is attributable to the chunking strategy alone, not to conversion quality.

---

## Source Paper

> J. I. Aizpurua, *"Physics-Informed Machine Learning for Transformer Condition
> Monitoring — Part I: Basic Concepts, Neural Networks, and Variants"*,
> 2025 8th International Advanced Research Workshop on Transformers (ARWTr),
> Baiona, Spain, 2025. arXiv:2512.22190v1

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM / Embeddings | Azure OpenAI (GPT-4o + text-embedding-3-small) |
| Orchestration | LangChain |
| Vector store | Supabase + pgvector |
| Semantic chunker | `langchain-experimental` SemanticChunker |
| Agentic chunker | Custom — LLM-as-chunker via structured JSON prompting |