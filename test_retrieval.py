"""
test_retrieval.py
------------------
Evaluation harness for the RAG Chunking Strategy Benchmark.

For every chunking strategy stored in the database, this script:
    1. Loads a Q&A dataset (dataset_rag.json) derived from the source paper.
    2. Retrieves the top-5 most similar chunks for each question using the
       strategy under evaluation.
    3. Asks a judge LLM whether the retrieved context is sufficient to answer
       the question correctly, given the ground-truth answer.
    4. Reports per-strategy accuracy (correct / total questions).

The judge LLM is instructed to focus on semantic equivalence, not exact
wording, making the evaluation robust to paraphrasing.

Usage:
    python test_retrieval.py

Expected output (example):
    === Evaluation Results ===
    markdown_header  :  21/25 correct (84.00%)
    page             :  18/25 correct (72.00%)
    semantic         :  22/25 correct (88.00%)
    agentic          :  23/25 correct (92.00%)

Environment variables required (loaded via .env):
    SUPABASE_URL             — Supabase project URL
    SUPABASE_KEY             — Supabase service-role key
    AZURE_OPENAI_API_KEY     — Azure OpenAI API key
    AZURE_OPENAI_ENDPOINT    — Azure OpenAI endpoint URL
    AZURE_OPENAI_API_VERSION — API version (e.g. "2024-02-01")
    AZURE_CHAT_DEPLOYMENT    — Deployment name for the judge chat model
"""

import json
import os
from collections import defaultdict

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from supabase import create_client
from tqdm import tqdm
from config.settings import settings
from utils.retrieval_chain import search_similar_chunks
from utils.llm import get_llm
load_dotenv()

# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------

supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)
# ---------------------------------------------------------------------------
# Judge LLM
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Evaluation prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = PromptTemplate(
    input_variables=["question", "ground_truth", "model_answer"],
    template="""You are an objective evaluator for a question-answering system.

Your task is to determine whether the retrieved content contains sufficient
information to correctly answer the question, given the ground-truth answer.

Evaluation guidelines:
- Focus on whether the retrieved content contains the necessary information.
- Exact wording does NOT need to match the ground truth.
- Minor differences in phrasing are acceptable.
- The answer is correct if the meaning is semantically equivalent.
- If key information is missing, misleading, or incorrect, mark it as incorrect.
- Ignore formatting, style, or grammar differences.

Question:
{question}

Ground Truth Answer:
{ground_truth}

Retrieved Content:
{model_answer}

Decision:
Respond with ONLY one of the following (no explanation):
Correct
Incorrect
""",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_all_methods() -> list[str]:
    """
    Fetch the distinct chunking method names present in the database.

    Returns:
        List of method identifier strings (e.g. ["markdown_header", "page",
        "semantic", "agentic"]).
    """
    res     = supabase.table("documents").select("method").execute()
    methods = list({row["method"] for row in res.data})
    return methods


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_chunking_methods(json_file_path: str) -> dict[str, dict]:
    """
    Run the full evaluation pipeline across all chunking strategies.

    For each method found in the database the function iterates over every
    Q&A pair in the dataset, retrieves the top-5 chunks, concatenates their
    text as context, and asks the judge LLM for a binary correct/incorrect
    verdict.

    Args:
        json_file_path: Path to the Q&A dataset JSON file.  Expected format:
                        [{"question": "...", "answer": "...", ...}, ...]

    Returns:
        Dictionary mapping each method name to its statistics:
            {"correct": int, "total": int}
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    methods = get_all_methods()
    print(f"Methods found in database: {methods}\n")

    stats   = defaultdict(lambda: {"correct": 0, "total": 0})
    llm     = get_llm()
    chain   = JUDGE_PROMPT | llm

    for method in methods:
        print(f"Evaluating method: '{method}'")

        for qa in tqdm(qa_pairs, desc=f"  {method}"):
            # Retrieve top-5 chunks for this question using the current method
            docs    = search_similar_chunks(query=qa["question"], method=str(method))
            context = "\n\n".join([d["content"] for d in docs])

            # Ask the judge LLM for a verdict
            verdict = chain.invoke(
                input={
                    "question":     qa["question"],
                    "ground_truth": qa["answer"],
                    "model_answer": context,
                }
            ).content.strip().lower()

            stats[method]["total"] += 1
            if verdict == "correct":
                stats[method]["correct"] += 1

    # -- Results report -------------------------------------------------------
    print("\n=== Evaluation Results ===")
    for method, s in sorted(stats.items()):
        accuracy = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {method:<20}: {s['correct']:>2}/{s['total']} correct ({accuracy:.2f}%)")

    return dict(stats)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    evaluate_chunking_methods("dataset_rag.json")
