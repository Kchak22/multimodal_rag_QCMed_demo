from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.embedder import Embedder
from src.vector_store import QdrantVectorStore
from src.rag_engine import RAGEngine
from src.llm_judge import LLMJudge
from src.evaluator import JudgeEvaluator
from llama_index.llms.ollama import Ollama
import json

import os
import json

# Load dataset
with open("../data/evaluation/golden_dataset_Cours12.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Initialize RAG
print("=== Initializing RAG System ===")
embedder = Embedder()

vector_store = QdrantVectorStore(collection_name="multimodal_rag", reset_collection=False)

rag = RAGEngine(
    vector_store=vector_store,
    embedder=embedder,
)

# init Model client
print("=== Initializing LLM Judge ===")
llm_client = Ollama(
    model="llama3",
    base_url="http://localhost:11434",
    request_timeout=120.0
    )

# Init LLM judge
llm_judge_instance = LLMJudge(llm=llm_client)
evaluator = JudgeEvaluator(rag_pipeline=rag, judge=llm_judge_instance)

# Evaluate
print("=== Starting Evaluation ===")
results = evaluator.evaluate_dataset(dataset)
print("=== Evaluation Complete ===")

print("=== Results ===")
print("Aggregated metrics:", results["aggregated"])

print("=== Saving Results ===")
output_dir = "../experiments"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "latest_results.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Results saved to {output_path}")