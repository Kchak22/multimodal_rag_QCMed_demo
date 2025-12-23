from typing import Dict

from tqdm import tqdm
from src.metrics import aggregate_metrics

class JudgeEvaluator:
    """
    Orchestrates evaluation of a RAG system with LLM-as-a-Judge.
    """

    def __init__(self, rag_pipeline, judge):
        """
        Args:
            rag_pipeline: instance of your RAG engine with .query() method
            judge: instance of LLMJudge
        """
        self.rag = rag_pipeline
        self.judge = judge

    def evaluate_dataset(self, dataset: Dict) -> Dict:
        """
        dataset: dict with keys:
            - metadata
            - samples: list of QA items

        Returns:
            - aggregated metrics
            - raw per-sample metrics
        """
        scores = []

        for item in tqdm(dataset["samples"], desc="Evaluating questions"):
            # 1. Run RAG
            result = self.rag.query(item["question"], return_context=True)

            # 2. Judge evaluation
            metrics = self.judge.evaluate(
                question=item["question"],
                ground_truth=item["answer"],
                rag_answer=result["answer"],
                context=result["context"],
                retrieved_sources=result["retrieved_docs"]
            )

            scores.append(metrics)

        aggregated = aggregate_metrics(scores)

        return {
            "aggregated": aggregated,
            "raw_scores": scores
        }
