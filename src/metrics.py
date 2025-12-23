from typing import List, Dict

def aggregate_metrics(scores: List[Dict]) -> Dict:
    """
    Aggregate list of LLMJudge metrics over all evaluation items.
    """
    n = len(scores)
    if n == 0:
        return {}

    aggregated = {
        "precision_contexte_mean": sum(s.get("precision_contexte", 0.0) for s in scores) / n,
        "fidelite_mean": sum(s.get("fidelite", 0.0) for s in scores) / n,
        "pertinence_mean": sum(s.get("pertinence", 0.0) for s in scores) / n,
        "precision_sources_mean": sum(s.get("precision_sources", 0.0) for s in scores) / n
    }
    return aggregated
