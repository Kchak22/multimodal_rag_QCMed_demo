import json

def safe_json_parse(text: str) -> dict:
    """
    Safely parse JSON output from LLM, fallback to empty metrics.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "precision_contexte": 0.0,
            "fidelite": 0.0,
            "pertinence": 0.0,
            "precision_sources": 0.0,
            "commentaire": "JSON parsing failed"
        }

def format_sources_for_prompt(retrieved_docs: list) -> str:
    """
    Format retrieved sources into readable string for LLM prompt.
    """
    lines = []
    for doc in retrieved_docs:
        text_snippet = doc.get("text", "")[:150]
        source = doc.get("metadata", {}).get("source_document", "Unknown")
        lines.append(f"- {source}: {text_snippet}...")
    return "\n".join(lines)
