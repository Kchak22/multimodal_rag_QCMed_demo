import json
from typing import List, Dict
import re

class LLMJudge:
    """
    Wrapper for LLM-as-a-Judge evaluation (French).
    """

    def __init__(self, llm, max_tokens=1500, temperature=0.0):
        """
        Args:
            llm: LLM client instance with .complete(prompt) method
            max_tokens: max tokens for LLM completion
            temperature: LLM temperature
        """
        self.llm = llm
        self.max_tokens = max_tokens
        self.temperature = temperature

    def evaluate(
        self,
        question: str,
        ground_truth: str,
        rag_answer: str,
        context: str,
        retrieved_sources: List[Dict]
    ) -> Dict:
        """
        Evaluate a single question-answer pair with retrieved context and sources.
        Returns a JSON-like dict with metrics.
        """
        prompt = self.build_prompt(question, ground_truth, rag_answer, context, retrieved_sources)
        llm_output = self.llm.complete(prompt)
        answer = str(llm_output)
        llm_output_to_json = self.parse_json(answer)
        llm_output_to_json["question"] = question
        llm_output_to_json["SystemAnswer"] = rag_answer
        return llm_output_to_json

    def build_prompt(
        self,
        question: str,
        ground_truth: str,
        rag_answer: str,
        context: str,
        retrieved_sources: List[Dict]
    ) -> str:
        # Convert sources to concise string
        sources_str = "\n".join([
            f"- {s['metadata'].get('source_document', 'Unknown')}: {s['text'][:150]}..."
            for s in retrieved_sources
        ])
        return f"""
Tu es un évaluateur expert pour systèmes de question-réponse en français.

Question :
{question}

Contexte récupéré :
{context}

Sources récupérées :
{sources_str}

Réponse générée par le système :
{rag_answer}

Réponse correcte :
{ground_truth}

Évalue sur une échelle de 0 à 1 :
1. Précision du contexte
2. Fidélité
3. Pertinence
4. Précision des sources

IMPORTANT :
- Réponds UNIQUEMENT avec un objet JSON valide
- N’ajoute AUCUN texte avant ou après le JSON
- Pas de markdown, pas d’explication, pas de titre
- Ajoute un simple commentaire en français dans le champ "commentaire" concernant la qualité globale de la réponse

Format EXACT attendu :
{{
  "precision_contexte": 0.0,
  "fidelite": 0.0,
  "pertinence": 0.0,
  "precision_sources": 0.0,
  "commentaire": "string",
}}
"""

    @staticmethod
    def parse_json(llm_output: str) -> Dict:
        """
        Extracts the JSON object from the LLM output.
        Returns default values if parsing fails.
        """
        try:
            # Try parsing directly first
            return json.loads(llm_output)
        except json.JSONDecodeError:
            # If fails, try to extract JSON object inside {...}
            match = re.search(r"\{.*\}", llm_output, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        print("Warning: JSON parsing failed for LLM output.")
        # Fallback default
        return {
            "precision_contexte": 0.0,
            "fidelite": 0.0,
            "pertinence": 0.0,
            "precision_sources": 0.0,
            "commentaire": "JSON parsing failed",
            "question": "",
            "SystemAnswer": "",
        }