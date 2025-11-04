import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from utils.evaluation import METRIC2DESCRIPTION


class BaseAgent(ABC):
    def __init__(self, api_client: Any = None):
        self.api_client = api_client

    @abstractmethod
    def run(self, context: Dict) -> Dict:
        raise NotImplementedError

    def call(self, messages: Optional[List[Dict[str, str]]] = None) -> str:
        return self.api_client.call(messages=messages)

    def _format_target_paper(self, paper: Dict[str, Any]) -> str:
        return f"Target paper title: {paper.get('title')} \nTarget paper abstract: {paper.get('abstract')} \n\n"

    def _format_related_papers(self, references: List[Dict[str, Any]], include_abstract: bool = True) -> str:
        return "".join(
            [
                f"Related paper #{index+1} title: {related_paper.get('title')}"
                + (f"\nRelated paper #{index+1} abstract: {related_paper.get('abstract')}" if include_abstract else "")
                + "\n\n"
                for index, related_paper in enumerate(references)
            ]
        )

    def _format_entities(self, entities: List[str]) -> str:
        return f"Entities (separated by the token '|'): {' | '.join(entities) if entities else ''} \n\n"

    def _format_feedbacks(self, type: str, feedbacks: Dict[str, Dict[str, Any]]) -> str:
        return "\n".join([
            f"{metric}: \n- Definition: {METRIC2DESCRIPTION[type][metric]} \n- Review: {feedback['review']} \n- Feedback: {feedback['feedback']} \n"
            for metric, feedback in feedbacks.items()
        ]) + "\n"

    def parse_output(self, text: str, expected_keys: List[str]) -> Dict[str, Any]:
        if not text:
            return {key: None for key in expected_keys}

        outputs = {}
        for i, key in enumerate(expected_keys):
            try:
                # Pattern to capture content between the current key and the next, or end of string
                next_key = expected_keys[i + 1] if i + 1 < len(expected_keys) else None
                pattern = f"{key}:\s*(.*?)\s*(?:\n{next_key}:|$)"
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

                if match:
                    outputs[key.lower()] = match.group(1).strip()
                else:
                    outputs[key.lower()] = None
            except re.error:
                outputs[key.lower()] = None # Handle cases with invalid regex patterns

        # Special handling for rating to make it more robust
        if 'rating' in outputs:
            rating_text = outputs.get('rating')
            if rating_text:
                match = re.search(r'([1-5])', rating_text)
                if match:
                    try:
                        outputs['rating'] = int(match.group(1))
                    except (ValueError, IndexError):
                        outputs['rating'] = None
                else:
                    outputs['rating'] = None

        return outputs
