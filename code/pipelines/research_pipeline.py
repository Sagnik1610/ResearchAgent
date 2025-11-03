from typing import Dict
from tqdm.auto import tqdm

from pipelines.agents import (
    ProblemIdentifier,
    ProblemValidator,
    MethodDeveloper,
    MethodValidator,
    ExperimentDesigner,
    ExperimentValidator,
)
from utils.evaluation import get_avg_feedbacks_score, get_num_feedbacks_scores


class ResearchPipeline:
    """
    Iterative pipeline:
      1. Identify and validate research problems
      2. Propose and validate methods
      3. Design and validate experiments

    Works seamlessly with Krutrim/OpenAI-compatible API clients.
    """

    def __init__(self, api_client=None, iterations: int = 3):
        self.iterations = max(1, iterations)
        self.problem_identifier = ProblemIdentifier(api_client)
        self.problem_validator = ProblemValidator(api_client)
        self.method_developer = MethodDeveloper(api_client)
        self.method_validator = MethodValidator(api_client)
        self.experiment_designer = ExperimentDesigner(api_client)
        self.experiment_validator = ExperimentValidator(api_client)

        self._labels = [
            "Problem Identifier", "Problem Validator",
            "Method Developer", "Method Validator",
            "Experiment Designer", "Experiment Validator",
        ]
        self._label_width = max(len(s) for s in self._labels)

    def _log(self, message: str) -> None:
        tqdm.write(str(message))

    def _fmt(self, label: str) -> str:
        return f"[{label.ljust(self._label_width)}]"

    def _select_best(self, history_list, key_label: str):
        """Safely select the best item from a list based on feedback scores."""
        if not history_list:
            return {}
        scored = [
            (get_avg_feedbacks_score(item.get("feedbacks") or {}), item)
            for item in history_list
            if get_num_feedbacks_scores(item.get("feedbacks") or {}) > 0
        ]
        if not scored:
            return history_list[-1]  # fallback: last iteration
        best_item = max(scored, key=lambda x: x[0])[1]
        self._log(f"{self._fmt(key_label)} Selected best item with avg feedback score.")
        return best_item

    def run(self, context: Dict) -> Dict:
        history = {"problems": [], "methods": [], "experiments": []}
        self._log(f"{self._fmt('Paper Title')} {context['paper']['title']}")

        # --- Problem Ideation Loop ---
        for i in range(self.iterations):
            self._log(f"{self._fmt('Problem Identifier')} Iteration {i + 1}/{self.iterations} — generating problem…")
            context.update(self.problem_identifier.run(context) or {})

            self._log(f"{self._fmt('Problem Validator')} Iteration {i + 1}/{self.iterations} — validating problem…")
            context.update(self.problem_validator.run(context) or {})

            history["problems"].append(
                {
                    "problem": context.get("problem"),
                    "rationale": context.get("problem_rationale"),
                    "feedbacks": context.get("problem_feedbacks"),
                }
            )

        best_problem = self._select_best(history["problems"], "Problem Validator")
        context.update(best_problem)

        # --- Method Development Loop ---
        for i in range(self.iterations):
            self._log(f"{self._fmt('Method Developer')} Iteration {i + 1}/{self.iterations} — proposing method…")
            context.update(self.method_developer.run(context) or {})

            self._log(f"{self._fmt('Method Validator')} Iteration {i + 1}/{self.iterations} — validating method…")
            context.update(self.method_validator.run(context) or {})

            history["methods"].append(
                {
                    "method": context.get("method"),
                    "rationale": context.get("method_rationale"),
                    "feedbacks": context.get("method_feedbacks"),
                }
            )

        best_method = self._select_best(history["methods"], "Method Validator")
        context.update(best_method)

        # --- Experiment Design Loop ---
        for i in range(self.iterations):
            self._log(f"{self._fmt('Experiment Designer')} Iteration {i + 1}/{self.iterations} — designing experiment…")
            context.update(self.experiment_designer.run(context) or {})

            self._log(f"{self._fmt('Experiment Validator')} Iteration {i + 1}/{self.iterations} — validating experiment…")
            context.update(self.experiment_validator.run(context) or {})

            history["experiments"].append(
                {
                    "experiment": context.get("experiment"),
                    "rationale": context.get("experiment_rationale"),
                    "feedbacks": context.get("experiment_feedbacks"),
                }
            )

        best_experiment = self._select_best(history["experiments"], "Experiment Validator")
        context.update(best_experiment)

        context.update({"history": history})
        return context
