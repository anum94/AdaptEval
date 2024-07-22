import evaluate
import nltk

from metrics.metric import Metric


class Rouge(Metric):

    metric_name = "rouge"

    def __init__(self) -> None:
        nltk.download("punkt", quiet=True)
        self.metric = evaluate.load(self.metric_name)

    def compute(self, predictions: list, references: list) -> dict:
        return self.metric.compute(
            predictions=predictions, references=references, use_stemmer=True
        )
