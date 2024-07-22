class Metric:

    metric_name: str = ""

    def __init__(self) -> None:
        pass

    def compute(self, predictions: list, references: list) -> dict:
        pass
