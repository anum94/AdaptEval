from metrics.rouge import Rouge
from metrics.bertscore import BertScore
from metrics.vocab_overlap import VocabOverlab
translate_metric_name = {"rouge": Rouge, "bertscore": BertScore, "vocab_overlab": VocabOverlab}


def load_metrics(
    metrics: list,
) -> list:

    res = []
    for metric in metrics:
        assert metric in translate_metric_name
        res.append(translate_metric_name[metric]())

    return res
