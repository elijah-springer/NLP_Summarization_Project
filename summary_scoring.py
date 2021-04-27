from rouge import Rouge

from nltk import sent_tokenize


def rouge_score(machine_generated: str, human_reference: str):
    return Rouge().get_scores(machine_generated, human_reference)


def simple_score(machine_generated: str, human_reference: str):
    pass
