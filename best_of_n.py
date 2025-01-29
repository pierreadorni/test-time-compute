import csv
from pathlib import Path
from itertools import chain
import numpy as np

import tyro
import pandas as pd

from custom_dataset import NaturalBench
from eval import extract_answer, get_scores 

@tyro.cli
def main(output_paths: list[Path]):
    #dataset = load_dataset("BaiqiL/NaturalBench")
    #question_types = list(chain.from_iterable([[item["Question_Type"]] * 4 for item in dataset["train"]]))

    def load_output_file(p: Path):
        df = pd.read_csv(p, names=["idx", "answer"], index_col="idx").sort_index()["answer"]
        #df.apply(lambda x: extract_answer(x, "yes_no"))
        return df
    outputs = [load_output_file(p) for p in output_paths]
    outputs = pd.concat(outputs, axis=1, keys=list(range(len(outputs))))
    print(outputs.head())
    output_file = outputs.mode(axis=1)[0]

    print("Loading dataset")
    dataset = NaturalBench()

    print("Number of outputs:", len(output_file))
    answers = {}
    number_answered_samples = len(output_file) // 4
    for i in range(number_answered_samples):
        answers[i] = {
            "q0_i0": extract_answer(
                output_file[i * 4], dataset.naturalbench[i * 4][3]
            ),
            "q0_i1": extract_answer(
                output_file[i * 4 + 1], dataset.naturalbench[i * 4 + 1][3]
            ),
            "q1_i0": extract_answer(
                output_file[i * 4 + 2], dataset.naturalbench[i * 4 + 2][3]
            ),
            "q1_i1": extract_answer(
                output_file[i * 4 + 3], dataset.naturalbench[i * 4 + 3][3]
            ),
        }

    # 5. Calculate the scores
    scores = get_scores(answers)
    print(scores)
