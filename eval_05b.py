import re
from accelerate import Accelerator

from llava.model.builder import load_pretrained_model
import torch
from custom_dataset import collate_fn_factory, NaturalBench

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from accelerate.utils import gather_object


SUFFIX_FOR_VQA = {
    "yes_no": "Please answer Yes or No.",
    "multiple_choice": "Please output the letter corresponding to the correct option.",
}


def extract_answer(output_string, task_type="yes_no"):
    """
    Extracts the answer from the output string based on the task type.

    Parameters:
    output_string (str): The output string.
    task_type (str): The type of task. Must be either "yes_no" or "multiple_choice".

    Returns:
    int:
        1 if "yes" or "A"
        0 if "no" or "B"
        -1 if no relevant answer is found.
        Raises a ValueError if an unsupported task_type is provided.
    """

    def find_word_position(string, word):
        pattern = r"\b" + re.escape(word) + r"\b"
        match = re.search(pattern, string, re.IGNORECASE)
        if match:
            return match.start()
        return -1

    if task_type not in ["yes_no", "multiple_choice"]:
        raise ValueError(
            "Task type not supported. Must be 'yes_no' or 'multiple_choice'."
        )

    if task_type == "yes_no":
        position_yes_and_a = find_word_position(output_string, "yes")
        position_no_and_b = find_word_position(output_string, "no")
    elif task_type == "multiple_choice":
        position_yes_and_a = find_word_position(output_string, "A")
        position_no_and_b = find_word_position(output_string, "B")

    if position_yes_and_a == -1 and position_no_and_b == -1:
        print(f"No answer found in the output string: {output_string}.")
        return -1
    elif position_yes_and_a != -1 and position_no_and_b != -1:
        return 1 if position_yes_and_a < position_no_and_b else 0
    else:
        return 0 if position_yes_and_a == -1 else 1


def get_scores(scores):
    """
    Calculate various scores based on the given results.

    Args:
        scores (dict or list): A dictionary or list containing results where each result can be:
            - dict: {id: {"q0_i0": 1 or 0, "q0_i1": 1 or 0, "q1_i0": 1 or 0, "q1_i1": 1 or 0}, ...}
            - list: [[q0_i0 (1 or 0), q0_i1 (1 or 0), q1_i0 (1 or 0), q1_i1 (1 or 0)], ...]

    The keys "q0_i0", "q0_i1", "q1_i0", "q1_i1" represent combinations of questions and images:
        - "q0_i0" means question_0 on image_0
        - "q0_i1" means question_0 on image_1
        - "q1_i0" means question_1 on image_0
        - "q1_i1" means question_1 on image_1

    Returns:
        dict: A dictionary containing the calculated scores:
            - 'Q_Acc': Average question score
            - 'I_Acc': Average image score
            - 'Acc': Average binary VQA score
            - 'G_Acc': Average group score
    """
    Q_Acc = 0.0
    I_Acc = 0.0
    Acc = 0.0
    G_Acc = 0.0

    num_samples = len(scores)

    def calculate_image_score(result):
        image_correct = 0
        if isinstance(result, dict):
            if result["q0_i0"] == 1.0 and result["q1_i0"] == 0.0:
                image_correct += 1
            if result["q1_i1"] == 1.0 and result["q0_i1"] == 0.0:
                image_correct += 1
        elif isinstance(result, list):
            if result[0] == 1.0 and result[2] == 0.0:
                image_correct += 1
            if result[3] == 1.0 and result[1] == 0.0:
                image_correct += 1
        return image_correct

    def calculate_question_score(result):
        text_correct = 0
        if isinstance(result, dict):
            if result["q0_i0"] == 1.0 and result["q0_i1"] == 0.0:
                text_correct += 1
            if result["q1_i1"] == 1.0 and result["q1_i0"] == 0.0:
                text_correct += 1
        else:
            if result[0] == 1.0 and result[1] == 0.0:
                text_correct += 1
            if result[3] == 1.0 and result[2] == 0.0:
                text_correct += 1
        return text_correct

    def calculate_binary_score(result):
        binary_score_correct = 0
        if isinstance(result, dict):
            binary_score_correct += 1 if result["q0_i0"] == 1.0 else 0
            binary_score_correct += 1 if result["q0_i1"] == 0.0 else 0
            binary_score_correct += 1 if result["q1_i0"] == 0.0 else 0
            binary_score_correct += 1 if result["q1_i1"] == 1.0 else 0
        else:
            binary_score_correct += 1 if result[0] == 1.0 else 0
            binary_score_correct += 1 if result[1] == 0.0 else 0
            binary_score_correct += 1 if result[2] == 0.0 else 0
            binary_score_correct += 1 if result[3] == 1.0 else 0

        return binary_score_correct

    def calculate_group(result):
        group_correct = 0
        if calculate_question_score(result) == 2 and calculate_image_score(result) == 2:
            group_correct += 1

        return group_correct

    if isinstance(scores, dict):
        for _, result in scores.items():
            Q_Acc += calculate_question_score(result)
            I_Acc += calculate_image_score(result)
            Acc += calculate_binary_score(result)
            G_Acc += calculate_group(result)
    else:
        for result in scores:
            Q_Acc += calculate_question_score(result)
            I_Acc += calculate_image_score(result)
            Acc += calculate_binary_score(result)
            G_Acc += calculate_group(result)

    results = {
        "Q_Acc": Q_Acc / float(num_samples * 2),
        "I_Acc": I_Acc / float(num_samples * 2),
        "Acc": Acc / float(num_samples * 4),
        "G_Acc": G_Acc / num_samples,
    }

    return results


def run_eval(output_path: Path = Path("outputs/default_output.txt")):
    accelerator = Accelerator()

    def ranked_print(*msg):
        if accelerator.is_local_main_process:
            print(*msg)

    ranked_print("loading model")
    # 3. Test Models: use the naturalbench dataset to test your own models and get the "output_file" of your model
    warnings.filterwarnings("ignore")
    pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
    model_name = "llava_qwen"
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained,
        None,
        model_name,
        device_map=None,
        attn_implementation="flash_attention_2",
    )  # Add any other thing you want to pass in llava_model_args
    model.config.tokenizer_padding_side = "left"

    dataset = NaturalBench()
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_fn_factory(
            image_processor, model.config, tokenizer, accelerator.device
        ),
    )

    model, dataloader = accelerator.prepare(model.to(torch.bfloat16), dataloader)
    model.to(accelerator.device)

    model.eval()

    ranked_print("starting inference")
    outputs = []

    with tqdm(total=len(dataloader), desc="running inference", disable=not accelerator.is_local_main_process) as pbar:
        for input_ids, image_tensor, image_sizes, indices in dataloader:
            cont = model.module.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True,
                temperature=0.8,
                top_k=10,
                modalities=["image" for _ in image_tensor],
                max_new_tokens=200,
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            text_outputs = [text.encode("unicode_escape").decode("utf-8") for text in text_outputs]
            outputs += [",".join([str(i), t]) for i, t in zip(indices, text_outputs)]
            pbar.update(1)

    accelerator.wait_for_everyone()
    outputs = gather_object(outputs)

    # only runs in the main process (rank 0)
    if accelerator.is_local_main_process:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with output_path.open("w") as f:
            f.write("\n".join(outputs))

        output_file = output_path.open().readlines()
        outputs = []
        for line in output_file:
            parts = line.split(",")
            index = int(parts[0])
            ans = ",".join(parts[1:])
            outputs.append((index, ans))

        # sort the lines
        output_file = [o[1] for o in sorted(outputs, key=lambda o: o[0])]
        ranked_print(output_file)

        # 4. Extract the answer: extract the answer from the outputs (you could also use LLMs such as ChatGPT to extract the answer)
        ranked_print("Number of outputs:", len(output_file))
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
        ranked_print(scores)
        


if __name__ == "__main__":
    import tyro

    tyro.cli(run_eval)
