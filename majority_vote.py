import re
from datasets import load_dataset

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from custom_dataset import collate_fn_factory, NaturalBench

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

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
        pattern = r'\b' + re.escape(word) + r'\b'
        match = re.search(pattern, string, re.IGNORECASE)
        if match:
            return match.start()
        return -1
    
    if task_type not in ["yes_no", "multiple_choice"]:
        raise ValueError("Task type not supported. Must be 'yes_no' or 'multiple_choice'.")
    
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

@torch.no_grad()
def majority_vote(N, input_ids, image_tokens, model, max_length):
    batched_text_tokens = torch.repeat(text_tokens, N)
    outputs = model.generate(batched_text_tokens)
    answers = extract_answer(outputs)
    best_answers = torch.mode(answers.cpu()).cuda()
    return best_answers

def run_eval(output_path: Path = Path("outputs/default_output.txt")):
    print("loading dataset")

    print("loading model")
    # 3. Test Models: use the naturalbench dataset to test your own models and get the "output_file" of your model
    warnings.filterwarnings("ignore")
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation='flash_attention_2')  # Add any other thing you want to pass in llava_model_args
    model.config.tokenizer_padding_side = 'left'

    dataset = NaturalBench()
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collate_fn_factory(image_processor, model.config, tokenizer)
    )

    model.eval()

    print("starting inference")
    outputs = []
    for input_ids, image_tensor, image_sizes in tqdm(dataloader):
        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            modalities=["image" for _ in image_tensor],
            max_new_tokens=4096,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        outputs += text_outputs
        
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with output_path.open("w") as f:
        f.write("\n".join(outputs))

    output_file = output_path.open().readlines()
    # 4. Extract the answer: extract the answer from the outputs (you could also use LLMs such as ChatGPT to extract the answer)
    print("Number of outputs:", len(output_file))
    answers = {}
    number_answered_samples = len(output_file)//4
    for i in range(number_answered_samples):
        answers[i] = {
            "q0_i0": extract_answer(output_file[i*4], dataset.naturalbench[i*4][3]),
            "q0_i1": extract_answer(output_file[i*4+1], dataset.naturalbench[i*4+1][3]),
            "q1_i0": extract_answer(output_file[i*4+2], dataset.naturalbench[i*4+2][3]),
            "q1_i1": extract_answer(output_file[i*4+3], dataset.naturalbench[i*4+3][3])
        }

    #5. Calculate the scores
    scores = get_scores(answers)
    print(scores)

if __name__ == "__main__":
    import tyro
    tyro.cli(run_eval)
