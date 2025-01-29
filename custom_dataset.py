from torch.utils.data import Dataset
from datasets import load_dataset
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import copy

SUFFIX_FOR_VQA = {
    "yes_no": "Please answer Yes or No.",
    "multiple_choice": "Please output the letter corresponding to the correct option.",
}


class NaturalBench(Dataset):
    def __init__(self, vqa_suffix=SUFFIX_FOR_VQA):
        dataset = load_dataset("BaiqiL/NaturalBench")
        # print("constructing naturalbench")
        # 2.Use NaturalBench: construct 1900*4 [question, image, correct_answer] samples from the dataset with 1900 samples
        self.naturalbench = []
        # print(len(dataset["train"]))
        for i, item in enumerate(tqdm(dataset["train"])):
            self.naturalbench.append(
                [
                    item["Question_0"] + " " + vqa_suffix[item["Question_Type"]],
                    item["Image_0"],
                    item["Image_0_Question_0"],
                    item["Question_Type"],
                ]
            )
            self.naturalbench.append(
                [
                    item["Question_0"] + " " + vqa_suffix[item["Question_Type"]],
                    item["Image_1"],
                    item["Image_1_Question_0"],
                    item["Question_Type"],
                ]
            )
            self.naturalbench.append(
                [
                    item["Question_1"] + " " + vqa_suffix[item["Question_Type"]],
                    item["Image_0"],
                    item["Image_0_Question_1"],
                    item["Question_Type"],
                ]
            )
            self.naturalbench.append(
                [
                    item["Question_1"] + " " + vqa_suffix[item["Question_Type"]],
                    item["Image_1"],
                    item["Image_1_Question_1"],
                    item["Question_Type"],
                ]
            )

    def __len__(self):
        return len(self.naturalbench)

    def __getitem__(self, idx):
        return [*self.naturalbench[idx], idx]


def collate_fn_factory(image_processor, model_config, tokenizer, device):
    def collate_fn(batch):
        conv_template = "qwen_1_5"
        batch_image_tensor = process_images(
            [example[1] for example in batch], image_processor, model_config
        )
        batch_image_tensor = [
            _image.to(dtype=torch.bfloat16, device=device)
            for _image in batch_image_tensor
        ]
        batch_input_ids = []
        image_sizes = [example[1].size for example in batch]
        for example in batch:
            question = DEFAULT_IMAGE_TOKEN + "\n" + example[0]
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            batch_input_ids.append(input_ids)

        padded_input_ids = pad_sequence(
            batch_input_ids, batch_first=True, padding_side="left"
        )
        # answers = [example[2] for example in batch]
        indices = [example[4] for example in batch]

        return padded_input_ids, batch_image_tensor, image_sizes, indices

    return collate_fn
