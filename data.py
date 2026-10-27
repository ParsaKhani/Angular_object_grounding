import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -100
SYSTEM_PROMPT = "You are a helpful assistant."
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


class SFTDataset(Dataset):
    """LLaVA-style JSON list:
    {"image": "path/rel/to/image_folder.jpg", "conversations": [{"from": "human", "value": "<image> ..."},
                                                              {"from": "gpt", "value": "..."}, ...]}
    """

    def __init__(self, data_path, image_folder=""):
        with open(data_path) as f:
            self.data = json.load(f)
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        paths = item.get("image", [])
        paths = [paths] if isinstance(paths, str) else paths
        images = [Image.open(os.path.join(self.image_folder, p)).convert("RGB") for p in paths]
        return {"images": images, "conversations": [turn["value"] for turn in item["conversations"]]}


class SFTCollator:
    """Builds Qwen2.5-VL chat sequences; only assistant replies (and their <|im_end|>) are supervised."""

    def __init__(self, processor):
        self.processor = processor
        tokenizer = processor.tokenizer
        self.assistant_header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def format(self, conversations):
        text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        for i, turn in enumerate(conversations):
            if i % 2 == 0:
                turn = turn.replace("<image>\n", IMAGE_PLACEHOLDER).replace("<image>", IMAGE_PLACEHOLDER)
                text += f"<|im_start|>user\n{turn}<|im_end|>\n"
            else:
                text += f"<|im_start|>assistant\n{turn}<|im_end|>\n"
        return text

    def labels_for(self, input_ids):
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        header = torch.tensor(self.assistant_header)
        n = len(header)
        for b, ids in enumerate(input_ids):
            i = 0
            while i <= len(ids) - n:
                if torch.equal(ids[i:i + n], header):
                    start = i + n
                    end = start
                    while end < len(ids) and ids[end] != self.im_end_id:
                        end += 1
                    labels[b, start:end + 1] = ids[start:end + 1]
                    i = end + 1
                else:
                    i += 1
        return labels

    def __call__(self, examples):
        texts = [self.format(ex["conversations"]) for ex in examples]
        images = [img for ex in examples for img in ex["images"]] or None
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        batch["labels"] = self.labels_for(batch["input_ids"])
        return batch
