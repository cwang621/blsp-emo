import logging
import os
import sys
import json
import fire
import random
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass

from transformers import GenerationConfig
from src.configuration_qwen import QWenConfig
from src.modeling_qwen import QWenLMHeadModel
from src.tokenization_qwen import QWenTokenizer
from src.qwen_generation_utils import decode_tokens, get_stop_words_ids
from src.instruction_dataset import collate_tokens

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("emotion text generation")


EMOTION_FORMAT = "Continue the following sentence that reflects a '{emotion}' emotion tone in a coherent style: "

def get_shard_range(tot, nshard, rank):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(
        f"rank {rank} of {nshard}, process {end-start} "
        f"({start}-{end}) out of {tot}"
    )
    return start, end


def get_dataset(manifest, nshard, rank):
    with open(manifest, "r") as f:
        lines = f.readlines()
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start:end]
        lines = [json.loads(line.strip()) for line in lines]
    dataset = Dataset.from_list(lines)

    return dataset


@dataclass
class DataCollator:
    pad_id: int = 0

    def __call__(self, samples: List[Dict]):
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        audio = [sample["audio"] for sample in samples]
        text = [sample["text"] for sample in samples]
        instruction = [sample["instruction"] for sample in samples]
        emotion = [sample["emotion"] for sample in samples]

        input_ids = collate_tokens(input_ids, self.pad_id, left_pad=True)
        attention_mask = collate_tokens(attention_mask, 0, left_pad=True)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio": audio,
            "text": text,
            "instruction": instruction,
            "emotion": emotion,
        }
    

def generate(
    qwen_path,
    manifest,
    lab_dir,
    instruction="",
    nshard=4,
    rank=0,
    batch_size=4,
    use_emotion=False,
):
    accelerator = Accelerator()
    logger.info(accelerator.state)

    device = accelerator.device

    dataset = get_dataset(manifest, nshard, rank)
    tokenizer = QWenTokenizer.from_pretrained(qwen_path)
    generation_config = GenerationConfig.from_pretrained(qwen_path)
    generation_config.update(
        **{
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "max_new_tokens": 48,
            "min_new_tokens": 1,
        }
    )
    config = QWenConfig.from_pretrained(qwen_path)
    config.update(
        {
            "use_flash_attn": False ### flash_attn do not support V100
        }
    )
    model = QWenLMHeadModel.from_pretrained(qwen_path, config=config, torch_dtype=torch.float16)

    im_start_tokens = [tokenizer.im_start_id]
    im_end_tokens = [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")

    def process_dataset(batch):
        def _tokenize_str(role, content):
            return tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())
        text = batch["text"]
        input_ids = []
        emotion = batch.get("emotion", "")
        if use_emotion:
            _instruction = EMOTION_FORMAT.format(emotion=emotion)
            system_promt = "You are a helpful assistant. Your response should fulfill requests with empathy toward user's emotion tone."
        else:
            _instruction = instruction
            system_promt = "You are a helpful assistant."
        prompt = _instruction + text
        input_ids += im_start_tokens + _tokenize_str("system", f"{system_promt}") + im_end_tokens
        input_ids += nl_tokens
        input_ids += im_start_tokens + _tokenize_str("user", f"{prompt}") + im_end_tokens
        input_ids += nl_tokens
        input_ids += im_start_tokens + _tokenize_str("assistant", "")

        batch["input_ids"] = input_ids
        batch["attention_mask"] = [1] * len(batch["input_ids"])
        batch["audio"] = batch.get("audio", "")
        batch["text"] = text
        batch["instruction"] = _instruction
        batch["emotion"] = emotion
        return batch
    
    dataset = dataset.map(process_dataset)

    data_collator = DataCollator(generation_config.pad_token_id)
    dataloader = DataLoader(
        dataset, 
        collate_fn=data_collator, 
        batch_size=batch_size
    )

    ### prepare everything
    model, dataloader = accelerator.prepare(
        model, dataloader
    )
    model.to(device)
    model.eval()

    split = os.path.splitext(os.path.basename(manifest))[0]
    lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.jsonl"

    os.makedirs(lab_dir, exist_ok=True)

    progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
    stop_words_ids = get_stop_words_ids(generation_config.chat_format, tokenizer)
    with open(lab_path, "w") as f:
        for batch in dataloader:
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                generation_config=generation_config,
                stop_words_ids=stop_words_ids
            )
            input_length = batch["input_ids"].shape[1]
            outputs = outputs[:, input_length:]
            outputs = [
                decode_tokens(
                    output,
                    tokenizer,
                    raw_text_len=0,
                    context_length=0,
                    chat_format=generation_config.chat_format,
                    verbose=False,
                    return_end_reason=True,
                    errors='replace'
                )
                for output in outputs
            ]
            for audio, text, _instruction, emotion, output in zip(batch["audio"], batch["text"], batch["instruction"], batch["emotion"], outputs):
                output_text = output[0]
                early_stop = not ("<|im_end|>" in output[1])
                json_string = json.dumps(
                    {
                        "audio": audio,
                        "text": text,
                        "instruction": _instruction,
                        "emotion": emotion,
                        "output": output_text,
                        "early_stop": early_stop
                    },
                    ensure_ascii=False
                )
                print(json_string, file=f, flush=True)
            progress_bar.update(1)

    logger.info("finished successfully")
    

if __name__ == "__main__":
    fire.Fire({
        'generate': generate,
    })