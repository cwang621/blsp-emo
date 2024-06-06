import os
import argparse
import json
from tqdm import tqdm
import torch
from typing import Any, Dict, List, Optional, Union, Tuple
import re
import string

from dataclasses import dataclass
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader

from transformers import WhisperFeatureExtractor
from transformers import GenerationConfig

from src.modeling_blsp2 import Blsp2Model
from src.configuration_blsp2 import Blsp2Config
from src.tokenization_qwen import QWenTokenizer
from src.instruction_dataset import get_waveform
from src.qwen_generation_utils import decode_tokens, get_stop_words_ids

def collate_tokens(
        values: List[List[int]],
        pad_id: int
):
    size = max(len(v) for v in values)
    batch_size = len(values)
    res = torch.LongTensor(batch_size, size).fill_(pad_id)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(torch.LongTensor(v), res[i][-len(v):])

    return res


@dataclass
class DataCollator:
    pad_id: int = 0
    sampling_rate: int = 16000
    extractor: WhisperFeatureExtractor = WhisperFeatureExtractor()

    def __call__(self, samples: List[Dict]):
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        suffix_input_ids = [sample["suffix_input_ids"] for sample in samples]
        suffix_attention_mask = [sample["suffix_attention_mask"] for sample in samples]
        reference = [sample["reference"] for sample in samples]

        input_ids = collate_tokens(input_ids, self.pad_id)
        attention_mask = collate_tokens(attention_mask, 0)
        suffix_input_ids = collate_tokens(suffix_input_ids, self.pad_id)
        suffix_attention_mask = collate_tokens(suffix_attention_mask, 0)

        raw_speech = [
            get_waveform(sample["audio"], output_sample_rate=self.sampling_rate) if sample["audio"] is not None else []
            for sample in samples
        ]
        if all(len(sample) == 0 for sample in raw_speech):
            speech_values = None
            speech_attention_mask = None
        else:
            speech_inputs = self.extractor(
                raw_speech, 
                sampling_rate=self.sampling_rate, 
                return_attention_mask=True,
                return_tensors="pt"
            )
            speech_values = speech_inputs.input_features.to(torch.float16)
            speech_attention_mask = speech_inputs.attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "suffix_input_ids": suffix_input_ids,
            "suffix_attention_mask": suffix_attention_mask,
            "speech_values": speech_values,
            "speech_attention_mask": speech_attention_mask,
            "reference": reference
        }
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, default=None,
        help="Path to the input file", required=True
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Path to the output file", required=True
    )
    parser.add_argument(
        "--blsp_model", type=str, default=None,
        help="Path to the blsp model", required=True
    )
    parser.add_argument(
        "--instruction", type=str, default="",
        help="the general instruction for each example"
    )
    parser.add_argument(
        "--audio_field", type=str, default="",
        help="the audio filed for each example"
    )
    parser.add_argument(
        "--text_field", type=str, default="",
        help="the text field for each example"
    )
    parser.add_argument(
        "--reference_field", type=str, default="",
        help="the reference field for each example"
    )
    parser.add_argument(
        "--batch_size", type=int, default=24,
        help="batch size"
    )
    parser.add_argument(
        "--use_emotion", type=bool, default=False,
        help="use emotion sensitive system message"
    )
    ### args for generation
    parser.add_argument(
        "--max_new_tokens", type=int, default=128,
        help="max new tokens for generation"
    )
    parser.add_argument(
        "--min_new_tokens", type=int, default=1,
        help="min new tokens for generation"
    )
    parser.add_argument(
        "--do_sample", action="store_true",
        help="whether do sample. For ST task, we will use greedy search to ensure stable output"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="temperature for generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.5,
        help="top_p for generation"
    )
    parser.add_argument(
        "--top_k", type=int, default=0,
        help="top_k for generation"
    )
    args = parser.parse_args()

    print(args)


    tokenizer = QWenTokenizer.from_pretrained(args.blsp_model)
    extractor = WhisperFeatureExtractor.from_pretrained(args.blsp_model)
    generation_config = GenerationConfig.from_pretrained(args.blsp_model)
    stop_words_ids = get_stop_words_ids(generation_config.chat_format, tokenizer)

    with open(args.input_file, "r") as fin:
        lines = fin.readlines()
        lines = [json.loads(line.strip()) for line in lines]
    dataset = Dataset.from_list(lines)

    im_start_tokens = [tokenizer.im_start_id]
    im_end_tokens = [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")
    def process_dataset(batch):
        def _tokenize_str(role, content):
            return tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())
        instruction = args.instruction
        if not instruction:
            instruction = batch.get("instruction", "")
        if args.text_field:
            text = batch.get(args.text_field, "")
            instruction = instruction + text
        
        if args.use_emotion:
            system_prompt = "You are a helpful assistant. Your response should fulfill requests with empathy toward user's emotion tone."
        else:
            system_prompt = "You are a helpful assistant."
        
        ### prefix
        input_ids = []
        input_ids += im_start_tokens + _tokenize_str("system", system_prompt) + im_end_tokens
        input_ids += nl_tokens
        input_ids += im_start_tokens + _tokenize_str("user", instruction)
        batch["input_ids"] = input_ids
        batch["attention_mask"] = [1] * len(batch["input_ids"])
        ### audio
        if args.audio_field:
            batch["audio"] = batch.get(args.audio_field, None)
        else:
            batch["audio"] = None
        ### suffix
        suffix_input_ids = im_end_tokens + nl_tokens + im_start_tokens + \
            tokenizer.encode("assistant") # \n is removed and used as bos_token
        batch["suffix_input_ids"] = suffix_input_ids
        batch["suffix_attention_mask"] = [1] * len(batch["suffix_input_ids"])
        ### reference
        if args.reference_field:
            batch["reference"] = batch.get(args.reference_field, "")
        else:
            batch["reference"] = ""

        return batch
    
    dataset = dataset.map(process_dataset)
    model = Blsp2Model.from_pretrained(args.blsp_model, torch_dtype=torch.float16)
    model = model.half()  ### on A100, Qwen will automatically converting to bf16

    data_collator = DataCollator(generation_config.pad_token_id, extractor.sampling_rate, extractor)
    dataloader = DataLoader(
        dataset, 
        collate_fn=data_collator,
        batch_size=args.batch_size
    )
    
    generation_config.update(
        **{
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "num_beams": 1,
            "num_return_sequences": 1,
            "bos_token_id": nl_tokens[0], ### need bos_token_id when input_ids is not provided
        }
    )

    model = model.cuda()
    model.eval()
    with open(args.output_file, "w") as fout:
        for batch in tqdm(dataloader):
            outputs = model.generate(
                input_ids=batch["input_ids"].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
                suffix_input_ids=batch["suffix_input_ids"].cuda(),
                suffix_attention_mask=batch["suffix_attention_mask"].cuda(),
                speech_values=batch["speech_values"].cuda() if batch["speech_values"] is not None else None,
                speech_attention_mask=batch["speech_attention_mask"].cuda() if batch["speech_attention_mask"] is not None else None,
                generation_config=generation_config,
                stop_words_ids=stop_words_ids,
            )
            output_text = [
                decode_tokens(
                    output,
                    tokenizer,
                    raw_text_len=0,
                    context_length=0,
                    chat_format=generation_config.chat_format,
                    verbose=False,
                    errors='replace'
                )
                for output in outputs
            ]
            for reference, response in zip(batch["reference"], output_text):
                json_string = json.dumps(
                    {
                        "response": response,
                        "reference": reference
                    },
                    ensure_ascii=False,
                )
                print(json_string, file=fout, flush=True)

if __name__ == "__main__":
    main()
