#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.

import logging
import os
import sys
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Tuple
import random

import datasets
import evaluate
import torch

from peft import LoraConfig

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import WhisperConfig, WhisperFeatureExtractor, GenerationConfig
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers import WhisperFeatureExtractor

from src.instruction_dataset import load_instruction_datasets, InstructionDataCollator
from src.modeling_blsp2 import Blsp2Model
from src.modeling_whisper_encoder import WhisperEncoder
from src.configuration_blsp2 import Blsp2Config
from src.modeling_qwen import QWenLMHeadModel
from src.configuration_qwen import QWenConfig
from src.tokenization_qwen import QWenTokenizer



logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    blsp_model: str = field(
        default="", metadata={"help": "the path of blsp model"}
    )
    qwen_model: str = field(
        default="Qwen/Qwen-7B-Chat", metadata={"help": "the path of qwen model (ignored if blsp_model set)"}
    )
    whisper_model: str = field(
        default="openai/whisper-small", metadata={"help": "the path of whisper model (ignored if blsp_model set"}
    )
    lora_r: int = field(
        default=16, metadata={"help": "the rank of lora module"}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": "the alpha of lora module"}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "the dropout ratio of lora module"}
    )
    lora_target_modules: str = field(
        default="c_attn,c_proj,w1,w2", metadata={"help": "the target modules of lora module"}
        # default="q_proj,k_proj,v_proj,o_proj", metadata={"help": "the target modules of lora module"}
    )
    lora_scope: str = field(
        default="audio", metadata={"help": "scope of lora adaptation, choose from (audio, text, global)"}
    )
    unfreeze_qwen: bool = field(
        default=False, metadata={"help": "whether to unfreeze the qwen parameters (via lora)"}
    )
    unfreeze_whisper: bool = field(
        default=False, metadata={"help": "whether to unfreeze the whisper parameters"}
    )
    unfreeze_adapter: bool = field(
        default=False, metadata={"help": "whether to unfreeze the adapter parameters"}
    )
    blsp_config: str = field(
        default="", metadata={"help": "config file to specify other parameters"}
    )
    loss_names: str = field(
        default="response_kl", metadata={"help": "choose from (cif, response_ce, response_kl, input_kl)"
                                                 ", separated by ,"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_dirs: str = field(
        default="", metadata={"help": "directories (separated by '|') to load and save processed datasets, other data "
                                      "arguments ignored if set"}
    )
    manifest_dirs: str = field(
        default="", metadata={"help": "directories (separated by '|') to load dataset from manifest files"}
    )
    manifest_files: str = field(
        default="", metadata={"help": "manifest files (separated by '|' between datasets and then ',' between files) "
                                      "of the training manifest files"}
    )
    instructions: str = field(
        default="", metadata={"help": "instructions (separated by '|') for datasets"}
    )
    instruction_fields: str = field(
        default="", metadata={"help": "instruction_fields (separated by '|') to read from manifest_files"}
    )
    audio_instructions: str = field(
        default="", metadata={"help": "audio_instructions (separated by '|') for datasets, if provided"}
    )
    audio_instruction_fields: str = field(
        default="", metadata={"help": "audio_instruction_fields (separated by '|') to read from manifest_files, "
                                      "if provided"}
    )
    input_fields: str = field(
        default="", metadata={"help": "input_fields (separated by '|') to read from manifest_files"}
    )
    audio_fields: str = field(
        default="", metadata={"help": "audio_fields (separated by '|') to read from manifest_files"}
    )
    output_fields: str = field(
        default="", metadata={"help": "output_fields (separated by '|') to read from manifest_files"}
    )
    sample_probs: str = field(
        default="", metadata={"help": "sample_probs (separated by '|') for each dataset (needed for more than one "
                                      "dataset)"}
    )
    max_length: int = field(
        default=384, metadata={"help": "samples that have more text tokens than this limit are removed"}
    )
    dataset_save_dir: str = field(
        default="", metadata={"help": "save the resulting dataset for future use"}
    )
    interleave_stopping_strategy: str = field(
        default="first_exhausted", metadata={"help": "choose from 'first_exhausted' (default) and 'all_exhausted'"}
    )

def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Load tokenizer
    if model_args.blsp_model:
        tokenizer = QWenTokenizer.from_pretrained(model_args.blsp_model)
        generation_config = GenerationConfig.from_pretrained(model_args.blsp_model)
        extractor = WhisperFeatureExtractor.from_pretrained(model_args.blsp_model)
    else:
        tokenizer = QWenTokenizer.from_pretrained(model_args.qwen_model)
        generation_config = GenerationConfig.from_pretrained(model_args.qwen_model)
        extractor = WhisperFeatureExtractor.from_pretrained(model_args.whisper_model)

    ### 5. Load dataset
    dataset = load_instruction_datasets(data_args, tokenizer=tokenizer)

    # 6. Load pretrained model
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    if model_args.blsp_model:
        model = Blsp2Model.from_pretrained(model_args.blsp_model)
    else:
        whisper_config = WhisperConfig.from_pretrained(model_args.whisper_model)
        qwen_config = QWenConfig.from_pretrained(model_args.qwen_model)
        qwen_config.update({"use_flash_attn": False}) ### flash_attn do not support attention mask

        other_configs = {}
        if model_args.blsp_config:
            with open(model_args.blsp_config, 'r') as file:
                other_configs = json.load(file)
        blsp_config = Blsp2Config(whisper_config.to_dict(), qwen_config.to_dict(), **other_configs)

        model = Blsp2Model(blsp_config)
        model.whisper_model = WhisperEncoder.from_pretrained(model_args.whisper_model)
        model.qwen_model = QWenLMHeadModel.from_pretrained(model_args.qwen_model, config=qwen_config)

    if not model_args.unfreeze_qwen:
        for name, param in model.qwen_model.named_parameters():
            param.requires_grad = False
    else:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules.split(","),
            lora_dropout=model_args.lora_dropout,
            bias="none"
        )
        model.add_lora(lora_config, model_args.lora_scope)

    if not model_args.unfreeze_whisper:
        for name, param in model.whisper_model.named_parameters():
            param.requires_grad = False

    if not model_args.unfreeze_adapter:
        for name, param in model.adapter.named_parameters():
            param.requires_grad = False

    # 6. Define data collator
    data_collator = InstructionDataCollator(
        pad_id=generation_config.pad_token_id,
        sampling_rate=extractor.sampling_rate,
        extractor=extractor
    )


    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 8. Training
    model.set_loss_names(model_args.loss_names.split(","))
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    results = {}
    # 9. Save tokenizer for inference load
    if is_main_process(training_args.local_rank):
        tokenizer.save_pretrained(training_args.output_dir)
        extractor.save_pretrained(training_args.output_dir)
        generation_config.save_pretrained(training_args.output_dir)

    return results

if __name__ == "__main__":
    main()