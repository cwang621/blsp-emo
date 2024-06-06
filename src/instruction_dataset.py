import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
import fire
import soundfile as sf
import mmap
import io

import torch.distributed as dist
import numpy as np
import torch
import random
import datasets
from datasets import Features, Sequence, Value

from dataclasses import dataclass

from transformers import WhisperFeatureExtractor
try:
    from .tokenization_qwen import QWenTokenizer
except:
    from tokenization_qwen import QWenTokenizer

logger = logging.getLogger(__name__)

emotion2idx = {
    "neutral": 0,
    "happy": 1,
    "angry": 2,
    "sad": 3,
    "surprise": 4
}


feature_schema = Features({
    "start_ids": Sequence(Value("int64")),
    "start_mask": Sequence(Value("int64")),
    "start_labels": Sequence(Value("int64")),
    "instruction_ids": Sequence(Value("int64")),  # Changed from null to int64
    "instruction_mask": Sequence(Value("int64")),  # Changed from null to int64
    "instruction_labels": Sequence(Value("int64")),  # Changed from null to int64
    "audio_instruction_ids": Sequence(Value("int64")),  # Changed from null to int64
    "audio_instruction_mask": Sequence(Value("int64")),  # Changed from null to int64
    "audio_instruction_labels": Sequence(Value("int64")),  # Changed from null to int64
    "input_ids": Sequence(Value("int32")),  # Retained original dtype as int32
    "input_mask": Sequence(Value("int64")),
    "input_labels": Sequence(Value("int64")),
    "suffix_ids": Sequence(Value("int64")),
    "suffix_mask": Sequence(Value("int64")),
    "suffix_labels": Sequence(Value("int64")),
    "emotion_labels": Value("int64"),
    "to_keep": Value("bool"),
    "audio_path": Value("string"),
})


def process_dataset(
    batch,
    tokenizer,
    _tokenize_str,
    instruction="",
    instruction_field="",
    audio_instruction="",
    audio_instruction_field="",
    input_field="input",
    audio_field="audio",
    output_field="output",
    max_length=384,
    min_duration=1.0,
    max_duration=30.0,
    check_audio=True,
    use_emotion=False,
):
    if not input_field and not audio_field:
        raise ValueError(f"neither input_field nor audio_field is set for processing batch: {batch}")
    if not output_field:
        raise ValueError(f"output_field not set for processing batch: {batch}")
    if instruction_field:
        instruction = batch[instruction_field]
    if audio_instruction_field:
        audio_instruction = batch[audio_instruction_field]
    
    im_start_tokens = [tokenizer.im_start_id]
    im_end_tokens = [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")

    if use_emotion:
        system_prompt = "You are a helpful assistant. Your response should fulfill requests with empathy toward user's emotion tone."
    else:
        system_prompt = "You are a helpful assistant."

    start_ids = []
    start_ids += im_start_tokens + _tokenize_str(role="system", content=f"{system_prompt}") + im_end_tokens
    start_ids += nl_tokens
    start_ids += im_start_tokens + _tokenize_str(role="user")
    start_mask = [1] * len(start_ids)
    start_labels = [-100] * len(start_ids)

    instruction_ids, instruction_mask, instruction_labels = [], [], []
    if instruction:
        instruction_ids = _tokenize_str(content=instruction)
        instruction_mask = [1] * len(instruction_ids)
        instruction_labels = [-100] * len(instruction_ids)

    audio_instruction_ids, audio_instruction_mask, audio_instruction_labels = instruction_ids, instruction_mask, instruction_labels
    if audio_instruction:
        audio_instruction_ids = _tokenize_str(content=audio_instruction)
        audio_instruction_mask = [1] * len(audio_instruction_ids)
        audio_instruction_labels = [-100] * len(audio_instruction_ids)

    input_ids, input_mask, input_labels = [], [], []
    if input_field:
        input_ids = _tokenize_str(content=batch[input_field])
        input_mask = [1] * len(input_ids)
        input_labels = [-100] * len(input_ids)


    audio_path = ""
    to_keep = True
    if audio_field:
        audio_path = batch[audio_field]
        if check_audio:
            try:
                waveform = get_waveform(audio_path)
                duration = 1.0 * waveform.shape[0] / 16000
                if duration < min_duration or duration > max_duration:
                    to_keep = False
            except:
                to_keep = False

    suffix_ids, suffix_mask, suffix_labels = [], [], []
    new_ids = im_end_tokens + nl_tokens + im_start_tokens + _tokenize_str(role="assistant")
    suffix_ids += new_ids
    suffix_mask += [1] * len(new_ids)
    suffix_labels += [-100] * len(new_ids)

    early_stop = batch.get("early_stop", False)
    if early_stop:
        new_ids = _tokenize_str(content=batch[output_field])
    else:
        new_ids = _tokenize_str(content=batch[output_field]) + im_end_tokens + nl_tokens + [tokenizer.eod_id]
    suffix_ids += new_ids
    suffix_mask += [1] * len(new_ids)
    suffix_labels += new_ids

    if (len(start_ids) + len(instruction_ids) + len(input_ids) + len(suffix_ids)) > max_length:
        to_keep = False
    
    emotion = batch["emotion"]
    emotion_labels = emotion2idx[emotion]

    batch["start_ids"] = start_ids
    batch["start_mask"] = start_mask
    batch["start_labels"] = start_labels
    batch["instruction_ids"] = instruction_ids
    batch["instruction_mask"] = instruction_mask
    batch["instruction_labels"] = instruction_labels
    batch["audio_instruction_ids"] = audio_instruction_ids
    batch["audio_instruction_mask"] = audio_instruction_mask
    batch["audio_instruction_labels"] = audio_instruction_labels
    batch["input_ids"] = input_ids
    batch["input_mask"] = input_mask
    batch["input_labels"] = input_labels
    batch["suffix_ids"] = suffix_ids
    batch["suffix_mask"] = suffix_mask
    batch["suffix_labels"] = suffix_labels
    batch["emotion_labels"] = emotion_labels

    batch["to_keep"] = to_keep
    if audio_path:
        batch["audio_path"] = audio_path

    return batch

def load_instruction_dataset(
    manifest_dir="",
    manifest_files="",
    tokenizer=None,
    instruction="",
    instruction_field="",
    audio_instruction="",
    audio_instruction_field="",
    input_field="",
    audio_field="",
    output_field="",
    max_length=384,
    min_duration=1.0,
    max_duration=30.0,
    num_proc=8,
    use_emotion=False,
):
    if not manifest_files:
        logger.warning(f"loading processed dataset from {manifest_dir}")
        dataset = datasets.load_from_disk(manifest_dir)
        return dataset
    
    logger.warning(f"load dataset from scratch from {manifest_dir}/{manifest_files}")
    
    manifest_files_list = manifest_files.split(",")

    raw_dataset = datasets.load_dataset(
        manifest_dir, data_files=manifest_files_list, split="train", streaming=False
    )

    def _tokenize_str(role="", content=""):
        tokens = []
        if role:
            tokens += tokenizer.encode(role, allowed_special=set()) + tokenizer.encode("\n")
        if content:
            tokens += tokenizer.encode(content, allowed_special=set())
        return tokens

    dataset = raw_dataset.map(
        process_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "_tokenize_str": _tokenize_str,
            "instruction": instruction,
            "instruction_field": instruction_field,
            "audio_instruction": audio_instruction,
            "audio_instruction_field": audio_instruction_field,
            "input_field": input_field,
            "audio_field": audio_field,
            "output_field": output_field,
            "max_length": max_length,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "use_emotion": use_emotion,
        },
        features=feature_schema,
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=False,
        num_proc=num_proc,
    )

    def to_keep(flag):
        return flag

    dataset = dataset.filter(
        to_keep,
        input_columns=["to_keep"]
    )

    return dataset


def load_instruction_datasets(data_args, tokenizer=None, num_proc=8):
    if os.path.exists(data_args.dataset_save_dir) and os.listdir(data_args.dataset_save_dir):
        logger.warning(f"loading processed dataset from {data_args.dataset_save_dir}")
        dataset = datasets.load_from_disk(data_args.dataset_save_dir)
        return dataset

    manifest_keys = ["manifest_dirs", "manifest_files", "instructions", "instruction_fields",
                     "audio_instructions", "audio_instruction_fields", "input_fields",
                     "audio_fields", "output_fields"]
    if data_args.dataset_dirs:
        dataset_dirs = data_args.dataset_dirs.split("|")
        all_datasets = [load_instruction_dataset(manifest_dir=dataset_dir) for dataset_dir in dataset_dirs]
        num_datasets = len(all_datasets)
    else:
        manifest_values = [(getattr(data_args, key)).split("|") for key in manifest_keys]
        num_datasets = len(manifest_values[0])
        if num_datasets == 0:
            raise ValueError("no datasets specified")
        for i, key in enumerate(manifest_keys):
            if len(manifest_values[i]) != num_datasets:
                raise ValueError(f"unexpected number of {key} in {data_args}")
        all_datasets = [load_instruction_dataset(manifest_dir=manifest_values[0][i],
                                                 manifest_files=manifest_values[1][i],
                                                 instruction=manifest_values[2][i],
                                                 instruction_field=manifest_values[3][i],
                                                 audio_instruction=manifest_values[4][i],
                                                 audio_instruction_field=manifest_values[5][i],
                                                 input_field=manifest_values[6][i],
                                                 audio_field=manifest_values[7][i],
                                                 output_field=manifest_values[8][i],
                                                 tokenizer=tokenizer,
                                                 num_proc=num_proc)
                        for i in range(num_datasets)]
    if len(all_datasets) == 1:
        dataset = all_datasets[0]
    else:
        sample_probs = data_args.sample_probs.split("|")
        if len(sample_probs) == num_datasets:
            sample_probs = [float(prob) for prob in sample_probs]
        else:
            if data_args.sample_probs == "None":
                sample_probs = None
            else:
                raise ValueError(f"unexpected number of probabilities in {data_args}")
        dataset = datasets.interleave_datasets(all_datasets, stopping_strategy=data_args.interleave_stopping_strategy,
                                               probabilities=sample_probs)

    
    if data_args.dataset_save_dir and (not dist.is_initialized() or dist.get_rank() == 0):
        dataset.save_to_disk(data_args.dataset_save_dir)

    return dataset

def collate_tokens(
    values: List[List[int]],
    pad_id: int,
    left_pad: bool = False
):
    size = max(len(v) for v in values)
    batch_size = len(values)
    res = torch.LongTensor(batch_size, size).fill_(pad_id)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        if left_pad:
            copy_tensor(torch.LongTensor(v), res[i][-len(v): ])
        else:
            copy_tensor(torch.LongTensor(v), res[i][: len(v)])

    return res

def mmap_read(path: str, offset: int, length: int) -> bytes:
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_o:
            data = mmap_o[offset : offset + length]
    return data


def read_from_stored_zip(zip_path: str, offset: int, length: int) -> bytes:
    return mmap_read(zip_path, offset, length)

def is_sf_audio_data(data: bytes) -> bool:
    is_wav = data[0] == 82 and data[1] == 73 and data[2] == 70
    is_flac = data[0] == 102 and data[1] == 76 and data[2] == 97
    is_ogg = data[0] == 79 and data[1] == 103 and data[2] == 103
    return is_wav or is_flac or is_ogg


def get_waveform(
    path_or_fp: str,
    normalization=True,
    mono=True,
    frames=-1,
    start=0,
    always_2d=False,
    output_sample_rate=16000,
) -> Tuple[np.ndarray, int]:
    meta = path_or_fp.split(":")
    if len(meta) == 3:
        path_or_fp = meta[0]
        start = int(meta[1])
        frames = int(meta[2])
    else:
        path_or_fp = path_or_fp
    
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile to load WAV/FLACC/OGG/MP3/OPUS audios")

    ext = Path(path_or_fp).suffix
    if ext in [".wav", ".flac", ".ogg", ".mp3", ".opus"]:
        waveform, sample_rate = sf.read(
            path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start
        )
    elif ext in [".zip"]:
        data = read_from_stored_zip(path_or_fp, start, frames)
        assert is_sf_audio_data(data)
        f = io.BytesIO(data)
        waveform, sample_rate = sf.read(
            f, dtype="float32", always_2d=True
        )
    else:
        raise ValueError(f"Unsupported audio format: {ext}")
    
    waveform = waveform.T

    waveform, sample_rate = convert_waveform(waveform, sample_rate, to_mono=mono, to_sample_rate=output_sample_rate)
    if not normalization:
        waveform *= 2 ** 15
    if not always_2d:
        waveform = waveform.squeeze(axis=0)

    return waveform

def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization
    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """
    try:
        import torchaudio.sox_effects as ta_sox
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    effects = []
    if normalize_volume:
        effects.append(["gain", "-n"])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(["rate", f"{to_sample_rate}"])
    if to_mono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])
    if len(effects) > 0:
        is_np_input = isinstance(waveform, np.ndarray)
        _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(
            _waveform, sample_rate, effects
        )
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate
    return waveform, sample_rate


@dataclass
class InstructionDataCollator:
    """
    Data collator that will dynamically pad the inputs received.
    """
    pad_id: int = 0
    sampling_rate: int = 16000
    extractor: WhisperFeatureExtractor = WhisperFeatureExtractor()

    def __call__(self, samples: List[Dict]):
        start_ids = [sample["start_ids"] for sample in samples]
        start_mask = [sample["start_mask"] for sample in samples]
        start_labels = [sample["start_labels"] for sample in samples]
        instruction_ids = [sample["instruction_ids"] for sample in samples]
        instruction_mask = [sample["instruction_mask"] for sample in samples]
        instruction_labels = [sample["instruction_labels"] for sample in samples]
        audio_instruction_ids = [sample["audio_instruction_ids"] for sample in samples]
        audio_instruction_mask = [sample["audio_instruction_mask"] for sample in samples]
        audio_instruction_labels = [sample["audio_instruction_labels"] for sample in samples]
        input_ids = [sample["input_ids"] for sample in samples]
        input_mask = [sample["input_mask"] for sample in samples]
        input_labels = [sample["input_labels"] for sample in samples]
        suffix_ids = [sample["suffix_ids"] for sample in samples]
        suffix_mask = [sample["suffix_mask"] for sample in samples]
        suffix_labels = [sample["suffix_labels"] for sample in samples]
        emotion_labels = [sample["emotion_labels"] for sample in samples]

        start_ids = collate_tokens(start_ids, self.pad_id)
        start_mask = collate_tokens(start_mask, 0)
        start_labels = collate_tokens(start_labels, -100)
        instruction_ids = collate_tokens(instruction_ids, self.pad_id)
        instruction_mask = collate_tokens(instruction_mask, 0)
        instruction_labels = collate_tokens(instruction_labels, -100)
        audio_instruction_ids = collate_tokens(audio_instruction_ids, self.pad_id)
        audio_instruction_mask = collate_tokens(audio_instruction_mask, 0)
        audio_instruction_labels = collate_tokens(audio_instruction_labels, -100)
        input_ids = collate_tokens(input_ids, self.pad_id)
        input_mask = collate_tokens(input_mask, 0)
        input_labels = collate_tokens(input_labels, -100)
        suffix_ids = collate_tokens(suffix_ids, self.pad_id)
        suffix_mask = collate_tokens(suffix_mask, 0)
        suffix_labels = collate_tokens(suffix_labels, -100)
        emotion_labels = torch.LongTensor(emotion_labels)

        raw_speech = [
            get_waveform(sample["audio_path"], output_sample_rate=self.sampling_rate) if 'audio_path' in sample else []
            for sample in samples
        ]
        if all(len(sample) == 0 for sample in raw_speech):
            speech_values = None
            speech_mask = None
        else:
            speech_inputs = self.extractor(
                raw_speech, 
                sampling_rate=self.sampling_rate, 
                return_attention_mask=True,
                return_tensors="pt"
            )
            speech_values = speech_inputs.input_features
            speech_mask = speech_inputs.attention_mask

        return {
            "start_ids": start_ids,
            "start_mask": start_mask,
            "start_labels": start_labels,
            "instruction_ids": instruction_ids,
            "instruction_mask": instruction_mask,
            "instruction_labels": instruction_labels,
            "audio_instruction_ids": audio_instruction_ids,
            "audio_instruction_mask": audio_instruction_mask,
            "audio_instruction_labels": audio_instruction_labels,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "input_labels": input_labels,
            "suffix_ids": suffix_ids,
            "suffix_mask": suffix_mask,
            "suffix_labels": suffix_labels,
            "emotion_labels": emotion_labels,
            "speech_values": speech_values,
            "speech_mask": speech_mask
        }


def offline_process(
    dataroot="",
    manifest_files="",
    lm_path="",
    instruction="",
    instruction_field="",
    audio_instruction="",
    audio_instruction_field="",
    input_field="",
    audio_field="",
    output_field="",
    save_dir="",
    max_length=384,
    min_duration=1.0,
    max_duration=30.0,
    num_proc=8,
    use_emotion=False,
):
    text_tokenizer = QWenTokenizer.from_pretrained(lm_path)

    dataset = load_instruction_dataset(
        dataroot,
        manifest_files,
        text_tokenizer,
        instruction,
        instruction_field,
        audio_instruction,
        audio_instruction_field,
        input_field,
        audio_field,
        output_field,
        max_length,
        min_duration,
        max_duration,
        num_proc,
        use_emotion,
    )
    print(len(dataset))
    for key in dataset[0].keys():
        if key != "audio_path" and key != "to_keep" and key != "emotion_labels":
            print(key, len(dataset[0][key]))
        else:
            print(key, dataset[0][key])
    
    if save_dir:
        dataset.save_to_disk(save_dir)


if __name__ == "__main__":
    fire.Fire({
        "offline": offline_process,
    })