import argparse
import os
import random
import time
import logging
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from gradio import processing_utils
from accelerate import Accelerator

from transformers import WhisperFeatureExtractor
from transformers import GenerationConfig
from src.modeling_blsp2 import Blsp2Model
from src.tokenization_qwen import QWenTokenizer
from src.instruction_dataset import get_waveform
from src.qwen_generation_utils import get_stop_words_ids, decode_tokens

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("Chat Demo")

class ChatHistory(object):
    def __init__(self, 
        tokenizer, 
        extractor, 
        max_window_size=6144,
        max_new_tokens=512,
        use_emotion=False,
        speech_downsample_rate=16
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.max_window_size = max_window_size
        self.max_new_tokens = max_new_tokens
        self.speech_downsample_rate = speech_downsample_rate

        self.im_start_tokens = [tokenizer.im_start_id]
        self.im_end_tokens = [tokenizer.im_end_id]
        self.nl_tokens = tokenizer.encode("\n")

        ### add system
        if use_emotion:
            sys_prompt = "You are a helpful assistant. Your response should fulfill requests with empathy toward user's emotion tone."
        else:
            sys_prompt = "You are a helpful assistant."
        input_ids = self.im_start_tokens + self._tokenize_str("system", f"{sys_prompt}") + self.im_end_tokens
        input_ids = torch.LongTensor([input_ids])
        self.system_histroy = [(input_ids,)]
        self.system_length = input_ids.shape[1]

        self.reset()
    
    def reset(self):
        self.history = []
        self.lengths = []
        self.cur_length = self.system_length
        self.audio_file = []
        self.audio_to_history = True
    
    def _tokenize_str(self, role, content):
        return self.tokenizer.encode(
            role, allowed_special=set()
        ) + self.nl_tokens + self.tokenizer.encode(content, allowed_special=set())

    def add_text_history(self, role, text):
        input_ids =  self.nl_tokens + self.im_start_tokens + self._tokenize_str(role, text) + self.im_end_tokens
        input_ids = torch.LongTensor([input_ids])
        self.history.append(
            (input_ids,)
        )
        self.lengths.append(input_ids.shape[1])
        self.cur_length += input_ids.shape[1]

    def add_audio(self, audio_file):
        self.audio_to_history = False
        self.audio_file.append(audio_file)

    def add_speech_history(self, speech, text=""):
        if self.audio_to_history:
            return
        self.audio_to_history = True
        speech = get_waveform(speech, output_sample_rate=self.extractor.sampling_rate)
        speech_inputs = self.extractor(
            speech,
            sampling_rate=self.extractor.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt"
        )
        speech_values = speech_inputs.input_features.half()
        speech_attention_mask = speech_inputs.attention_mask

        input_ids = self.nl_tokens + self.im_start_tokens + self._tokenize_str("user", text)
        input_ids = torch.LongTensor([input_ids])
        self.history.append(
            (input_ids,)
        )
        self.lengths.append(input_ids.shape[1])
        self.cur_length += input_ids.shape[1]

        self.history.append(
            (speech_values, speech_attention_mask)
        )
        length = speech_attention_mask.sum().item() // self.speech_downsample_rate
        self.lengths.append(length)
        self.cur_length += length
        

        input_ids = [] + self.im_end_tokens
        input_ids = torch.LongTensor([input_ids])
        self.history.append(
            (input_ids,)
        )
        self.lengths.append(input_ids.shape[1])
        self.cur_length += input_ids.shape[1]
    
    def get_history(self):
        input_ids = self.nl_tokens + self.im_start_tokens + self.tokenizer.encode("assistant")
        input_ids = torch.LongTensor([input_ids])
        length = input_ids.shape[1]

        while self.cur_length > (self.max_window_size - self.max_new_tokens - length):
            pop_length = self.lengths.pop(0)
            self.history.pop(0)
            self.cur_length -= pop_length
        return self.system_histroy + self.history + [(input_ids,)]


def parse_args():
    parser = argparse.ArgumentParser(description="Chat Demo")
    parser.add_argument(
        "--blsp_model", type=str, default=None,
        help="Path to the blsp model", required=True
    )
    parser.add_argument(
        "--use_emotion", action="store_true",
        help="Path to the blsp model"
    )
    ### args for generation
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="max new tokens for generation"
    )
    parser.add_argument(
        "--min_new_tokens", type=int, default=1,
        help="min new tokens for generation"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5,
        help="temperature for generation"
    )
    parser.add_argument(
        "--max_window_size", type=int, default=6144,
        help="max length for previous context"
    )
    args = parser.parse_args()
    return args


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()

accelerator = Accelerator()
logger.info(accelerator.state)

device = accelerator.device

tokenizer = QWenTokenizer.from_pretrained(args.blsp_model)
extractor = WhisperFeatureExtractor.from_pretrained(args.blsp_model)
model = Blsp2Model.from_pretrained(args.blsp_model, torch_dtype=torch.float16)
model = model.half()
generation_config = GenerationConfig.from_pretrained(args.blsp_model)
stop_words_ids = get_stop_words_ids(generation_config.chat_format, tokenizer)

generation_config.update(
    **{
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "temperature": args.temperature,
        "max_window_size": args.max_window_size,
        "bos_token_id": tokenizer.encode("\n")[0],
        "num_return_sequences": 1,
    }
)

model = model.to(device)
model.eval()
history = ChatHistory(tokenizer, extractor, generation_config.max_window_size, generation_config.max_new_tokens, args.use_emotion)

print('Initialization Finished')


def gradio_reset():
    history.reset()
    return None, gr.update(value="", interactive=True), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True)


def gradio_answer(chatbot, num_beams, temperature):
    generation_config.update(
        **{
            "num_beams": num_beams, 
            "temperature": temperature,
        }
    )

    output = model.chat(
        history=history.get_history(),
        generation_config=generation_config,
        stop_words_ids=stop_words_ids,
        device=device
    )
    response = decode_tokens(
        output[0],
        tokenizer,
        raw_text_len=0,
        context_length=0,
        chat_format=generation_config.chat_format,
        verbose=False,
        errors='replace'
    )
    history.add_text_history("assistant", response)
    chatbot[-1][1] = ""
    for character in response:
        chatbot[-1][1] += character
        yield chatbot


title = """<h1 align="center">Demo of BLSP2</h1>"""
description = """<h3>This is the demo of BLSP2. Upload your audios and start chatting!</h3>"""
article = """<p><a href='https://xxx.github.io'><img src='https://xxx'></a></p><p><a href='https://github.com/xxx'><img src='https://xxx'></a></p><p><a href='xxx'><img src='xxx'></a></p>
"""


#TODO show examples below


def add_text(chatbot, user_message):
    chatbot = chatbot + [(user_message, None)]
    history.add_text_history("user", user_message)
    return chatbot, gr.update(value="", interactive=False)


def add_file(chatbot, gr_audio):
    history.add_audio(gr_audio.name)
    history.add_speech_history(history.audio_file[-1])
    chatbot = chatbot + [((gr_audio.name,), None)]
    return chatbot


def add_micophone_file(chatbot, gr_audio_mic):
    if gr_audio_mic is not None:
        audio = processing_utils.audio_from_file(gr_audio_mic)
        processing_utils.audio_to_file(audio[0], audio[1], gr_audio_mic + '.wav')
        gr_audio_mic_wav = gr_audio_mic+".wav"
        history.add_audio(gr_audio_mic_wav)
        history.add_speech_history(history.audio_file[-1])
        chatbot = chatbot + [((gr_audio_mic_wav,), None)]
    return chatbot, gr.update(value=None, interactive=True)


with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    # gr.Markdown(article)

    chatbot = gr.Chatbot([], elem_id="chatbot", height=750, avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))))

    with gr.Row():
        with gr.Column(scale=0.2, min_width=0, max_width=400):
            num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    interactive=True,
                    label="beam",
                )
                
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temp",
                )
        with gr.Column(scale=0.08, min_width=0, max_width=10):
            clear = gr.Button("Restart")
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
                container=False)
        with gr.Column(scale=0.08, min_width=0, max_width=10):
            btn = gr.UploadButton("📁", file_types=["video", "audio"])
        with gr.Column(scale=0.2, min_width=0, max_width=400):
            input_audio_mic = gr.Audio(
                label="🎤",
                type="filepath",
                source="microphone",
                visible=True,
            )

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        gradio_answer, [chatbot, num_beams, temperature], chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        gradio_answer, [chatbot, num_beams, temperature], chatbot
    )

    input_audio_mic.change(add_micophone_file, [chatbot, input_audio_mic], [chatbot, input_audio_mic], queue=False).then(
        gradio_answer, [chatbot, num_beams, temperature], chatbot
    )
    clear.click(gradio_reset, [], [chatbot, txt, input_audio_mic, btn], queue=False)

demo.queue()
demo.launch(share=False, enable_queue=True)