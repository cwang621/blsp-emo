"""BLSP2 config"""

from transformers import PretrainedConfig, WhisperConfig
from transformers import logging
from peft import LoraConfig


from .configuration_qwen import QWenConfig

logger = logging.get_logger(__name__)

class Blsp2Config(PretrainedConfig):
    def __init__(
        self, 
        whisper_config=None, 
        qwen_config=None,
        conv_kernel_sizes="5,5,5",
        adapter_inner_dim=512,
        adapter_hidden_layers=0,
        kd_temperature=2,
        kd_smoothing_weight=0.5,
        lora_config={},
        lora_scope="audio",
        adapter_type="subsampler",  # choose from "subsampler" or "cformer"
        num_pre_cif_layers=4,
        num_post_cif_layers=4,
        num_emotions=5,
        **kwargs
    ):
        super().__init__(**kwargs)

        if whisper_config is None:
            whisper_config = {}
            logger.info("whisper config is None. Initializing the WhisperConfig with default values")

        if qwen_config is None:
            qwen_config = {}
            logger.info("qwen config is None. Initializing the QwenConfig with default values")

        self.whisper_config = WhisperConfig(**whisper_config).to_dict()
        self.qwen_config = QWenConfig(**qwen_config).to_dict()
        self.lora_config = lora_config
        self.lora_scope = lora_scope

        self.conv_kernel_sizes = conv_kernel_sizes
        self.adapter_inner_dim = adapter_inner_dim
        self.adapter_hidden_layers = adapter_hidden_layers
        self.kd_temperature = kd_temperature

        self.adapter_type = adapter_type
        self.num_pre_cif_layers = num_pre_cif_layers
        self.num_post_cif_layers = num_post_cif_layers
        self.num_emotions = num_emotions