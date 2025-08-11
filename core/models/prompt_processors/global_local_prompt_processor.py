import json
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *

from threestudio.models.prompt_processors.base import PromptProcessorOutput

from core.utils.helper import HF_PATH


@threestudio.register("global-local-prompt-processor")
class GlobalLocalPromptProcessor(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        global_prompt_processor_type: str   = "stable-diffusion-prompt-processor"
        global_prompt_processor: dict       = field(default_factory=dict)
        
        local_prompt_processor_type: str    = "stable-diffusion-prompt-processor"
        local_prompt_processor: dict        = field(default_factory=dict)

        local_prompt_init_strategy: str     = "none"
        local_prompt_init_file_path: str    = ""
        local_additional_prompt: str        = ""
        local_prompt_layout: str            = ""

    cfg: Config

    def configure(self) -> None:
        if hasattr(self.cfg.global_prompt_processor, 'pretrained_model_name_or_path'):
            self.cfg.global_prompt_processor.pretrained_model_name_or_path = HF_PATH(
                self.cfg.global_prompt_processor.pretrained_model_name_or_path
            )
        if hasattr(self.cfg.local_prompt_processor, 'pretrained_model_name_or_path'):
            self.cfg.local_prompt_processor.pretrained_model_name_or_path = HF_PATH(
                self.cfg.local_prompt_processor.pretrained_model_name_or_path
            )

        self.global_prompt_processor = threestudio.find(self.cfg.global_prompt_processor_type)(
            self.cfg.global_prompt_processor
        )

        # load local prompts
        self.local_prompts = []
        if self.cfg.local_prompt_init_strategy == 'none':
            pass
        elif self.cfg.local_prompt_init_strategy == 'from_layout':
            with open(self.cfg.local_prompt_init_file_path, 'r') as f:
                bbox = json.load(f)['bbox']
            for b in bbox:
                text_prompt = b['prompt']
                if self.cfg.local_additional_prompt != '':
                    text_prompt = b['prompt'] + ', ' + self.cfg.local_additional_prompt
                self.local_prompts.append(text_prompt)
            if self.cfg.local_prompt_layout != '':
                text_prompt = self.cfg.local_prompt_layout
                if self.cfg.local_additional_prompt != '':
                    text_prompt = text_prompt + ', ' + self.cfg.local_additional_prompt
                self.local_prompts.append(text_prompt)
        else:
            raise NotImplementedError
        
        # local prompt processors
        self.local_prompt_processors = []
        local_cfg = self.cfg.local_prompt_processor
        for p in self.local_prompts:
            local_cfg.prompt = p
            self.local_prompt_processors.append(
                threestudio.find(self.cfg.local_prompt_processor_type)(
                    local_cfg
                )
            )

    def __call__(self) -> dict:
        ret_dict = {'global': self.global_prompt_processor()}
        if len(self.local_prompt_processors) != 0:
            ret_dict['local'] = [p() for p in self.local_prompt_processors]
        return ret_dict