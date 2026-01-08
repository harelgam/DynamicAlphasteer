from transformers import LlamaConfig
from transformers import LlamaForCausalLM
from AlphaSteerModel import *
from NaiveSteerModel import *
from DynamicAlphaSteerModel import DynamicAlphaLlamaForCausalLM

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


__all__ = [
    "MODELS_DICT", "AlphaSteer_MODELS_DICT", "Steer_MODELS_DICT",
    "AlphaSteer_STEERING_LAYERS", "AlphaSteer_CALCULATION_CONFIG",
]

MODELS_DICT = {
    "llama3.1": (LlamaForCausalLM, LlamaConfig, "meta-llama/Llama-3.1-8B-Instruct")
}

AlphaSteer_MODELS_DICT = {
    "llama3.1": (AlphaLlamaForCausalLM, LlamaConfig, "meta-llama/Llama-3.1-8B-Instruct")
}

Steer_MODELS_DICT = {
    "llama3.1": (SteerLlamaForCausalLM, LlamaConfig, "meta-llama/Llama-3.1-8B-Instruct")
}

AlphaSteer_STEERING_LAYERS = {
    "llama3.1": [8, 9, 10, 11, 12, 13, 14, 16, 18, 19]
}

AlphaSteer_CALCULATION_CONFIG = {
    "llama3.1":[(8, 0.6), (9, 0.6), (10, 0.6), (11, 0.6), (12, 0.4), (13, 0.5), (14, 0.6), (16, 0.6), (18, 0.6), (19, 0.6)]}
