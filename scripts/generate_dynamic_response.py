#!/usr/bin/env python3
"""
Generate responses using Dynamic AlphaSteer
Uses adaptive λ(x) predicted by gating networks
"""
import os
import argparse
import yaml
import json
import torch
import numpy as np
import time
from transformers import AutoTokenizer, LlamaConfig
from DynamicAlphaSteerModel import DynamicAlphaLlamaForCausalLM, GatingNetwork
from utils.const import AlphaSteer_STEERING_LAYERS
from jinja2 import Template
import logging

torch.manual_seed(42)
np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

template_jinja = """\
Please solve this problem, and put your final answer within \\boxed{}
This is the problem:
{{prompt}}
Please remember to put your final answer within \\boxed{}
"""


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--gating_dir", type=str, default=None, help="Override gating_dir from config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_path)
    for key, value in config.items():
        setattr(args, key, value)
    
    # Override gating_dir if provided via command line
    if args.gating_dir is not None:
        logger.info(f"Using gating_dir from command line: {args.gating_dir}")
    elif not hasattr(args, 'gating_dir'):
        raise ValueError("gating_dir must be specified either in config file or via --gating_dir argument")
    
    logger.info(f"args: {args}")
    
    # Load model configuration
    model_id = "meta-llama/Llama-3.1-8B-Instruct"  # Hardcoded for now
    config_obj = LlamaConfig.from_pretrained(model_id)
    hidden_dim = config_obj.hidden_size
    num_layers = config_obj.num_hidden_layers
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load steering matrix
    if not os.path.exists(args.steering_matrix_path):
        raise ValueError(f"Steering matrix not found: {args.steering_matrix_path}")
    
    steering_matrix = torch.load(args.steering_matrix_path, map_location=args.device)
    steering_matrix = steering_matrix.to(torch.bfloat16)
    logger.info(f"Loaded steering matrix from {args.steering_matrix_path}")
    
    # Load gating networks
    gating_dir = args.gating_dir
    steering_layers = AlphaSteer_STEERING_LAYERS[args.model_name]
    
    gating_networks = [None] * num_layers
    for layer in steering_layers:
        gating_path = os.path.join(gating_dir, f"gating_layer_{layer}.pt")
        if os.path.exists(gating_path):
            gating_net = GatingNetwork(hidden_dim)
            gating_net.load_state_dict(torch.load(gating_path, map_location=args.device))
            gating_net.eval()
            gating_networks[layer] = gating_net
            logger.info(f"Loaded gating network for layer {layer}")
        else:
            logger.warning(f"Gating network not found for layer {layer}: {gating_path}")
    
    # Load model with dynamic steering
    model = DynamicAlphaLlamaForCausalLM.from_pretrained(
        model_id,
        device_map=args.device,
        torch_dtype=torch.bfloat16
    )
    
    model.set_steering_parameters(
        steering_matrix=steering_matrix,
        gating_networks=gating_networks,
        base_strength=0.5  # Will be scaled to [-0.5, 0] range at inference
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token = tokenizer.pad_token
    
    # Load input prompts
    with open(args.input_file, "r") as f:
        prompts = json.load(f)
    
    # Output file setup
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    if args.output_file is not None:
        output_file = args.output_file
    else:
        output_file = args.input_file
    
    if hasattr(args, 'file_rename') and args.file_rename:
        output_file = output_file.replace(".json", f"_{timestamp}.json")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logger.info(f"output_file: {output_file}")
    
    if os.path.exists(output_file):
        logger.info(f"output_file already exists: {output_file}, load from it")
        with open(output_file, "r") as f:
            prompts = json.load(f)
    else:
        logger.info(f"output_file does not exist: {output_file}, generate from scratch")
    
    logger.info(f"len(prompts):\t{len(prompts)}")
    
    # Format prompts
    if "gsm8k" in args.input_file or "math" in args.input_file:
        logger.info("gsm8k or math, use template")
        template = Template(template_jinja)
        messages = [{"role": "user", "content": template.render(prompt=prompt[args.prompt_column])} for prompt in prompts]
        formatted_prompts = [
            tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
    else:
        logger.info("use template")
        messages = [{"role": "user", "content": prompt[args.prompt_column]} for prompt in prompts]
        formatted_prompts = [
            tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
    
    total_batches = (len(formatted_prompts) + args.batch_size - 1) // args.batch_size
    
    # Generate with SINGLE dynamic lambda (no strength list needed!)
    response_key = "response_dynamic"
    
    try:
        for i in range(0, len(formatted_prompts), args.batch_size):
            batch_idx = i // args.batch_size + 1
            
            # Skip if already generated
            if prompts[i].get(response_key) is not None:
                logger.info(f"Skipping batch {batch_idx}/{total_batches} (already generated)")
                continue
            
            batch_prompts = formatted_prompts[i:i + args.batch_size]
            batch_inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(args.device)
            
            input_lengths = [len(input_ids) for input_ids in batch_inputs["input_ids"]]
            batch_input_ids = batch_inputs["input_ids"]
            batch_attention_mask = batch_inputs["attention_mask"]
            
            start_time = time.time()
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                temperature=0.0,
            )
            end_time = time.time()
            
            # Process outputs
            for j, output in enumerate(batch_outputs):
                generated_part = output[input_lengths[j]:]
                response = tokenizer.decode(generated_part, skip_special_tokens=True)
                prompts[i + j][response_key] = response
            
            # Free memory
            del batch_outputs, batch_input_ids, batch_attention_mask
            torch.cuda.empty_cache()
            
            time_per_example = (end_time - start_time) / min(args.batch_size, len(formatted_prompts) - i)
            logger.info(f"Processed batch {batch_idx}/{total_batches}, "
                       f"time taken: {end_time - start_time:.2f} seconds, "
                       f"time per example: {time_per_example:.2f} seconds")
        
        # Save results
        with open(output_file, "w") as f:
            json.dump(prompts, f, indent=4)
        logger.info(f"Saved results to {output_file}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        with open(output_file, "w") as f:
            json.dump(prompts, f, indent=4)
        logger.info(f"Progress saved to {output_file}")
        raise e