#!/usr/bin/env python3
"""
Train Gating Networks for Dynamic AlphaSteer
Trains one gating network per steering layer to predict λ(x) ∈ [0, 1]
"""
import torch
import os
import argparse
import logging
from DynamicAlphaSteerModel import train_gating_network
from utils.const import AlphaSteer_CALCULATION_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train gating networks for dynamic steering")
    parser.add_argument("--embedding_dir", type=str, required=True, help="Directory containing embeddings")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., llama3.1)")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save gating networks")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension for MLP")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    
    logger.info(f"Training gating networks for {args.model_name}")
    logger.info(f"Embedding directory: {args.embedding_dir}")
    
    # Get steering layers for this model
    layers_ratio_list = AlphaSteer_CALCULATION_CONFIG[args.model_name]
    steering_layers = [layer for layer, _ in layers_ratio_list]
    logger.info(f"Steering layers: {steering_layers}")
    
    # Load embeddings
    logger.info("Loading embeddings...")
    
    # Benign embeddings
    # Label: 0 (network predicts 0, scaled to λ=0 at inference = no steering)
    H_benign_train = torch.load(
        f"{args.embedding_dir}/embeds_benign_train.pt", 
        map_location=device
    ).float()
    H_coconot_pref = torch.load(
        f"{args.embedding_dir}/embeds_coconot_pref.pt", 
        map_location=device
    ).float()
    H_coconot_original = torch.load(
        f"{args.embedding_dir}/embeds_coconot_original.pt", 
        map_location=device
    ).float()
    
    # Combine benign data
    indices_borderline = torch.randperm(H_coconot_original.size(0))[:4000 - H_coconot_pref.size(0)]
    H_benign = torch.cat([
        H_benign_train,
        H_coconot_original[indices_borderline],
        H_coconot_pref
    ], dim=0).to(device)
    logger.info(f"Benign embeddings shape: {H_benign.shape}")
    
    # Malicious embeddings
    # Label: 1 (network predicts 1, scaled to λ=-0.5 at inference = full steering)
    H_harmful_train = torch.load(
        f"{args.embedding_dir}/embeds_harmful_train_1000.pt", 
        map_location=device
    ).float()
    H_jailbreak_train = torch.load(
        f"{args.embedding_dir}/embeds_jailbreak_train.pt", 
        map_location=device
    ).float()
    
    # Sample jailbreak to balance dataset
    indices = torch.randperm(H_jailbreak_train.size(0))[:1000]
    H_malicious = torch.cat([
        H_harmful_train,
        H_jailbreak_train[indices]
    ], dim=0).to(device)
    logger.info(f"Malicious embeddings shape: {H_malicious.shape}")
    
    # Borderline embeddings
    # Label: 0.5 (network predicts 0.5, scaled to λ=-0.25 at inference = moderate steering)
    remaining_indices = torch.tensor([
        i for i in range(H_coconot_original.size(0)) 
        if i not in indices_borderline
    ])
    H_borderline = H_coconot_original[remaining_indices[:1000]].to(device)
    logger.info(f"Borderline embeddings shape: {H_borderline.shape}")
    
    # Get dimensions
    num_layers = H_benign.shape[1]
    d_model = H_benign.shape[2]
    logger.info(f"Model dimensions: {num_layers} layers, {d_model} hidden dim")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Train one gating network per steering layer
    gating_networks = {}
    for layer in steering_layers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training gating network for layer {layer}")
        logger.info(f"{'='*60}")
        
        # Extract activations for this layer
        benign_layer = H_benign[:, layer, :]
        malicious_layer = H_malicious[:, layer, :]
        borderline_layer = H_borderline[:, layer, :]
        
        # Train gating network
        save_path = os.path.join(args.save_dir, f"gating_layer_{layer}.pt")
        gating_net = train_gating_network(
            d_model=d_model,
            benign_activations=benign_layer,
            malicious_activations=malicious_layer,
            borderline_activations=borderline_layer,
            hidden_dim=args.hidden_dim,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            save_path=save_path
        )
        
        gating_networks[layer] = gating_net
        logger.info(f"Saved gating network for layer {layer} to {save_path}")
    
    logger.info(f"\n{'='*60}")
    logger.info("All gating networks trained successfully!")
    logger.info(f"Saved {len(gating_networks)} gating networks to {args.save_dir}")
    logger.info(f"{'='*60}")
    
    # Cleanup
    H_benign = None
    H_malicious = None
    H_borderline = None
    torch.cuda.empty_cache()