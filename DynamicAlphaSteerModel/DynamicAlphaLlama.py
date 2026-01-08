"""
Dynamic AlphaSteer Model with Adaptive Lambda Prediction
Combines null-space projection with input-dependent steering strength
"""
import logging
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from utils.mask_utils import get_last_valid_token_index
from .GatingNetwork import GatingNetwork

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

__all__ = ['DynamicAlphaLlamaForCausalLM']


class DynamicAlphaLlamaDecoderLayer(LlamaDecoderLayer):
    """
    Llama decoder layer with dynamic adaptive steering
    Uses gating network to predict λ(x) for each input
    """
    def __init__(
        self, 
        config: LlamaConfig, 
        layer_idx: int, 
        steering_matrix: Optional[torch.Tensor] = None,
        gating_network: Optional[GatingNetwork] = None,
        base_strength: float = 1.0
    ):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        
        device = next(self.parameters()).device
        if steering_matrix is not None:
            self.steering_matrix = steering_matrix.to(device)
        else:
            self.steering_matrix = None
        
        self.gating_network = gating_network
        if self.gating_network is not None:
            self.gating_network = self.gating_network.to(device)
            self.gating_network.eval()  # Always in eval mode during inference
        
        self.base_strength = base_strength
    
    def set_steering_parameters(
        self,
        steering_matrix: Optional[torch.Tensor] = None,
        gating_network: Optional[GatingNetwork] = None,
        base_strength: float = 1.0,
        device: Optional[torch.device] = None
    ):
        device = next(self.parameters()).device if device is None else device
        dtype = next(self.parameters()).dtype  # Get model dtype
        
        if steering_matrix is not None and torch.any(steering_matrix):
            self.steering_matrix = steering_matrix.to(device=device, dtype=dtype)
        
        if gating_network is not None:
            self.gating_network = gating_network.to(device=device, dtype=dtype)
            self.gating_network.eval()
        
        self.base_strength = base_strength
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass with dynamic adaptive steering
        
        Key innovation: h' = h + λ(x) · (∆P h)
        where λ(x) is predicted by the gating network
        """
        # Only apply steering on initial input
        should_apply_steering = (
            hidden_states.shape[1] > 1
            and self.steering_matrix is not None
            and torch.any(self.steering_matrix)
            and self.gating_network is not None
        )
        
        if should_apply_steering:
            # Ensure correct device
            if self.steering_matrix.device != hidden_states.device:
                self.steering_matrix = self.steering_matrix.to(hidden_states.device)
            if self.gating_network is not None:
                self.gating_network = self.gating_network.to(hidden_states.device)
            
            B, T, D = hidden_states.shape
            device = hidden_states.device
            
            # Get last valid token index for each sample
            last_idx = get_last_valid_token_index(
                attention_mask=attention_mask,
                seq_len=T,
                batch_size=B,
                device=device,
            )  # (B,)
            
            batch_idx = torch.arange(B, device=device)
            last_hidden = hidden_states[batch_idx, last_idx, :]  # (B, D)
            
            # Predict adaptive λ(x) using gating network
            # Network outputs [0, 1], we scale to [-base_strength, 0]
            # 0 → no steering (benign), 1 → full negative steering (malicious)
            with torch.no_grad():  # Gating network is frozen during inference
                adaptive_lambda = self.gating_network(last_hidden)  # (B,) in [0, 1]
                adaptive_lambda = -self.base_strength * adaptive_lambda  # Scale to [-base_strength, 0]
            
            # Compute steering vector: ∆P h
            steering_vector = last_hidden @ self.steering_matrix  # (B, D)
            
            # Apply adaptive scaling: λ(x) · (∆P h)
            steering_vector = steering_vector * adaptive_lambda.unsqueeze(1)  # (B, D)
            
            # Add steering to all tokens
            steering_vector = steering_vector.unsqueeze(1)  # (B, 1, D)
            hidden_states = hidden_states + steering_vector
        
        # Standard transformer layer forward pass
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states
        residual = hidden_states
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        
        return outputs


class DynamicAlphaLlamaModel(LlamaModel):
    """Llama model with dynamic adaptive steering layers"""
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([
            DynamicAlphaLlamaDecoderLayer(
                config=config,
                layer_idx=layer_idx,
            )
            for layer_idx in range(config.num_hidden_layers)
        ])
    
    def set_steering_parameters(
        self,
        steering_matrix: Optional[torch.Tensor] = None,
        gating_networks: Optional[list] = None,
        base_strength: float = 1.0,
        device: Optional[torch.device] = None
    ):
        device = next(self.parameters()).device if device is None else device
        dtype = next(self.parameters()).dtype  # Get model dtype
        
        if steering_matrix is not None:
            steering_matrix = steering_matrix.to(device=device, dtype=dtype)
        
        for layer_idx, layer in enumerate(self.layers):
            layer_steering_matrix = None
            layer_gating_network = None
            
            if steering_matrix is not None:
                layer_steering_matrix = steering_matrix[layer_idx]
            
            if gating_networks is not None and layer_idx < len(gating_networks):
                layer_gating_network = gating_networks[layer_idx]
            
            layer.set_steering_parameters(
                steering_matrix=layer_steering_matrix,
                gating_network=layer_gating_network,
                base_strength=base_strength,
                device=device
            )
            torch.cuda.empty_cache()
        
        self.print_steering_parameters()
    
    def print_steering_parameters(self):
        logger.info("Dynamic Steering Parameters:")
        logger.info(f"{'Layer':<10}{'Has Steering':<15}{'Has Gating':<15}{'Base Strength'}")
        logger.info("="*60)
        for layer_idx, layer in enumerate(self.layers):
            has_steering = "Yes" if layer.steering_matrix is not None else "No"
            has_gating = "Yes" if layer.gating_network is not None else "No"
            logger.info(f"{layer_idx:<10}{has_steering:<15}{has_gating:<15}{layer.base_strength:.2f}")


class DynamicAlphaLlamaForCausalLM(LlamaForCausalLM):
    """Llama for causal LM with dynamic adaptive steering"""
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = DynamicAlphaLlamaModel(config=config)
    
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
        *model_args,
        steering_matrix: Optional[torch.Tensor] = None,
        gating_networks: Optional[list] = None,
        base_strength: float = 1.0,
        **kwargs
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.set_steering_parameters(
            steering_matrix=steering_matrix,
            gating_networks=gating_networks,
            base_strength=base_strength
        )
        return model
    
    def set_steering_parameters(
        self,
        steering_matrix: Optional[torch.Tensor] = None,
        gating_networks: Optional[list] = None,
        base_strength: float = 1.0
    ):
        device = next(self.parameters()).device
        if steering_matrix is not None:
            steering_matrix = steering_matrix.to(device)
        
        self.model.set_steering_parameters(
            steering_matrix=steering_matrix,
            gating_networks=gating_networks,
            base_strength=base_strength,
            device=device
        )