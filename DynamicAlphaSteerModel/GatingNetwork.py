"""
Gating Network for Dynamic Lambda Prediction
Predicts λ(x) ∈ [0, 1] which is then scaled to [-base_strength, 0] at inference

Training labels:
- Benign prompts: target = 0 (scales to λ=0, no steering)
- Malicious prompts: target = 1 (scales to λ=-base_strength, full steering)
- Borderline prompts: target = 0.5 (scales to λ=-base_strength/2, moderate steering)

Note: Original AlphaSteer uses negative λ values (0 to -0.5) to steer away from harmful directions.
"""
import torch
import torch.nn as nn
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class GatingNetwork(nn.Module):
    """
    MLP-based gating network that predicts steering strength λ(x) ∈ [0, 1]
    
    Args:
        d_model: Hidden dimension of the model
        hidden_dim: Hidden dimension of the MLP (default: 512)
        dropout_rate: Dropout rate for regularization (default: 0.1)
    """
    def __init__(self, d_model: int, hidden_dim: int = 512, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        logger.info(f"Initialized GatingNetwork with d_model={d_model}, hidden_dim={hidden_dim}")
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through gating network
        
        Args:
            h: Hidden states of shape (batch_size, d_model)
            
        Returns:
            lambda_values: Predicted λ(x) values of shape (batch_size,)
        """
        return self.mlp(h).squeeze(-1)  # (batch_size,)


def train_gating_network(
    d_model: int,
    benign_activations: torch.Tensor,
    malicious_activations: torch.Tensor,
    borderline_activations: torch.Tensor = None,
    hidden_dim: int = 512,
    dropout_rate: float = 0.1,
    num_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = 'cuda',
    save_path: str = None
) -> GatingNetwork:
    """
    Train a gating network to predict λ(x) ∈ [0, 1]
    
    The network outputs [0, 1], which is scaled at inference:
    - λ_actual = -base_strength * network_output
    - So λ_actual ∈ [-base_strength, 0]
    
    Args:
        d_model: Hidden dimension
        benign_activations: Benign prompt activations (N_b, d_model), target=0
        malicious_activations: Malicious prompt activations (N_m, d_model), target=1
        borderline_activations: Optional borderline activations (N_border, d_model), target=0.5
        hidden_dim: MLP hidden dimension
        dropout_rate: Dropout rate
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: L2 regularization
        device: Device to train on
        save_path: Path to save the trained model
        
    Returns:
        Trained GatingNetwork
    """
    # Initialize model
    model = GatingNetwork(d_model, hidden_dim, dropout_rate).to(device)
    
    # Prepare training data
    X_benign = benign_activations.to(device)
    y_benign = torch.zeros(len(X_benign), device=device)
    
    X_malicious = malicious_activations.to(device)
    y_malicious = torch.ones(len(X_malicious), device=device)
    
    # Combine data
    X_list = [X_benign, X_malicious]
    y_list = [y_benign, y_malicious]
    
    if borderline_activations is not None:
        X_borderline = borderline_activations.to(device)
        y_borderline = torch.full((len(X_borderline),), 0.5, device=device)
        X_list.append(X_borderline)
        y_list.append(y_borderline)
    
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    # Shuffle data
    indices = torch.randperm(len(X))
    X, y = X[indices], y[indices]
    
    # Split train/val
    n_train = int(0.9 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()  # Binary Cross-Entropy for λ ∈ [0, 1]
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_train_loss / num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)
        model.train()
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}")
        
        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Save model
    if save_path:
        torch.save(model.state_dict(), save_path)
        logger.info(f"Saved gating network to {save_path}")
    
    return model