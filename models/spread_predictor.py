import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
import logging

class SpreadPredictor(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 1):
        """
        LSTM-based neural network for spread prediction.
        Args:
            input_size: Number of input features (volume, trades, high, low)
            hidden_size: Number of LSTM units
            num_layers: Number of LSTM layers
        """
        super(SpreadPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 3)  # 3 outputs: take-profit, bid-loss, stop-loss spreads
        self.softplus = nn.Softplus()  # Ensure positive spread values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, features]
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch, seq_len, hidden_size]
        # Use the last output in the sequence
        last_out = lstm_out[:, -1, :]  # [batch, hidden_size]
        out = self.dropout(last_out)
        out = self.fc(out)
        out = self.softplus(out)
        return out

class SpreadDataProcessor:
    def __init__(self, sequence_length: int = 10):
        """
        Initialize the data processor.
        Args:
            sequence_length: Number of time steps to use for prediction
        """
        self.sequence_length = sequence_length

    def prepare_data(self, candle_data: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare candle data for training.
        Args:
            candle_data: List of candle dictionaries from Hyperliquid API
        Returns:
            Tuple of (X, y) tensors for training
            X: Market activity and volatility features
            y: Optimal spread parameters (to be determined from historical performance)
        """
        # Extract features
        features = []
        for candle in candle_data:
            # Calculate volatility as (high - low) / low
            volatility = (float(candle['h']) - float(candle['l'])) / float(candle['l'])
            features.append([
                float(candle['v']),     # Volume
                float(candle['n']),     # Number of trades
                float(candle['h']),     # High price
                float(candle['l'])      # Low price
            ])
        features = np.array(features)
        
        # Create sequences
        if len(features) < self.sequence_length:
            return torch.FloatTensor([]), torch.FloatTensor([])
        elif len(features) == self.sequence_length:
            # Only one sequence possible
            X = features.reshape(1, self.sequence_length, features.shape[1])
            y = np.array([[0.001, 0.002, 0.003]])  # Placeholder
            return torch.FloatTensor(X), torch.FloatTensor(y)
        else:
            n_sequences = len(features) - self.sequence_length
            X = np.zeros((n_sequences, self.sequence_length, features.shape[1]))
            y = np.zeros((n_sequences, 3))  # 3 spread values
            for i in range(n_sequences):
                X[i] = features[i:i + self.sequence_length]
                # TODO: Calculate optimal spreads based on historical performance
                y[i] = [0.001, 0.002, 0.003]  # Example spreads
            return torch.FloatTensor(X), torch.FloatTensor(y)

def train_model(
    model: SpreadPredictor,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> List[float]:
    """
    Train the neural network model.
    Args:
        model: The neural network model
        X: Input features tensor
        y: Target spread values tensor
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    Returns:
        List of training losses
    """
    logger = logging.getLogger(__name__)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    # Adjust batch size if dataset is too small
    if len(X) < batch_size:
        batch_size = len(X)
    
    n_batches = max(1, len(X) // batch_size)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(X))
            # Get batch
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if (epoch + 1) % 2 == 0:  # Log more frequently
            logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    return losses 