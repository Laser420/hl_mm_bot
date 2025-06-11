import torch
import numpy as np
from models.spread_predictor import SpreadPredictor, SpreadDataProcessor, train_model
from utils.data_fetcher import DataFetcher
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_spread_predictor():
    # Initialize data fetcher
    logger.info("Initializing data fetcher...")
    data_fetcher = DataFetcher()
    
    # Fetch recent price data
    logger.info("Fetching recent price data...")
    candle_data = data_fetcher.fetch_price_data()
    logger.info(f"Fetched {len(candle_data)} candles")
    
    # Initialize data processor
    logger.info("Initializing data processor...")
    processor = SpreadDataProcessor(sequence_length=10)
    
    # Prepare data
    logger.info("Preparing data for training...")
    X, y = processor.prepare_data(candle_data)
    logger.info(f"Input shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    # Initialize model
    logger.info("Initializing neural network model...")
    model = SpreadPredictor(input_size=4, hidden_size=64)
    
    # Log model architecture
    logger.info("Model architecture:")
    logger.info(model)
    
    # Train model for 2 epochs with detailed logging
    logger.info("Starting training...")
    losses = train_model(
        model=model,
        X=X,
        y=y,
        epochs=2,  # Just 2 epochs for testing
        batch_size=32,
        learning_rate=0.001
    )
    
    # Test prediction
    logger.info("\nTesting model prediction...")
    with torch.no_grad():
        # Get first sequence
        test_input = X[0:1]  # Shape: [1, sequence_length, features]
        logger.info(f"Test input shape: {test_input.shape}")
        
        # Make prediction
        prediction = model(test_input)
        logger.info(f"Predicted spreads: {prediction.numpy()}")
        logger.info("Spread interpretation:")
        logger.info(f"- Take-profit spread: {prediction[0][0]:.6f}")
        logger.info(f"- Bid-loss spread: {prediction[0][1]:.6f}")
        logger.info(f"- Stop-loss spread: {prediction[0][2]:.6f}")

if __name__ == "__main__":
    test_spread_predictor() 