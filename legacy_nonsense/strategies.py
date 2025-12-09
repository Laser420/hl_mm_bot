#!/usr/bin/env python3
"""
Volatility-based strategy implementations for dynamic spread and position sizing
"""

import logging
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VolatilityMetrics:
    """Container for volatility-related metrics"""
    realized_vol: float  # Recent realized volatility
    predicted_vol: float  # Model predicted volatility (if applicable)
    confidence: float  # Model confidence score (0-1)
    
    
@dataclass
class StrategyOutput:
    """Container for strategy recommendations"""
    spread_bps: float  # Recommended spread in basis points
    max_position: float  # Recommended maximum position size
    update_interval: float  # Recommended update frequency
    volatility_metrics: VolatilityMetrics


class VolatilityStrategy(ABC):
    """Abstract base class for volatility-based trading strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.price_history: List[float] = []
        self.volatility_history: List[float] = []
        
        # Caching for performance optimization
        self.last_calculation_time = 0.0
        self.last_price_for_calculation = 0.0
        self.cached_strategy_output: Optional[StrategyOutput] = None
        
    @abstractmethod
    def calculate_volatility_metrics(self, candle_data: List[Dict]) -> VolatilityMetrics:
        """Calculate current volatility metrics from candle data"""
        pass
    
    @abstractmethod
    def calculate_strategy_parameters(self, 
                                    current_price: float,
                                    volatility_metrics: VolatilityMetrics,
                                    current_position: float) -> StrategyOutput:
        """Calculate recommended strategy parameters based on volatility"""
        pass
    
    def update_price_history(self, price: float, max_history: int = 1000):
        """Update internal price history for volatility calculations"""
        self.price_history.append(price)
        if len(self.price_history) > max_history:
            self.price_history.pop(0)
    
    def calculate_realized_volatility(self, lookback_minutes: int = 60) -> float:
        """Calculate realized volatility from recent price data"""
        if len(self.price_history) < 2:
            return 0.0
        
        # Use last N prices for volatility calculation
        recent_prices = self.price_history[-lookback_minutes:] if len(self.price_history) >= lookback_minutes else self.price_history
        
        if len(recent_prices) < 2:
            return 0.0
        
        # Calculate log returns
        log_returns = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] > 0 and recent_prices[i] > 0:
                log_returns.append(np.log(recent_prices[i] / recent_prices[i-1]))
        
        if not log_returns:
            return 0.0
        
        # Annualized volatility (assuming 1-minute intervals)
        return np.std(log_returns) * np.sqrt(525600)  # Minutes in a year
    
    def should_recalculate(self, current_price: float, calculation_interval: float, min_price_change: float) -> bool:
        """Check if strategy parameters should be recalculated based on time and price thresholds"""
        current_time = time.time()
        
        # Check time threshold (OR condition 1)
        time_elapsed = current_time - self.last_calculation_time
        if time_elapsed >= calculation_interval:
            return True
        
        # Check price change threshold (OR condition 2)
        if self.last_price_for_calculation > 0:
            price_change_ratio = abs(current_price - self.last_price_for_calculation) / self.last_price_for_calculation
            if price_change_ratio >= min_price_change:
                return True
        
        return False
    
    def update_calculation_cache(self, current_price: float, strategy_output: StrategyOutput):
        """Update cache with new calculation results"""
        self.last_calculation_time = time.time()
        self.last_price_for_calculation = current_price
        self.cached_strategy_output = strategy_output


class LinearVolatilityStrategy(VolatilityStrategy):
    """Simple linear relationship between volatility and spread/position sizing"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Get risk management parameters from trading config
        trading_config = config.get('trading', {})
        strategy_config = config.get('volatility_strategy', {}).get('linear', {})
        
        # Spread scaling parameters
        self.base_spread_bps = trading_config.get('base_spread_bps', 20)
        self.max_spread_bps = trading_config.get('max_spread_bps', 100)
        self.vol_spread_multiplier = strategy_config.get('vol_spread_multiplier', 100)
        
        # Position scaling parameters  
        self.base_position_size = trading_config.get('base_position_size', 0.05)
        self.min_position_size = trading_config.get('min_position_size', 0.01)
        self.vol_position_divisor = strategy_config.get('vol_position_divisor', 2.0)
        
        # Performance configuration
        self.calculation_interval = strategy_config.get('calculation_interval', 30)
        self.min_price_change = strategy_config.get('min_price_change', 0.002)
        
        logger.info(f"LinearVolatilityStrategy initialized - Base spread: {self.base_spread_bps} bps")
        
    def calculate_volatility_metrics(self, candle_data: List[Dict]) -> VolatilityMetrics:
        """Calculate realized volatility from candle data"""
        if not candle_data:
            return VolatilityMetrics(0.0, 0.0, 0.0)
        
        # Extract prices from candle data and update history
        for candle in candle_data[-60:]:  # Last 60 minutes
            close_price = float(candle.get('c', 0))  # Close price
            if close_price > 0:
                self.update_price_history(close_price)
        
        realized_vol = self.calculate_realized_volatility()
        
        # For linear strategy, predicted volatility equals realized
        return VolatilityMetrics(
            realized_vol=realized_vol,
            predicted_vol=realized_vol,
            confidence=1.0  # Always confident in linear model
        )
    
    def calculate_strategy_parameters(self, 
                                    current_price: float,
                                    volatility_metrics: VolatilityMetrics,
                                    current_position: float) -> StrategyOutput:
        """Calculate parameters using linear volatility scaling"""
        
        # Check if recalculation is needed
        if not self.should_recalculate(current_price, self.calculation_interval, self.min_price_change):
            if self.cached_strategy_output is not None:
                logger.debug("Using cached linear strategy parameters")
                return self.cached_strategy_output
        
        vol = volatility_metrics.realized_vol
        
        # Linear spread scaling: higher volatility = wider spreads
        spread_bps = min(
            self.base_spread_bps + (vol * self.vol_spread_multiplier),
            self.max_spread_bps
        )
        
        # Inverse position scaling: higher volatility = smaller positions
        max_position = max(
            self.base_position_size / (1 + vol * self.vol_position_divisor),
            self.min_position_size
        )
        
        logger.debug(f"Linear strategy - Vol: {vol:.4f}, Spread: {spread_bps:.1f} bps, "
                    f"Max pos: {max_position:.4f}")
        
        strategy_output = StrategyOutput(
            spread_bps=spread_bps,
            max_position=max_position,
            update_interval=15.0,  # Fixed interval - not strategy-controlled
            volatility_metrics=volatility_metrics
        )
        
        # Cache the result
        self.update_calculation_cache(current_price, strategy_output)
        
        return strategy_output


class LSTMVolatilityStrategy(VolatilityStrategy):
    """LSTM-based volatility prediction for dynamic strategy parameters"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Get risk management parameters from trading config
        trading_config = config.get('trading', {})
        strategy_config = config.get('volatility_strategy', {}).get('lstm', {})
        
        # Model parameters
        self.lookback_periods = strategy_config.get('lookback_periods', 60)
        self.model_path = strategy_config.get('model_path', 'models/lstm_volatility.pt')
        self.model_retrain_interval = strategy_config.get('model_retrain_interval', 10800)  # Seconds
        
        # Strategy parameters
        self.base_spread_bps = trading_config.get('base_spread_bps', 20)
        self.max_spread_bps = trading_config.get('max_spread_bps', 120)
        self.base_position_size = trading_config.get('base_position_size', 0.05)
        self.min_position_size = trading_config.get('min_position_size', 0.01)
        
        # Performance configuration
        self.calculation_interval = strategy_config.get('calculation_interval', 60)
        self.min_price_change = strategy_config.get('min_price_change', 0.003)
        self.model_prediction_interval = strategy_config.get('model_prediction_interval', 300)
        
        # Model components (lazy loaded)
        self.model = None
        self.scaler = None
        self.last_retrain_time = 0
        self.last_prediction_time = 0
        self.cached_volatility_metrics: Optional[VolatilityMetrics] = None
        
        # Feature engineering
        self.feature_history: List[List[float]] = []
        
        logger.info(f"LSTMVolatilityStrategy initialized - Lookback: {self.lookback_periods} periods")
        
    def _lazy_load_model(self):
        """Lazy load the LSTM model and dependencies"""
        if self.model is not None:
            return
            
        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import MinMaxScaler
            import pickle
            import os
            
            # Define LSTM model architecture
            class VolatilityLSTM(nn.Module):
                def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=1):
                    super(VolatilityLSTM, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                    self.dropout = nn.Dropout(0.2)
                    self.fc = nn.Linear(hidden_size, output_size)
                    
                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.dropout(out[:, -1, :])  # Take last output
                    out = self.fc(out)
                    return out
            
            # Load or create model
            if os.path.exists(self.model_path):
                self.model = torch.load(self.model_path)
                logger.info(f"Loaded LSTM model from {self.model_path}")
            else:
                self.model = VolatilityLSTM()
                logger.warning(f"LSTM model not found at {self.model_path}, using untrained model")
            
            # Load or create scaler
            scaler_path = self.model_path.replace('.pt', '_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded scaler from {scaler_path}")
            else:
                self.scaler = MinMaxScaler()
                logger.warning("Scaler not found, using new MinMaxScaler")
            
            self.model.eval()
            
        except ImportError as e:
            logger.error(f"Missing dependencies for LSTM model: {e}")
            logger.info("Install: pip install torch scikit-learn")
            
        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}")
            self.model = None
    
    def _extract_features(self, candle_data: List[Dict]) -> List[List[float]]:
        """Extract features from candle data for LSTM input"""
        features = []
        
        for i, candle in enumerate(candle_data):
            try:
                # Basic OHLCV features
                open_price = float(candle.get('o', 0))
                high_price = float(candle.get('h', 0))
                low_price = float(candle.get('l', 0))
                close_price = float(candle.get('c', 0))
                volume = float(candle.get('v', 0))
                
                if close_price <= 0:
                    continue
                
                # Calculated features
                price_range = (high_price - low_price) / close_price if close_price > 0 else 0
                volume_weighted_price = volume * close_price if volume > 0 else close_price
                
                # Return calculation (if previous candle exists)
                log_return = 0.0
                if i > 0 and len(candle_data) > i:
                    prev_close = float(candle_data[i-1].get('c', 0))
                    if prev_close > 0:
                        log_return = np.log(close_price / prev_close)
                
                features.append([
                    log_return,
                    price_range,
                    volume_weighted_price / 1000000,  # Scale volume
                    (close_price - open_price) / close_price if close_price > 0 else 0,  # Body ratio
                    (high_price + low_price) / (2 * close_price) if close_price > 0 else 0  # Midpoint ratio
                ])
                
            except (ValueError, KeyError) as e:
                logger.debug(f"Error extracting features from candle {i}: {e}")
                continue
                
        return features
    
    def calculate_volatility_metrics(self, candle_data: List[Dict]) -> VolatilityMetrics:
        """Calculate volatility using LSTM prediction"""
        
        # Calculate realized volatility as baseline
        for candle in candle_data[-60:]:
            close_price = float(candle.get('c', 0))
            if close_price > 0:
                self.update_price_history(close_price)
        
        realized_vol = self.calculate_realized_volatility()
        
        # Check if we should use cached LSTM prediction
        current_time = time.time()
        if (self.cached_volatility_metrics is not None and 
            current_time - self.last_prediction_time < self.model_prediction_interval):
            logger.debug("Using cached LSTM volatility prediction")
            # Update realized volatility but keep cached prediction
            return VolatilityMetrics(
                realized_vol=realized_vol,
                predicted_vol=self.cached_volatility_metrics.predicted_vol,
                confidence=self.cached_volatility_metrics.confidence
            )
        
        # Try LSTM prediction
        try:
            self._lazy_load_model()
            
            if self.model is None:
                # Fall back to realized volatility
                return VolatilityMetrics(realized_vol, realized_vol, 0.5)
            
            # Extract features
            features = self._extract_features(candle_data[-self.lookback_periods:])
            
            if len(features) < self.lookback_periods:
                logger.debug(f"Insufficient features for LSTM: {len(features)} < {self.lookback_periods}")
                return VolatilityMetrics(realized_vol, realized_vol, 0.3)
            
            # Prepare input for model
            import torch
            
            # Use last lookback_periods features
            input_features = np.array(features[-self.lookback_periods:])
            
            # Scale features if scaler is fitted
            if hasattr(self.scaler, 'n_features_in_'):
                input_features = self.scaler.transform(input_features.reshape(-1, input_features.shape[-1])).reshape(input_features.shape)
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.FloatTensor(input_features).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(input_tensor).item()
                predicted_vol = max(0.0, prediction)  # Ensure non-negative
            
            # Calculate confidence based on prediction vs realized volatility
            if realized_vol > 0:
                confidence = 1.0 / (1.0 + abs(predicted_vol - realized_vol) / realized_vol)
            else:
                confidence = 0.7
            
            logger.debug(f"LSTM prediction - Realized: {realized_vol:.4f}, Predicted: {predicted_vol:.4f}, Confidence: {confidence:.2f}")
            
            # Cache the result
            volatility_metrics = VolatilityMetrics(realized_vol, predicted_vol, confidence)
            self.cached_volatility_metrics = volatility_metrics
            self.last_prediction_time = current_time
            
            return volatility_metrics
            
        except Exception as e:
            logger.error(f"Error in LSTM volatility prediction: {e}")
            return VolatilityMetrics(realized_vol, realized_vol, 0.1)
    
    def calculate_strategy_parameters(self, 
                                    current_price: float,
                                    volatility_metrics: VolatilityMetrics,
                                    current_position: float) -> StrategyOutput:
        """Calculate parameters using LSTM volatility prediction"""
        
        # Check if recalculation is needed
        if not self.should_recalculate(current_price, self.calculation_interval, self.min_price_change):
            if self.cached_strategy_output is not None:
                logger.debug("Using cached LSTM strategy parameters")
                return self.cached_strategy_output
        
        # Use predicted volatility if confidence is high, otherwise blend
        if volatility_metrics.confidence > 0.7:
            vol = volatility_metrics.predicted_vol
        else:
            # Blend predicted and realized based on confidence
            vol = (volatility_metrics.confidence * volatility_metrics.predicted_vol + 
                   (1 - volatility_metrics.confidence) * volatility_metrics.realized_vol)
        
        # Non-linear scaling for more responsive adjustments
        vol_factor = np.sqrt(vol)  # Square root to reduce extreme adjustments
        
        # Spread calculation with volatility ceiling
        spread_bps = min(
            self.base_spread_bps * (1 + vol_factor * 3),
            self.max_spread_bps
        )
        
        # Position sizing with exponential decay for high volatility
        position_scale = np.exp(-vol_factor * 2)  # Exponential decay
        max_position = max(
            self.base_position_size * position_scale,
            self.min_position_size
        )
        
        # Update interval: more frequent updates during high volatility
        update_interval = max(10, 30 / (1 + vol_factor * 2))
        
        logger.debug(f"LSTM strategy - Vol: {vol:.4f} (conf: {volatility_metrics.confidence:.2f}), "
                    f"Spread: {spread_bps:.1f} bps, Max pos: {max_position:.4f}")
        
        strategy_output = StrategyOutput(
            spread_bps=spread_bps,
            max_position=max_position,
            update_interval=update_interval,
            volatility_metrics=volatility_metrics
        )
        
        # Cache the result
        self.update_calculation_cache(current_price, strategy_output)
        
        return strategy_output


def create_volatility_strategy(strategy_type: str, config: Dict) -> VolatilityStrategy:
    """Factory function to create volatility strategy instances"""
    
    if strategy_type.lower() == 'linear':
        return LinearVolatilityStrategy(config)
    elif strategy_type.lower() == 'lstm':
        return LSTMVolatilityStrategy(config)
    else:
        logger.warning(f"Unknown strategy type: {strategy_type}, defaulting to linear")
        return LinearVolatilityStrategy(config)