import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
import os
import pickle
import tensorflow as tf
import xgboost as xgb
import optuna
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Conv1D, BatchNormalization
from tensorflow.keras.layers import MaxPooling1D, GlobalAveragePooling1D, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import networkx as nx
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("copper_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Error handling decorator
def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

class CopperPricePredictionModel:
    """
    A comprehensive model for copper price prediction using a hybrid approach of
    1D-CNN, XGBoost, SSO (Sparrow Search Optimization), and MVGT (Multi-view Graph Transformer).
    """
    
    def __init__(self, data_path=None, lookback_days=60, forecast_horizons=[1, 7, 14, 30]):
        """
        Initialize the copper price prediction model.
        
        Args:
            data_path (str): Path to the CSV file containing the data.
            lookback_days (int): Number of days to look back for prediction.
            forecast_horizons (list): List of horizons (in days) for forecasting.
        """
        self.data_path = data_path
        self.lookback_days = lookback_days
        self.forecast_horizons = forecast_horizons
        self.target_column = 'HG=F_Close'  # Default target for copper futures
        
        # Initialize components
        self.data = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.cnn_model = None
        self.lstm_model = None
        self.xgb_model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_names = None
        self.train_dates = self.val_dates = self.test_dates = None
        self.sequence_data = {}
        
        # Initialize model parameters
        self.cnn_params = {
            'filters': [64, 128, 256],
            'kernel_size': 3,
            'pool_size': 2,
            'dense_units': 128,
            'dropout_rate': 0.2
        }
        
        self.lstm_params = {
            'units': [128, 64],
            'dropout_rate': 0.2,
            'recurrent_dropout': 0.2
        }
        
        self.xgb_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror'
        }
        
        # Create output directory
        self.output_dir = os.path.join(os.getcwd(), 'copper_model_output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Check system compatibility
        self._check_system_compatibility()
    
    @error_handler
    def _check_system_compatibility(self):
        """Check if the system is compatible with the model requirements."""
        logger.info(f"Python version: {os.sys.version}")
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"XGBoost version: {xgb.__version__}")
        logger.info(f"NumPy version: {np.__version__}")
        logger.info(f"Pandas version: {pd.__version__}")
        
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU(s) detected: {len(gpus)}")
            for gpu in gpus:
                logger.info(f"  {gpu}")
            # Set memory growth to avoid OOM errors
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logger.warning(f"Error setting memory growth: {e}")
        else:
            logger.warning("No GPU detected. Model will run on CPU, which may be significantly slower.")
    
    @error_handler
    def load_data(self, data_path=None):
        """
        Load and preprocess data from CSV.
        
        Args:
            data_path (str, optional): Path to the CSV file. If None, uses the path specified during initialization.
        """
        if data_path is not None:
            self.data_path = data_path
        
        if self.data_path is None:
            raise ValueError("No data path specified.")
        
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            # Load data
            self.data = pd.read_csv(self.data_path)
            
            # Basic data validation
            if self.data.empty:
                raise ValueError("Loaded data is empty.")
            
            # Convert date column to datetime
            if 'Date' in self.data.columns:
                self.data['Date'] = pd.to_datetime(self.data['Date'])
                self.data.set_index('Date', inplace=True)
            
            # Sort by date
            self.data.sort_index(inplace=True)
            
            # Check for missing values
            missing_values = self.data.isnull().sum()
            missing_percent = (missing_values / len(self.data)) * 100
            
            if missing_values.sum() > 0:
                logger.warning(f"Missing values detected:\n{pd.concat([missing_values, missing_percent], axis=1, keys=['Count', 'Percent'])}")
                
                # Forward fill missing values (simple imputation)
                self.data.fillna(method='ffill', inplace=True)
                
                # Check if any missing values remain
                remaining_missing = self.data.isnull().sum().sum()
                if remaining_missing > 0:
                    logger.warning(f"{remaining_missing} missing values remain after forward fill. Using backward fill.")
                    self.data.fillna(method='bfill', inplace=True)
            
            # Check if target column exists
            if self.target_column not in self.data.columns:
                available_columns = ", ".join(self.data.columns)
                raise ValueError(f"Target column '{self.target_column}' not found in data. Available columns: {available_columns}")
            
            # Display data information
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            logger.info(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
            logger.info(f"Columns: {', '.join(self.data.columns)}")
            
            # Add basic technical indicators
            self._add_technical_indicators()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    @error_handler
    def _add_technical_indicators(self):
        """Add technical indicators to the dataset."""
        logger.info("Adding technical indicators...")
        
        # Extract price data for copper
        copper_data = self.data[[self.target_column]].copy()
        
        # Calculate returns
        copper_data['returns'] = copper_data[self.target_column].pct_change()
        copper_data['log_returns'] = np.log(copper_data[self.target_column] / copper_data[self.target_column].shift(1))
        
        # Add moving averages
        for window in [5, 10, 20, 50]:
            copper_data[f'ma_{window}'] = copper_data[self.target_column].rolling(window=window).mean()
            copper_data[f'std_{window}'] = copper_data[self.target_column].rolling(window=window).std()
            copper_data[f'upper_band_{window}'] = copper_data[f'ma_{window}'] + 2 * copper_data[f'std_{window}']
            copper_data[f'lower_band_{window}'] = copper_data[f'ma_{window}'] - 2 * copper_data[f'std_{window}']
            
            # Momentum indicators
            copper_data[f'momentum_{window}'] = copper_data[self.target_column].diff(window)
            copper_data[f'roc_{window}'] = copper_data[self.target_column].pct_change(window) * 100
        
        # Calculate RSI
        def calculate_rsi(data, window=14):
            delta = data.diff()
            gain = delta.mask(delta < 0, 0)
            loss = -delta.mask(delta > 0, 0)
            
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        copper_data['rsi_14'] = calculate_rsi(copper_data[self.target_column])
        
        # Calculate MACD
        copper_data['ema_12'] = copper_data[self.target_column].ewm(span=12, adjust=False).mean()
        copper_data['ema_26'] = copper_data[self.target_column].ewm(span=26, adjust=False).mean()
        copper_data['macd'] = copper_data['ema_12'] - copper_data['ema_26']
        copper_data['macd_signal'] = copper_data['macd'].ewm(span=9, adjust=False).mean()
        copper_data['macd_hist'] = copper_data['macd'] - copper_data['macd_signal']
        
        # Add historical volatility
        for window in [5, 10, 20]:
            copper_data[f'volatility_{window}'] = copper_data['log_returns'].rolling(window=window).std() * np.sqrt(252)
        
        # Add all calculated features to main dataframe
        for col in copper_data.columns:
            if col != self.target_column:
                self.data[col] = copper_data[col]
        
        # Drop rows with NaN values (resulting from rolling windows)
        self.data.dropna(inplace=True)
        
        logger.info(f"Technical indicators added. New shape: {self.data.shape}")
    
    @error_handler
    def prepare_data(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Prepare data for training, including splitting, scaling, and sequence creation.
        
        Args:
            train_ratio (float): Ratio of data to use for training.
            val_ratio (float): Ratio of data to use for validation.
            test_ratio (float): Ratio of data to use for testing.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        if train_ratio + val_ratio + test_ratio != 1.0:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        logger.info("Preparing data for model training...")
        
        # Get all feature columns except the target
        self.feature_names = [col for col in self.data.columns if col != self.target_column]
        
        # Split into features and target
        X = self.data[self.feature_names].values
        y = self.data[self.target_column].values.reshape(-1, 1)
        dates = self.data.index.values
        
        # Scale features and target
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)
        
        # Calculate split indices
        n_samples = len(X_scaled)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        # Split data chronologically
        X_train, y_train = X_scaled[:train_end], y_scaled[:train_end]
        X_val, y_val = X_scaled[train_end:val_end], y_scaled[train_end:val_end]
        X_test, y_test = X_scaled[val_end:], y_scaled[val_end:]
        
        # Save dates for each split
        self.train_dates = dates[:train_end]
        self.val_dates = dates[train_end:val_end]
        self.test_dates = dates[val_end:]
        
        logger.info(f"Data split: Train={X_train.shape}, Validation={X_val.shape}, Test={X_test.shape}")
        
        # Create sequences for time series models
        for horizon in self.forecast_horizons:
            logger.info(f"Creating sequences for horizon {horizon}...")
            X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, lookback=self.lookback_days, horizon=horizon)
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val, lookback=self.lookback_days, horizon=horizon)
            X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, lookback=self.lookback_days, horizon=horizon)
            
            self.sequence_data[horizon] = {
                'X_train': X_train_seq,
                'y_train': y_train_seq,
                'X_val': X_val_seq,
                'y_val': y_val_seq,
                'X_test': X_test_seq,
                'y_test': y_test_seq
            }
            
            logger.info(f"Sequence shapes for horizon {horizon}: X_train={X_train_seq.shape}, y_train={y_train_seq.shape}")
        
        # Save flat data for XGBoost
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        logger.info("Data preparation completed successfully.")
        
        return True
    
    @error_handler
    def _create_sequences(self, X, y, lookback, horizon=1):
        """
        Create sequences for time series modeling.
        
        Args:
            X (np.ndarray): Feature array.
            y (np.ndarray): Target array.
            lookback (int): Number of time steps to look back.
            horizon (int): Number of time steps ahead to predict.
            
        Returns:
            tuple: (X_seq, y_seq) sequences for model training.
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - lookback - horizon + 1):
            X_seq.append(X[i:i+lookback])
            y_seq.append(y[i+lookback+horizon-1])
        
        return np.array(X_seq), np.array(y_seq)
    
    @error_handler
    def build_cnn_model(self):
        """
        Build and compile the 1D-CNN model for feature extraction.
        
        Returns:
            tf.keras.Model: Compiled CNN model.
        """
        logger.info("Building 1D-CNN model...")
        
        # Get input shape from training data
        input_shape = self.sequence_data[self.forecast_horizons[0]]['X_train'].shape[1:]
        
        # Define model architecture
        inputs = Input(shape=input_shape)
        
        # Multiple convolutional layers with increasing complexity
        x = Conv1D(filters=self.cnn_params['filters'][0], 
                  kernel_size=self.cnn_params['kernel_size'], 
                  activation='relu', 
                  padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=self.cnn_params['pool_size'])(x)
        x = Dropout(self.cnn_params['dropout_rate'])(x)
        
        x = Conv1D(filters=self.cnn_params['filters'][1], 
                  kernel_size=self.cnn_params['kernel_size'], 
                  activation='relu', 
                  padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=self.cnn_params['pool_size'])(x)
        x = Dropout(self.cnn_params['dropout_rate'])(x)
        
        x = Conv1D(filters=self.cnn_params['filters'][2], 
                  kernel_size=self.cnn_params['kernel_size'], 
                  activation='relu', 
                  padding='same')(x)
        x = BatchNormalization()(x)
        
        # Global average pooling to reduce dimensionality
        x = GlobalAveragePooling1D()(x)
        
        # Dense layer for feature representation
        x = Dense(self.cnn_params['dense_units'], activation='relu')(x)
        x = Dropout(self.cnn_params['dropout_rate'])(x)
        
        # Output layer
        outputs = Dense(1)(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='mean_squared_error')
        
        logger.info(f"1D-CNN model summary: {model.summary()}")
        self.cnn_model = model
        
        return model
    
    @error_handler
    def build_lstm_model(self):
        """
        Build and compile the LSTM model.
        
        Returns:
            tf.keras.Model: Compiled LSTM model.
        """
        logger.info("Building LSTM model...")
        
        # Get input shape from training data
        input_shape = self.sequence_data[self.forecast_horizons[0]]['X_train'].shape[1:]
        
        # Create sequential model
        model = Sequential()
        
        # LSTM layers
        model.add(LSTM(units=self.lstm_params['units'][0], 
                       return_sequences=True, 
                       dropout=self.lstm_params['dropout_rate'],
                       recurrent_dropout=self.lstm_params['recurrent_dropout'],
                       input_shape=input_shape))
        
        model.add(LSTM(units=self.lstm_params['units'][1], 
                       dropout=self.lstm_params['dropout_rate'],
                       recurrent_dropout=self.lstm_params['recurrent_dropout']))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='mean_squared_error')
        
        logger.info(f"LSTM model summary: {model.summary()}")
        self.lstm_model = model
        
        return model
    
    @error_handler
    def build_mvgt_features(self, X):
        """
        Build Multi-View Graph Transformer features.
        
        Args:
            X (np.ndarray): Input features.
            
        Returns:
            np.ndarray: Transformed features.
        """
        logger.info("Building MVGT features...")
        
        # Create correlation matrix
        if len(X.shape) == 3:
            # For sequence data, use the last time step
            X_flat = X[:, -1, :]
        else:
            X_flat = X
            
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X_flat.T)
        
        # Create graph from correlation matrix
        G = nx.from_numpy_array(np.abs(corr_matrix))
        
        # Get graph properties
        centrality = np.array(list(nx.eigenvector_centrality(G, max_iter=1000).values()))
        clustering = np.array(list(nx.clustering(G).values()))
        
        # Create graph features
        graph_features = np.column_stack([centrality, clustering])
        
        # Replicate for each sample
        graph_features_expanded = np.tile(graph_features, (X_flat.shape[0], 1))
        
        logger.info(f"MVGT features shape: {graph_features_expanded.shape}")
        
        return graph_features_expanded
    
    @error_handler
    def optimize_xgboost_with_sso(self, X_train, y_train, X_val, y_val, n_trials=100):
        """
        Optimize XGBoost hyperparameters using Sparrow Search Optimization (SSO) via Optuna.
        
        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation targets.
            n_trials (int): Number of optimization trials.
            
        Returns:
            dict: Optimized hyperparameters.
        """
        logger.info("Optimizing XGBoost with SSO (via Optuna)...")
        
        def objective(trial):
            param = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
                'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True)
            }
            
            if param['booster'] == 'gbtree' or param['booster'] == 'dart':
                param['max_depth'] = trial.suggest_int('max_depth', 3, 9)
                param['eta'] = trial.suggest_float('eta', 0.01, 0.3, log=True)
                param['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
                param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
                param['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
                param['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 1.0)
            
            if param['booster'] == 'dart':
                param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
                param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
                param['rate_drop'] = trial.suggest_float('rate_drop', 0.0, 0.5)
                param['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 0.5)
            
            # Train model with early stopping
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            model = xgb.train(param, dtrain, 
                              num_boost_round=1000,
                              evals=[(dval, 'validation')],
                              early_stopping_rounds=50,
                              verbose_eval=False)
            
            preds = model.predict(dval)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            
            return rmse
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best XGBoost parameters: {study.best_params}")
        logger.info(f"Best RMSE: {study.best_value}")
        
        # Update model parameters
        self.xgb_params.update(study.best_params)
        
        return study.best_params
    
    @error_handler
    def train_xgboost_model(self, X_train=None, y_train=None, X_val=None, y_val=None, optimize=True):
        """
        Train the XGBoost model.
        
        Args:
            X_train (np.ndarray, optional): Training features. If None, uses the internal data.
            y_train (np.ndarray, optional): Training targets. If None, uses the internal data.
            X_val (np.ndarray, optional): Validation features. If None, uses the internal data.
            y_val (np.ndarray, optional): Validation targets. If None, uses the internal data.
            optimize (bool): Whether to optimize hyperparameters.
            
        Returns:
            xgboost.Booster: Trained XGBoost model.
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        if X_val is None:
            X_val = self.X_val
        if y_val is None:
            y_val = self.y_val
        
        # Add MVGT features
        X_train_mvgt = np.hstack([X_train, self.build_mvgt_features(X_train)])
        X_val_mvgt = np.hstack([X_val, self.build_mvgt_features(X_val)])
        
        logger.info(f"Training XGBoost model with data shape: {X_train_mvgt.shape}...")
        
        # Optimize hyperparameters if requested
        if optimize:
            self.optimize_xgboost_with_sso(X_train_mvgt, y_train, X_val_mvgt, y_val)
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train_mvgt, label=y_train)
        dval = xgb.DMatrix(X_val_mvgt, label=y_val)
        
        # Set up parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            **self.xgb_params
        }
        
        # Train model with early stopping
        start_time = time.time()
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        training_time = time.time() - start_time
        
        logger.info(f"XGBoost training completed in {training_time:.2f} seconds")
        
        # Calculate feature importance
        importance = model.get_score(importance_type='gain')
        importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("Top 10 feature importance:")
        for feature, score in importance[:10]:
            logger.info(f"  {feature}: {score}")
        
        self.xgb_model = model
        
        return model
    
    @error_handler
    def train_cnn_model(self, horizon=1, epochs=100, batch_size=32):
        """
        Train the 1D-CNN model.
        
        Args:
            horizon (int): Forecast horizon to train for.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            
        Returns:
            tf.keras.Model: Trained CNN model.
        """
        if horizon not in self.sequence_data:
            raise ValueError(f"No data prepared for horizon {horizon}")
        
        logger.info(f"Training 1D-CNN model for horizon {horizon}...")
        
        # Get sequence data
        X_train = self.sequence_data[horizon]['X_train']
        y_train = self.sequence_data[horizon]['y_train']
        X_val = self.sequence_data[horizon]['X_val']
        y_val = self.sequence_data[horizon]['y_val']
        
        # Build model if not already built
        if self.cnn_model is None:
            self.build_cnn_model()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        # Train model
        start_time = time.time()
        history = self.cnn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start_time
        
        logger.info(f"1D-CNN training completed in {training_time:.2f} seconds")
        
        # Plot training history
        self._plot_training_history(history, f'cnn_h{horizon}')
        
        return self.cnn_model
    
    @error_handler
    def train_lstm_model(self, horizon=1, epochs=100, batch_size=32):
        """
        Train the LSTM model.
        
        Args:
            horizon (int): Forecast horizon to train for.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            
        Returns:
            tf.keras.Model: Trained LSTM model.
        """
        if horizon not in self.sequence_data:
            raise ValueError(f"No data prepared for horizon {horizon}")
        
        logger.info(f"Training LSTM model for horizon {horizon}...")
        
        # Get sequence data
        X_train = self.sequence_data[horizon]['X_train']
        y_train = self.sequence_data[horizon]['y_train']
        X_val = self.sequence_data[horizon]['X_val']
        y_val = self.sequence_data[horizon]['y_val']
        
        # Build model if not already built
        if self.lstm_model is None:
            self.build_lstm_model()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        # Train model
        start_time = time.time()
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start_time
        
        logger.info(f"LSTM training completed in {training_time:.2f} seconds")
        
        # Plot training history
        self._plot_training_history(history, f'lstm_h{horizon}')
        
        return self.lstm_model
    
    @error_handler
    def _plot_training_history(self, history, model_name):
        """
        Plot training history for a model.
        
        Args:
            history (tf.keras.callbacks.History): Training history.
            model_name (str): Name of the model for the plot title.
        """
        plt.figure(figsize=(12, 5))
        
        # Plot training & validation loss values
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{model_name} - Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Plot learning rate if available
        if 'lr' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['lr'])
            plt.title('Learning Rate')
            plt.ylabel('Learning Rate')
            plt.xlabel('Epoch')
            plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{model_name}_history.png'))
        plt.close()
    
    @error_handler
    def extract_cnn_features(self, X, horizon=1):
        """
        Extract features using the trained CNN model.
        
        Args:
            X (np.ndarray): Input data.
            horizon (int): Horizon for which the CNN was trained.
            
        Returns:
            np.ndarray: Extracted features.
        """
        if self.cnn_model is None:
            raise ValueError("CNN model not trained. Call train_cnn_model() first.")
        
        # Create a feature extraction model using the CNN
        feature_layer_name = 'global_average_pooling1d'
        feature_model = Model(
            inputs=self.cnn_model.input,
            outputs=self.cnn_model.get_layer(feature_layer_name).output
        )
        
        # Extract features
        features = feature_model.predict(X)
        
        logger.info(f"Extracted CNN features shape: {features.shape}")
        
        return features
    
    @error_handler
    def create_ensemble_prediction(self, X_seq, X_flat, horizon=1):
        """
        Create an ensemble prediction using CNN, LSTM, and XGBoost models.
        
        Args:
            X_seq (np.ndarray): Sequence data for CNN and LSTM.
            X_flat (np.ndarray): Flat data for XGBoost.
            horizon (int): Forecast horizon.
            
        Returns:
            np.ndarray: Ensemble predictions.
        """
        # Check if models are trained
        if self.cnn_model is None or self.lstm_model is None or self.xgb_model is None:
            raise ValueError("All models must be trained before creating ensemble predictions.")
        
        # Get CNN predictions
        cnn_preds = self.cnn_model.predict(X_seq)
        
        # Get LSTM predictions
        lstm_preds = self.lstm_model.predict(X_seq)
        
        # Add MVGT features to XGBoost input
        X_xgb = np.hstack([X_flat, self.build_mvgt_features(X_flat)])
        
        # Get XGBoost predictions
        dxgb = xgb.DMatrix(X_xgb)
        xgb_preds = self.xgb_model.predict(dxgb).reshape(-1, 1)
        
        # Create ensemble predictions (weighted average)
        # Weights can be optimized based on validation performance
        ensemble_preds = 0.4 * cnn_preds + 0.3 * lstm_preds + 0.3 * xgb_preds
        
        return ensemble_preds
    
    @error_handler
    def evaluate_models(self, horizon=1):
        """
        Evaluate all models on the test set.
        
        Args:
            horizon (int): Forecast horizon to evaluate.
            
        Returns:
            dict: Performance metrics for each model.
        """
        if horizon not in self.sequence_data:
            raise ValueError(f"No data prepared for horizon {horizon}")
        
        logger.info(f"Evaluating models for horizon {horizon}...")
        
        # Get test data
        X_test_seq = self.sequence_data[horizon]['X_test']
        y_test = self.sequence_data[horizon]['y_test']
        
        # Get predictions
        cnn_preds = self.cnn_model.predict(X_test_seq)
        lstm_preds = self.lstm_model.predict(X_test_seq)
        
        # Add MVGT features to XGBoost input
        X_test_xgb = np.hstack([self.X_test, self.build_mvgt_features(self.X_test)])
        dtest = xgb.DMatrix(X_test_xgb)
        xgb_preds = self.xgb_model.predict(dtest).reshape(-1, 1)
        
        # Create ensemble predictions
        ensemble_preds = 0.4 * cnn_preds + 0.3 * lstm_preds + 0.3 * xgb_preds
        
        # Inverse transform predictions and actual values
        y_test_inv = self.target_scaler.inverse_transform(y_test)
        cnn_preds_inv = self.target_scaler.inverse_transform(cnn_preds)
        lstm_preds_inv = self.target_scaler.inverse_transform(lstm_preds)
        xgb_preds_inv = self.target_scaler.inverse_transform(xgb_preds)
        ensemble_preds_inv = self.target_scaler.inverse_transform(ensemble_preds)
        
        # Calculate metrics
        metrics = {}
        for name, preds in [('CNN', cnn_preds_inv), ('LSTM', lstm_preds_inv), 
                           ('XGBoost', xgb_preds_inv), ('Ensemble', ensemble_preds_inv)]:
            rmse = np.sqrt(mean_squared_error(y_test_inv, preds))
            mae = mean_absolute_error(y_test_inv, preds)
            r2 = r2_score(y_test_inv, preds)
            
            metrics[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            logger.info(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # Plot predictions
        self._plot_predictions(y_test_inv, cnn_preds_inv, lstm_preds_inv, 
                              xgb_preds_inv, ensemble_preds_inv, horizon)
        
        return metrics
    
    @error_handler
    def _plot_predictions(self, y_true, cnn_preds, lstm_preds, xgb_preds, ensemble_preds, horizon):
        """
        Plot model predictions against actual values.
        
        Args:
            y_true (np.ndarray): True values.
            cnn_preds (np.ndarray): CNN predictions.
            lstm_preds (np.ndarray): LSTM predictions.
            xgb_preds (np.ndarray): XGBoost predictions.
            ensemble_preds (np.ndarray): Ensemble predictions.
            horizon (int): Forecast horizon.
        """
        plt.figure(figsize=(15, 8))
        
        # Get dates for the test set
        test_dates = self.test_dates[self.lookback_days+horizon-1:]
        
        # Limit to last 100 points for visibility
        if len(test_dates) > 100:
            start_idx = len(test_dates) - 100
            test_dates = test_dates[start_idx:]
            y_true = y_true[start_idx:]
            cnn_preds = cnn_preds[start_idx:]
            lstm_preds = lstm_preds[start_idx:]
            xgb_preds = xgb_preds[start_idx:]
            ensemble_preds = ensemble_preds[start_idx:]
        
        # Plot predictions
        plt.plot(test_dates, y_true, 'k-', label='Actual')
        plt.plot(test_dates, cnn_preds, 'b-', label='CNN')
        plt.plot(test_dates, lstm_preds, 'g-', label='LSTM')
        plt.plot(test_dates, xgb_preds, 'y-', label='XGBoost')
        plt.plot(test_dates, ensemble_preds, 'r-', label='Ensemble')
        
        plt.title(f'Copper Price Predictions (Horizon = {horizon} days)')
        plt.xlabel('Date')
        plt.ylabel('Copper Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'predictions_h{horizon}.png'))
        plt.close()
    
    @error_handler
    def generate_multi_horizon_forecasts(self, n_steps=30):
        """
        Generate forecasts for multiple horizons.
        
        Args:
            n_steps (int): Number of steps to forecast.
            
        Returns:
            dict: Forecasts for each horizon.
        """
        logger.info(f"Generating multi-horizon forecasts for {n_steps} steps...")
        
        # Get the most recent data for forecasting
        X_recent = self.data[self.feature_names].values[-self.lookback_days:]
        X_recent_scaled = self.feature_scaler.transform(X_recent)
        
        # Dictionary to store forecasts
        forecasts = {}
        
        # Generate forecasts for each horizon
        for horizon in self.forecast_horizons:
            # Reshape for CNN and LSTM
            X_seq = np.array([X_recent_scaled])
            
            # Get predictions
            cnn_pred = self.cnn_model.predict(X_seq)
            lstm_pred = self.lstm_model.predict(X_seq)
            
            # Add MVGT features
            X_xgb = np.hstack([X_recent_scaled[-1:], self.build_mvgt_features(X_recent_scaled[-1:])])
            dxgb = xgb.DMatrix(X_xgb)
            xgb_pred = self.xgb_model.predict(dxgb).reshape(-1, 1)
            
            # Create ensemble prediction
            ensemble_pred = 0.4 * cnn_pred + 0.3 * lstm_pred + 0.3 * xgb_pred
            
            # Inverse transform to get actual price
            ensemble_pred_inv = self.target_scaler.inverse_transform(ensemble_pred)
            
            # Store forecast
            forecasts[horizon] = {
                'prediction': ensemble_pred_inv[0, 0],
                'horizon': horizon,
                'date': datetime.now() + timedelta(days=horizon)
            }
        
        # Generate a sequence of forecasts for the n_steps ahead
        sequence_forecast = []
        
        # Use the shortest horizon for iterative forecasting
        min_horizon = min(self.forecast_horizons)
        
        # Current data points
        X_current = X_recent_scaled.copy()
        
        for step in range(n_steps):
            # Reshape for CNN and LSTM
            X_seq = np.array([X_current])
            
            # Get predictions
            cnn_pred = self.cnn_model.predict(X_seq)
            lstm_pred = self.lstm_model.predict(X_seq)
            
            # Add MVGT features
            X_xgb = np.hstack([X_current[-1:], self.build_mvgt_features(X_current[-1:])])
            dxgb = xgb.DMatrix(X_xgb)
            xgb_pred = self.xgb_model.predict(dxgb).reshape(-1, 1)
            
            # Create ensemble prediction
            ensemble_pred = 0.4 * cnn_pred + 0.3 * lstm_pred + 0.3 * xgb_pred
            
            # Inverse transform to get actual price
            ensemble_pred_inv = self.target_scaler.inverse_transform(ensemble_pred)
            
            # Store forecast
            sequence_forecast.append({
                'step': step + 1,
                'prediction': ensemble_pred_inv[0, 0],
                'date': datetime.now() + timedelta(days=step + 1)
            })
            
            # Update X_current for next step prediction (simple approach)
            # In a real application, you would need to update all features
            X_current = np.roll(X_current, -1, axis=0)
            X_current[-1, 0] = ensemble_pred[0, 0]  # Update only the target feature
        
        forecasts['sequence'] = sequence_forecast
        
        # Plot the sequence forecast
        self._plot_sequence_forecast(sequence_forecast)
        
        return forecasts
    
    @error_handler
    def _plot_sequence_forecast(self, sequence_forecast):
        """
        Plot the sequence forecast.
        
        Args:
            sequence_forecast (list): List of sequence forecast dictionaries.
        """
        plt.figure(figsize=(15, 8))
        
        # Extract data
        steps = [f['step'] for f in sequence_forecast]
        predictions = [f['prediction'] for f in sequence_forecast]
        dates = [f['date'] for f in sequence_forecast]
        
        # Get recent actual data for context
        recent_dates = self.data.index[-30:]
        recent_prices = self.data[self.target_column].values[-30:]
        
        # Plot recent actual data
        plt.plot(recent_dates, recent_prices, 'k-', label='Historical')
        
        # Plot forecast
        plt.plot(dates, predictions, 'r-', label='Forecast')
        plt.axvline(x=datetime.now(), color='g', linestyle='--', label='Today')
        
        plt.title('Copper Price 30-Day Forecast')
        plt.xlabel('Date')
        plt.ylabel('Copper Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sequence_forecast.png'))
        plt.close()
    
    @error_handler
    def save_model(self, path=None):
        """
        Save the trained model components.
        
        Args:
            path (str, optional): Directory path to save the model. If None, uses the default output directory.
        """
        if path is None:
            path = self.output_dir
            
        os.makedirs(path, exist_ok=True)
        
        logger.info(f"Saving model to {path}...")
        
        # Save XGBoost model
        if self.xgb_model is not None:
            self.xgb_model.save_model(os.path.join(path, 'xgboost_model.json'))
        
        # Save CNN model
        if self.cnn_model is not None:
            self.cnn_model.save(os.path.join(path, 'cnn_model'))
        
        # Save LSTM model
        if self.lstm_model is not None:
            self.lstm_model.save(os.path.join(path, 'lstm_model'))
        
        # Save scalers
        if self.feature_scaler is not None:
            with open(os.path.join(path, 'feature_scaler.pkl'), 'wb') as f:
                pickle.dump(self.feature_scaler, f)
        
        if self.target_scaler is not None:
            with open(os.path.join(path, 'target_scaler.pkl'), 'wb') as f:
                pickle.dump(self.target_scaler, f)
        
        # Save model parameters
        params = {
            'lookback_days': self.lookback_days,
            'forecast_horizons': self.forecast_horizons,
            'target_column': self.target_column,
            'feature_names': self.feature_names,
            'cnn_params': self.cnn_params,
            'lstm_params': self.lstm_params,
            'xgb_params': self.xgb_params
        }
        
        with open(os.path.join(path, 'model_params.pkl'), 'wb') as f:
            pickle.dump(params, f)
        
        logger.info("Model saved successfully.")
    
    @classmethod
    @error_handler
    def load_model(cls, path):
        """
        Load a saved model.
        
        Args:
            path (str): Directory path where the model is saved.
            
        Returns:
            CopperPricePredictionModel: Loaded model.
        """
        logger.info(f"Loading model from {path}...")
        
        # Create new instance
        model = cls()
        
        # Load model parameters
        with open(os.path.join(path, 'model_params.pkl'), 'rb') as f:
            params = pickle.load(f)
        
        model.lookback_days = params['lookback_days']
        model.forecast_horizons = params['forecast_horizons']
        model.target_column = params['target_column']
        model.feature_names = params['feature_names']
        model.cnn_params = params['cnn_params']
        model.lstm_params = params['lstm_params']
        model.xgb_params = params['xgb_params']
        
        # Load scalers
        with open(os.path.join(path, 'feature_scaler.pkl'), 'rb') as f:
            model.feature_scaler = pickle.load(f)
        
        with open(os.path.join(path, 'target_scaler.pkl'), 'rb') as f:
            model.target_scaler = pickle.load(f)
        
        # Load XGBoost model
        model.xgb_model = xgb.Booster()
        model.xgb_model.load_model(os.path.join(path, 'xgboost_model.json'))
        
        # Load CNN model
        model.cnn_model = load_model(os.path.join(path, 'cnn_model'))
        
        # Load LSTM model
        model.lstm_model = load_model(os.path.join(path, 'lstm_model'))
        
        logger.info("Model loaded successfully.")
        
        return model
    
    @error_handler
    def run_full_pipeline(self, data_path, train=True, evaluate=True, forecast=True):
        """
        Run the full model pipeline.
        
        Args:
            data_path (str): Path to the data CSV file.
            train (bool): Whether to train the models.
            evaluate (bool): Whether to evaluate the models.
            forecast (bool): Whether to generate forecasts.
            
        Returns:
            dict: Results including evaluation metrics and forecasts.
        """
        results = {}
        
        # 1. Load and prepare data
        logger.info("Step 1: Loading and preparing data...")
        self.load_data(data_path)
        self.prepare_data()
        
        # 2. Train models
        if train:
            logger.info("Step 2: Training models...")
            # Train for the smallest horizon first
            min_horizon = min(self.forecast_horizons)
            
            # Train CNN
            logger.info("Training CNN model...")
            self.train_cnn_model(horizon=min_horizon)
            
            # Train LSTM
            logger.info("Training LSTM model...")
            self.train_lstm_model(horizon=min_horizon)
            
            # Train XGBoost with SSO
            logger.info("Training XGBoost model with SSO...")
            self.train_xgboost_model(optimize=True)
            
            # Save the trained models
            logger.info("Saving trained models...")
            self.save_model()
        
        # 3. Evaluate models
        if evaluate:
            logger.info("Step 3: Evaluating models...")
            evaluation_metrics = {}
            
            for horizon in self.forecast_horizons:
                metrics = self.evaluate_models(horizon=horizon)
                evaluation_metrics[f'horizon_{horizon}'] = metrics
            
            results['evaluation'] = evaluation_metrics
        
        # 4. Generate forecasts
        if forecast:
            logger.info("Step 4: Generating forecasts...")
            forecasts = self.generate_multi_horizon_forecasts()
            results['forecasts'] = forecasts
            
            # Print forecast summary
            for horizon in self.forecast_horizons:
                forecast = forecasts[horizon]
                logger.info(f"{horizon}-day forecast: {forecast['prediction']:.2f} on {forecast['date'].strftime('%Y-%m-%d')}")
            
            logger.info("30-day sequence forecast:")
            for step, forecast in enumerate(forecasts['sequence']):
                if step % 5 == 0 or step == len(forecasts['sequence']) - 1:  # Print every 5 days and the last day
                    logger.info(f"  Day {forecast['step']}: {forecast['prediction']:.2f} on {forecast['date'].strftime('%Y-%m-%d')}")
        
        logger.info("Pipeline execution completed successfully.")
        return results

def main():
    """Main function to run the model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Copper Price Prediction Model')
    parser.add_argument('--data', type=str, required=True, help='Path to the data CSV file')
    parser.add_argument('--lookback', type=int, default=60, help='Number of days to look back')
    parser.add_argument('--horizons', type=str, default='1,7,14,30', help='Comma-separated list of forecast horizons')
    parser.add_argument('--no-train', action='store_true', help='Skip training')
    parser.add_argument('--no-eval', action='store_true', help='Skip evaluation')
    parser.add_argument('--no-forecast', action='store_true', help='Skip forecasting')
    parser.add_argument('--load-model', type=str, help='Path to load a saved model')
    parser.add_argument('--output', type=str, default='./copper_model_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Parse horizons
    horizons = [int(h) for h in args.horizons.split(',')]
    
    try:
        # Create or load model
        if args.load_model:
            model = CopperPricePredictionModel.load_model(args.load_model)
        else:
            model = CopperPricePredictionModel(lookback_days=args.lookback, forecast_horizons=horizons)
            model.output_dir = args.output
        
        # Run pipeline
        results = model.run_full_pipeline(
            data_path=args.data,
            train=not args.no_train,
            evaluate=not args.no_eval,
            forecast=not args.no_forecast
        )
        
        # Print summary
        if 'evaluation' in results:
            logger.info("===== Model Performance Summary =====")
            for horizon, metrics in results['evaluation'].items():
                logger.info(f"\nHorizon: {horizon}")
                for model_name, model_metrics in metrics.items():
                    rmse = model_metrics['RMSE']
                    mae = model_metrics['MAE']
                    r2 = model_metrics['R2']
                    logger.info(f"  {model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        
        logger.info("Model execution completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
