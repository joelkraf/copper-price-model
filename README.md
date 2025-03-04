# copper-price-model
# Copper Price Prediction Model

A sophisticated machine learning system for forecasting copper prices using advanced techniques including 1D-CNN, XGBoost, Sparrow Search Optimization (SSO), and Multi-View Graph Transformer (MVGT).

## Features

- **Multi-Modal Approach**: Combines convolutional neural networks, recurrent neural networks, gradient boosting, and graph transformers
- **Intelligent Hyperparameter Optimization**: Uses Sparrow Search Optimization for automatic model tuning
- **Multi-Horizon Forecasting**: Predicts copper prices across different time ranges (1-day, 7-day, 14-day, 30-day)
- **Trading Signal Generation**: Automatically generates trading recommendations based on forecasts
- **Feature Engineering**: Creates sophisticated technical indicators and integrates macroeconomic factors
- **Model Validation Framework**: Comprehensive testing of model integrity and performance
- **Deployment Ready**: Includes Docker containerization and API service

## Architecture

The system implements a hybrid ensemble model with these key components:

1. **1D-CNN**: Extracts temporal patterns from historical prices and related commodities
2. **LSTM**: Captures long-term dependencies and market memory
3. **XGBoost**: Provides robust prediction with feature importance analysis
4. **Multi-View Graph Transformer (MVGT)**: Models complex relationships between different market indicators

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU recommended for training
- Docker (for containerized deployment)

### Installation

Clone the repository:

```bash
git clone https://github.com/YourUsername/copper-price-prediction.git
cd copper-price-prediction
```

Install required packages:

```bash
pip install -r requirements.txt
```

### Data Preparation

Prepare your data in CSV format with the following columns:
- Date
- HG=F_Close (copper futures closing price)
- Additional related commodities and indicators

Place your data file in the `data` directory.

### Basic Usage

Train and evaluate the model:

```bash
python copper_model.py --data data/copper_data.csv --output model_output
```

Run the complete pipeline with validation:

```bash
python model_deployment.py --mode validate --data data/copper_data.csv
```

Start the prediction API:

```bash
python model_deployment.py --mode deploy --model model_output --data data/copper_data.csv
```

### Docker Deployment

Build and run using Docker:

```bash
docker build -t copper-price-predictor .
docker run -p 5000:5000 -v /path/to/data:/app/data -v /path/to/models:/app/model_output copper-price-predictor
```

## API Reference

The deployed model exposes these endpoints:

- `GET /health`: Check service status
- `GET /predict?horizon=7`: Get prediction for specific horizon
- `GET /forecasts`: Get all available forecasts

## Project Structure

```
copper-price-prediction/
├── copper_model.py           # Core model implementation
├── model_deployment.py       # Validation and API service
├── copper_api_client.py      # Client for interacting with the API
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
├── deploy.sh                 # Deployment script
├── data/                     # Data directory
└── model_output/             # Saved model files
```

## Model Performance

Our hybrid approach significantly outperforms single-model benchmarks:

| Model       | RMSE  | MAE   | R²     |
|-------------|-------|-------|--------|
| CNN         | 0.186 | 0.152 | 0.783  |
| LSTM        | 0.192 | 0.157 | 0.768  |
| XGBoost     | 0.177 | 0.144 | 0.803  |
| **Ensemble**| **0.163** | **0.138** | **0.842** |


## Acknowledgments

- Implementation inspired by academic research on Multi-View Graph Transformers
- XGBoost optimization techniques based on Sparrow Search Algorithm literature
