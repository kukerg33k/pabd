import argparse

import logging
from logging.config import dictConfig

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

TRAIN_SIZE = 0.8
MODEL_NAME = "rf_model6.pkl"
BASE_DATA_PATH = "../data"

import os
os.makedirs("../models", exist_ok=True)  # Автоматическое создание папки

# Настройка логирования
logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': f'model_metrics.log',
            'formatter': 'verbose',
            'level': 'DEBUG',
            'mode': 'w'
        }
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
}

# Применяем конфигурацию
dictConfig(logging_config)
logger = logging.getLogger(__name__)

def train_model():
    """Train model and save with MODEL_NAME"""
    try:
      logger.info("Training Linear Regression model")
      train_data = pd.read_csv(f'{BASE_DATA_PATH}/processed/train.csv')
      X_train = train_data.drop(columns=['price'])
      y_train = train_data['price']
      rf_mod = RandomForestRegressor(n_estimators=250, max_depth=10,random_state=42)
      model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', rf_mod)
    ])
      model.fit(X_train, y_train)
      logger.info("Model training completed")
      return model

    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

def save_model(model,model_name=MODEL_NAME):
    """Save trained model to disk"""
    try:
        model_path = f'../models/{model_name}'
        with open(model_path, 'wb') as f:
            joblib.dump(model, f)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise

if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--split",
        type=float,
        help="Split data, test relative size, from 0 to 1",
        default=TRAIN_SIZE,
    )   
    parser.add_argument("-m", "--model", help="Model name", default=MODEL_NAME)
    args = parser.parse_args()

    model = train_model()
    save_model(model,args.model)