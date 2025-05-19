"""This is full life cycle for ml model"""

import argparse

import logging
from logging.config import dictConfig
import datetime
import pickle

import cianparser
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

TRAIN_SIZE = 0.8
MODEL_NAME = "rf_model.pkl"
BASE_DATA_PATH = "data"

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


def parse_cian(location="Москва", start_page=1, end_page=50,rooms=1):
    """Parse data to data/raw"""
    try:
      logger.info(f"Starting data collection for {location}, pages {start_page}-{end_page}")
      parser = cianparser.CianParser(location=location)
      raw_data_path = f'{BASE_DATA_PATH}/raw/data_{datetime.date.today()}_rooms_{rooms}.csv'
      data = parser.get_flats(
          deal_type="sale",
          rooms=(rooms),
          with_saving_csv=False,
          additional_settings={
              "start_page": start_page,
              "end_page": end_page,
          })
      df = pd.DataFrame(data)
      df.to_csv(raw_data_path,
                encoding='utf-8',
                index=False)
      logger.info(f"Data successfully saved to {raw_data_path}")
      return raw_data_path

    except Exception as e:
        logger.error(f"Data collection failed: {str(e)}")
        raise

def preprocess_data(raw_data_path):
    """Filter and remove"""
    try:
    #   processed_path = f'{BASE_DATA_PATH}/processed/processed_data.csv'
      processed_path = f'{BASE_DATA_PATH}/processed/processed_data.csv'
      logger.info(f"Preprocessing data from {raw_data_path}")
      df = pd.read_csv(raw_data_path)
      df['url_id'] = df['url'].map(lambda x: x.split('/')[-2])
      drop_col = ['location','deal_type','accommodation_type','commissions','price_per_month','residential_complex','house_number','author','street','url','district','underground','floors_count']
      df = df.drop(drop_col, axis=1)
      df = df.set_index('url_id')
      df = df.dropna(subset=['price'])
      df['author_type'] = df['author_type'].fillna('unknown')
      price_low = df['price'].quantile(0.05) 
      price_high = df['price'].quantile(0.95) 
      df = df[(df['price'] >= price_low) & (df['price'] <= price_high)]

      price_low = df['total_meters'].quantile(0.05)  
      price_high = df['total_meters'].quantile(0.95) 
      df = df[(df['total_meters'] >= price_low) & (df['total_meters'] <= price_high)]

      df = pd.get_dummies(df, columns=['author_type'], drop_first=True)

      df.to_csv(processed_path,
                encoding='utf-8',
                index=False)
      logger.info(f"Processed data saved to {processed_path}")
      return processed_path

    except Exception as e:
      logger.error(f"Data preprocessing failed: {str(e)}")
      raise

def load_and_prepare_data(processed_path, train_size=TRAIN_SIZE):
    """Load and prepare data for training"""
    try:
        logger.info(f"Loading and preparing data from {processed_path}")
        data = pd.read_csv(processed_path)
        X = data.drop('price', axis=1)
        y = data['price']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=42)

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def train_model(X_train, y_train):
    """Train model and save with MODEL_NAME"""
    try:
      logger.info("Training Linear Regression model")
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

def test_model(model, X_test, y_test):
    """Test model with new data"""
    try:
      logger.info("Evaluating model performance")
      # Предсказание на тестовой выборке
      y_pred = model.predict(X_test)

      # Метрики
      metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mae': np.mean(np.abs(y_test - y_pred)),
        }

      # Создаем логгер
      logger.info("Model Metrics:")
      logger.info(f"MSE: {metrics['mse']:.2f}")
      logger.info(f"RMSE: {metrics['rmse']:.2f}")
      logger.info(f"R2: {metrics['r2']:.4f}")
      logger.info(f"MAE: {metrics['mae']:.2f} рублей")
      pass

    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise

def save_model(model,model_name=MODEL_NAME):
    """Save trained model to disk"""
    try:
        model_path = f'models/{model_name}'
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

    processed_data = preprocess_data(f'{BASE_DATA_PATH}/raw/combined.csv')
    X_train, X_test, y_train, y_test = load_and_prepare_data(processed_data,args.split)
    model = train_model(X_train, y_train)
    metrics = test_model(model, X_test, y_test)
    save_model(model,args.model)