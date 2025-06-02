import logging
from logging.config import dictConfig
from joblib import load
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


TRAIN_SIZE = 0.8
MODEL_NAME = "rf_model6.pkl"
BASE_DATA_PATH = "../data"

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

model = load(f'../models/{MODEL_NAME}')

def test_model(model):
    """Test model with new data"""
    try:
      logger.info("Evaluating model performance")
      test_data = pd.read_csv(f'{BASE_DATA_PATH}/processed/test.csv')
      X_test = test_data.drop(columns=['price'])
      y_test = test_data['price']

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

if __name__ == "__main__":
    metrics = test_model(model)