import logging
from logging.config import dictConfig
import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.8
BASE_DATA_PATH = "../data"

import os
os.makedirs("../data/processed", exist_ok=True)  # Автоматическое создание папки

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

def preprocess(raw_data_path,train_size):
    """Filter and remove"""
    try:
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

      X = df.drop('price', axis=1)
      y = df['price']

      # Train-test split
      X_train, X_test, y_train, y_test = train_test_split(
          X, y, train_size=train_size, random_state=42)

      train_data = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
      test_data = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)

      train_data.to_csv(f'{BASE_DATA_PATH}/processed/train.csv',
                      encoding='utf-8',
                      index=False)
      test_data.to_csv(f'{BASE_DATA_PATH}/processed/test.csv',
                      encoding='utf-8',
                      index=False)

    except Exception as e:
      logger.error(f"Data preprocessing failed: {str(e)}")
      raise

if __name__ == "__main__":
    processed_data = preprocess(f'{BASE_DATA_PATH}/raw/combined.csv',TRAIN_SIZE)