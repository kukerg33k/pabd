from flask import Flask, render_template, request 
from logging.config import dictConfig
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },         
            "file": {
                "class": "logging.FileHandler",
                "filename": "service/flask.log",
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)

app = Flask(__name__)

model = None
try:
    model = joblib.load('models/rf_model.pkl')
    app.logger.info("Model loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading model: {str(e)}")


# Маршрут для отображения формы
@app.route('/')
def index():
    return render_template('index.html')

# Маршрут для обработки данных формы
@app.route('/api/numbers', methods=['POST'])
def process_numbers():
    # Здесь можно добавить обработку полученных чисел
    # Для примера просто возвращаем их обратно
    data = request.get_json()
    app.logger.info(f'Requst data: {data}')
        
    try:
        area = float(data['area'])
        rooms = int(data['rooms'])
        floor = int(data['floor'])

        if model:
            input_df = pd.DataFrame([{
        'floor': data['floor'], #этаж
        'total_meters': data['area'], #кол-во метров
        'rooms_count': data['rooms'], #кол-во комнат
        'author_type': 'developer' #автор объявления
    }])

            input_df = pd.get_dummies(input_df, columns=['author_type'])

            # Добавляем недостающие колонки, если их нет в новых данных
            expected_columns = model.feature_names_in_
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Упорядочиваем колонки как при обучении
            input_df = input_df[expected_columns]

            predicted_price = model.predict(input_df)[0]
            return {
                'status': 'success',
                'predicted_price': predicted_price
            }
        else:
            return {
                'status': 'error',
                'message': 'Model not loaded'
            }, 500
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }, 500


    # return {'status': 'success', 'data': 'Числа успешно обработаны'}

if __name__ == '__main__':
    app.run(debug=True)
