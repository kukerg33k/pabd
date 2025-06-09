from flask import Flask, render_template, request, jsonify
from logging.config import dictConfig
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

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

CORS(app)  # Отключаем CORS проверку д  ля всех маршрутов
auth = HTTPBasicAuth()

users = {
    "admin": generate_password_hash("admin")
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username


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
@auth.login_required
def process_numbers():
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
            return jsonify({
                'status': 'success',
                'predicted_price': predicted_price,
                'user': auth.current_user()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)