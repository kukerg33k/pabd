# Используем официальный Python образ
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы проекта
COPY . .

# Устанавливаем Gunicorn (если его нет в requirements.txt)
RUN pip install gunicorn

# Порт, который будет использоваться
EXPOSE 8000

# Команда для запуска Gunicorn с одним воркером
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "service.app:app"]