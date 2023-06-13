FROM python:3.10.4

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt 

COPY . .

EXPOSE 5000

CMD ["python", "src/app.py"]