FROM python:3.9 as builder

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /usr/src/app/wheels -r requirements.txt

FROM python:3.9-slim

WORKDIR /app

COPY --from=builder /usr/src/app/wheels /wheels/
RUN pip install --no-cache /wheels/*

COPY . .

CMD ["python", "main.py"]