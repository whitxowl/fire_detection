FROM ultralytics/ultralytics:latest-cpu

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir numpy

ENTRYPOINT ["python", "test.py"]