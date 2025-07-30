FROM python:3.12-slim

WORKDIR /app

RUN pip install --upgrade pip && pip install 'uv[cli]' --upgrade

COPY . .

RUN uv pip install --editable . --verbose --python /usr/local/bin/python

CMD ["python", "src/run_pipeline.py"]
