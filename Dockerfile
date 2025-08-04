FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt 

#install jupyter lab
RUN pip install --no-cache-dir jupyterlab

COPY data/raw/ data/raw/
COPY Notebooks/ Notebooks/
COPY src/powerco_churn/ src/powerco_churn/
COPY converter.py .
COPY main.py .

# Add src to PYTHONPATH
ENV PYTHONPATH=/app/src

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]