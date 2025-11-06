FROM apache/airflow:2.8.4-python3.11

USER root
# 🧩 Install compilers and CMake for building phik/wordcloud
RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc g++ make cmake \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

# ✅ Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt