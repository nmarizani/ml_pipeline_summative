## Vaccine Demand Prediction Project

## Project Overview
The Vaccine Demand Prediction Project is designed to predict vaccine demand across categories such as **high demand**, **medium demand** and **low demand** using a machine learning pipeline. This is to aid in local production of vaccines for high demand areas. 
It integrates **data ingestion, preprocessing, model training, and retraining**, supported by a **FastAPI backend** and evaluated under stress using **Locust flood request simulations**.  

## Demo Video
**Watch the Demo on YouTube:**[]

---

## 🌐 Live URLs
- **FastAPI Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)  
- **Locust Load Testing Dashboard:** [http://localhost:8089](http://localhost:8089)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
``bash
git https://github.com/nmarizani/ml_pipeline_summative.git

---

### 2. Create & Activate a Virtual Environment
``bash
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows

---

### 3. Install Dependencies
``bash
pip install --upgrade pip
pip install -r requirements.txt

---

4. Run the FastAPI Application
``bash
uvicorn src.main:app --host 127.0.0.1 --port 8000
Access the app documentation at: http://localhost:8000/docs

---

### 5. Running the Locust Flood Request Simulation
``bash
locust -f src/locustfile.py --host=http://localhost:8000
Then open http://localhost:8089 in your browser to start the simulation.

---

### 6. Project Structure
├── data/                 # Dataset for training/testing
│   ├── train/
│   └── test/
├── models/               # Saved ML models
├── src/                  # Source code
│   ├── locustfile.py     # Locust load testing script
│   ├── preprocessing.py  # Data preprocessing utilities
│   ├── train.py          # Training pipeline
│   └── utils.py          # Helper functions
├── main.py               # FastAPI backend entry point
├── requirements.txt      # Dependencies
└── README.md             # Project documentation

---

### Results from Flood Request Simulation
The system was tested with 40 concurrent users performing bulk uploads.

Metric	Result
Total Requests	484
Failures	483 (100%)
Median Response Time	2100 ms
95th Percentile Response Time	16000 ms
99th Percentile Response Time	19000 ms
Max Response Time	20156 ms
Average RPS	9.6

### Observations
Nearly all requests failed with 500 Internal Server Error due to heavy load.

Indicates a bottleneck in:

File upload handling (synchronous I/O).

Database transaction limits.

Retraining logic being triggered too frequently.