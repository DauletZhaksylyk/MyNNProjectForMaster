# Social Engineering Fraud Detection Using User Behavior Analysis and Machine Learning

This repository contains a machine learning prototype for detecting fraudulent and social engineering conversations based on text and call transcript analysis. The system combines transformer-based text classification with rule-based behavioral indicators to estimate fraud probability, assign a risk level, and highlight suspicious conversation fragments.

---

## Overview

The goal of this project is to identify potentially fraudulent communication scenarios such as:

- impersonation
- SMS code requests
- money transfer pressure
- remote access requests
- other social engineering tactics

The system supports:
- single text analysis
- full call transcript analysis
- suspicious segment extraction
- risk scoring with explanation
- local API and browser interface

---

## Key Features

- Transformer-based fraud classification
- Behavioral feature extraction from text
- Hybrid decision logic (ML + heuristics)
- Transcript segmentation for suspicious detection
- Risk labels: LOW, MEDIUM, HIGH, CRITICAL
- Decision output: normal, suspicious, fraud
- Flask web interface
- CLI testing support
- Text augmentation utilities

---

## Repository Structure

MyNNProjectForMaster/
- AugFile.py
- FraudDetector.py
- TestFile.py
- fraud_dataset_clean_final.csv
- requirements.txt
- server.py
- setup_env.bat
- text_utils.py
- trainFile.py

---

## How It Works

The pipeline follows these steps:

1. Input preprocessing
- transcript cleaning  
- speaker normalization  
- masking sensitive data  
- text normalization  

2. Structured feature extraction
The system identifies behavioral signals such as:
- code/SMS verification request  
- money transfer request  
- urgency  
- threat language  
- authority impersonation  
- sensitive data request  
- remote access request  
- victim confusion or resistance  

3. Model inference  
A transformer model predicts fraud probability.

4. Heuristic adjustment  
The probability is adjusted using explicit fraud indicators and safe patterns.

5. Decision generation  
The system outputs:
- fraud probability  
- predicted class  
- risk level  
- decision reasons  
- recommendation  
- suspicious segments  

---

## Example Use Cases

- post-call fraud screening  
- customer support monitoring  
- social engineering detection  
- banking / telecom anti-fraud systems  
- behavioral cybersecurity research  

---

## Installation

1. Clone the repository

git clone https://github.com/DauletZhaksylyk/MyNNProjectForMaster.git  
cd MyNNProjectForMaster  

2. Create a virtual environment  

python -m venv venv  

Activate it:

Windows  
venv\Scripts\activate  

Linux / macOS  
source venv/bin/activate  

3. Install dependencies  

pip install -r requirements.txt  

---

## Requirements

Main dependencies:
- transformers  
- torch  
- pandas  
- numpy  
- scikit-learn  
- joblib  
- flask  

---

## Running the Project

Option 1: Run the web interface  

python server.py  

Then open the local URL and input a transcript.

Option 2: CLI testing  

python TestFile.py --model-dir path/to/model --text "Здравствуйте, я звоню из банка. Назовите код из SMS."  

Or from file:  

python TestFile.py --model-dir path/to/model --file sample.txt  

---

## Expected Output

The system returns JSON with:

- predicted_class  
- fraud_probability  
- risk_level  
- decision_reasons  
- markers  
- recommendation  
- suspicious_segments  

---

## Research Contribution

This project is part of a Master’s research:

Detection and prevention of social engineering attacks using machine learning and user behavior analysis.

Focus areas:
- cybersecurity  
- NLP  
- behavioral analysis  
- fraud detection  
- interpretable ML  

---

## Potential Improvements

- multilingual datasets  
- improved model fine-tuning  
- real-time detection  
- dialogue-aware models  
- explainable AI visualization  
- production integration  

---

## Author

Daulet Zhaksylyk  
Master’s Student — Information Security Systems  

Research interests:

- Cybersecurity  
- Social Engineering Detection  
- Machine Learning  
- User Behavior Analysis  

---

## License

This project is intended for research and academic purposes.
