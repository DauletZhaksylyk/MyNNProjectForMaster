# Social Engineering Fraud Detection Using User Behavior Analysis and Machine Learning

This repository contains a machine learning prototype for detecting fraudulent and social engineering conversations based on text and call transcript analysis. The system combines transformer-based text classification with rule-based behavioral indicators to estimate fraud probability, assign a risk level, and highlight suspicious conversation fragments.

## Overview

The goal of this project is to identify potentially fraudulent communication scenarios such as impersonation, requests for SMS codes, money transfer pressure, remote access requests, and other common social engineering tactics.

The system supports:
- single text analysis
- full call transcript analysis
- suspicious segment extraction
- risk scoring with explanation
- simple local API and browser interface

## Key Features

- **Transformer-based fraud classification**
- **Behavioral feature extraction** from text
- **Hybrid decision logic**: model prediction + heuristic risk adjustment
- **Transcript segmentation** for suspicious fragment detection
- **Risk labels**: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`
- **Decision output**: `normal`, `suspicious`, `fraud`
- **Flask web interface** for quick manual testing
- **CLI testing script** for local evaluation
- **Text augmentation utilities** for dataset expansion

## Repository Structure

```text
MyNNProjectForMaster/
├── AugFile.py                     # text augmentation methods
├── FraudDetector.py               # main fraud detection logic
├── TestFile.py                    # local CLI testing
├── fraud_dataset_clean_final.csv  # dataset
├── requirements.txt               # project dependencies
├── server.py                      # Flask API + simple web UI
├── setup_env.bat                  # environment setup (Windows)
├── text_utils.py                  # text preprocessing and feature extraction
└── trainFile.py                   # training pipeline
How It Works

The pipeline follows these steps:

Input preprocessing
transcript cleaning
speaker normalization
masking of sensitive data
text normalization
Structured feature extraction
The system identifies behavioral and linguistic signals such as:
code/SMS verification request
money transfer request
urgency
threat language
authority impersonation
sensitive data request
remote access request
victim confusion or resistance
Model inference
A transformer model predicts the probability of fraud.
Heuristic adjustment
The initial model probability is adjusted using explicit fraud indicators and benign patterns.
Decision generation
The final output includes:
fraud probability
predicted class
risk level
decision reasons
recommendation
suspicious transcript segments
Example Use Cases
post-call fraud screening
suspicious customer support conversation review
social engineering attack analysis
banking/telecom anti-fraud research
human behavior analysis in deceptive communication

Installation

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
Requirements

Main dependencies include:

transformers
torch
pandas
numpy
scikit-learn
joblib
flask
Running the Project
Option 1: Run the Flask web interface
python server.py

Then open the local address shown in the terminal and paste a transcript for analysis.

Option 2: Test from the command line
python TestFile.py --model-dir path/to/model --text "Здравствуйте, я звоню из банка. Назовите код из SMS."

Or analyze a transcript from a file:
python TestFile.py --model-dir path/to/model --file sample.txt

Expected Output

The system returns a structured JSON response containing:

predicted_class
fraud_probability
risk_level
decision_reasons
markers
recommendation
suspicious_segments
Research Contribution

This project is part of a Master’s research work focused on:

Developing a system for detection and prevention of social engineering attacks based on user behavior analysis and machine learning technologies.

The research combines:

cybersecurity
NLP
behavioral analysis
fraud detection
interpretable ML decision support
Potential Improvements

Future work may include:

larger multilingual datasets
better model fine-tuning
real-time call analysis
speaker-aware dialogue modeling
explainable AI visualization
integration with anti-fraud platforms
Author

Daulet Zhaksyllyk
Master’s student in Information Security Systems
Research area: Cybersecurity, Social Engineering Detection, Machine Learning, User Behavior Analysis

License

This project is intended for research and academic purposes.
