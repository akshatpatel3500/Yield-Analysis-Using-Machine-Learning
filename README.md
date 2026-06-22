# Crop Recommendation using Machine Learning

A machine learning system that recommends the best crop to plant based on soil and climate conditions, served through a Flask web app. Built following the CRISP-DM process, it trains and compares seven classifiers and deploys the best one for real-time predictions.

![Crop field](crop.jpg)

## Problem

Choosing the right crop depends on many interacting factors — soil nutrients (N, P, K), pH, temperature, humidity, and rainfall. This project turns those measurements into an accurate crop recommendation.

## Dataset

[Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) (Kaggle) — 2,200 samples across 22 crops, with 7 soil/climate features and the crop label.

> To run the notebook, download `Crop_recommendation.csv` from the link above and place it in the project folder. (The original project loaded this from AWS S3; this version uses the public file for reproducibility.)

## Approach

1. **EDA** — feature distributions and per-crop analysis
2. **Preprocessing** — normalization, label encoding, imputation, 80/20 train/test split
3. **Modeling** — seven classifiers, each tuned with GridSearchCV:
   Logistic Regression · KNN · Decision Tree · Random Forest · SVM · Gaussian Naive Bayes · LDA
4. **Evaluation** — accuracy, precision, recall, F1, and ROC-AUC

## Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **Random Forest** | **99.31%** | **99.99%** |
| Decision Tree | 98.63% | 99.33% |
| SVM | 98.18% | 99.33% |
| KNN | 97.50% | 97.90% |
| LDA | 94.31% | 99.91% |
| Logistic Regression | 92.04% | 99.68% |

Random Forest was selected as the final model and deployed.

## Web App

The trained Random Forest model is served through a **Flask** app: enter soil and climate values in a web form and get an instant crop recommendation. The model, label encoder, and scaler are loaded from the saved `.pkl` files.

### Run it locally

```bash
git clone https://github.com/akshatpatel3500/crop-recommendation.git
cd crop-recommendation
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

> Note: the `.pkl` files were saved with scikit-learn < 1.3. `requirements.txt` pins `scikit-learn==1.2.2` so they load correctly — adjust to the version you trained with if needed.

## Repository Structure

```
crop-recommendation/
├── crop_recommendation.ipynb   # full analysis: EDA, modeling, evaluation
├── app.py                      # Flask web app
├── templates/index.html        # web form
├── random_forest_model.pkl     # trained model
├── label_encoder.pkl           # crop label encoder
├── scaler.pkl                  # feature scaler
├── requirements.txt
└── crop.jpg
```

## Tech Stack

Python · scikit-learn · pandas · NumPy · Flask · Matplotlib · Seaborn

## Team

A group project for DATA-245 at San Jose State University by Akshat Patel, Swaraj Kulkarni, Suhail Chopra, Harika Boniya, and Sneha Karri.
