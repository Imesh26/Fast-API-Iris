# Iris ML Model API

A tiny FastAPI app that predicts the iris flower species from four measurements. It’s fast, simple, and perfect for demos or assignments.

## What this does

Given four numbers — `sepal_length`, `sepal_width`, `petal_length`, `petal_width` — the API returns one of:

- `setosa`
- `versicolor`
- `virginica`

It also includes a confidence score when available.

## Why this model?

 This use a scikit-learn Pipeline with `StandardScaler` + `LogisticRegression`. It trains in seconds, performs well on the Iris dataset, and keeps the code easy to read.

## Quick start

```bash
#Create & activate a clean environment (Anaconda)
conda create -n irisapi python=3.11 -y
conda activate irisapi

#Install dependencies
pip install -r requirements.txt

#Train and save the model
python train_model.py   # creates model.pkl + model_info.json

#Run the API
uvicorn main:app --reload

```
