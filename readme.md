# NBA Player Position Classification

## Overview

This project classifies NBA players into their respective positions (PG, SG, SF, PF, C) based on performance statistics using various machine learning models. The dataset used is from the 2021 NBA season.

## Prerequisites

- Python 3.x
- pandas
- scikit-learn

Install the required packages using:

```sh
pip install pandas scikit-learn

## Dataset
Ensure the dataset file `nba2021.csv` is in the project directory.

## Steps

### Load the dataset and filter rows:
- Load the dataset from `nba2021.csv`.
- Filter out rows where minutes per game (MP) are less than 20.

### Prepare the data:
- Select important columns (`AST`, `TRB`, `ORB`, `BLK`, `DRB`, `3PA`) and the target variable (`Pos`).
- Round numeric columns to two decimal places.
- Map positions to numerical labels.

### Split the data:
- Split the data into training and testing sets.

### Standardize the features:
- Standardize the feature columns using `StandardScaler`.

### Train and evaluate the model:
- Train a `RandomForestClassifier` with varying hyperparameters.
- Evaluate the model's performance on the test set.
- Print training score, test score, and confusion matrix.

### Cross-validation:
- Perform cross-validation to assess model performance.
- Print cross-validation scores and average score.

## Results
- The training score, test score, and confusion matrix are printed to evaluate the model's performance.
- Cross-validation scores are also provided to assess the model's stability.

```
