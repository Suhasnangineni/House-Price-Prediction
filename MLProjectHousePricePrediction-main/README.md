# House Price Prediction

A machine learning project that predicts house prices using various regression techniques on the Boston Housing dataset.

## Problem Statement

Predict house prices based on location, area, number of rooms, and amenities.

## Real-Life Application

Used by real estate platforms like Zillow, MagicBricks for property valuation.

## Dataset

- **Source**: Kaggle - Boston Housing Dataset
- **Samples**: 506 rows
- **Features**: 13 (crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat)
- **Target**: medv (Median home value in $1000s)

### Key Features

| Feature | Description |
|---------|-------------|
| crim | Per capita crime rate |
| zn | Residential land proportion |
| indus | Non-retail business acres |
| chas | Charles River dummy |
| nox | Nitric oxides concentration |
| rm | Average rooms per dwelling |
| age | Owner-occupied units built prior to 1940 |
| dis | Distance to employment centers |
| rad | Accessibility to highways |
| tax | Property tax rate |
| ptratio | Pupil-teacher ratio |
| b | 1000(Bk - 0.63)^2 |
| lstat | % lower status population |

## Algorithms Implemented

### 1. Linear Regression
- **Univariate**: Using only `rm` (rooms) feature
- **Multivariate**: Using all 13 features
- **Feature Selection**: Top 6 correlated features

### 2. Polynomial Regression
- Degree 2 and Degree 3

### 3. Gradient Descent
- SGDRegressor with constant and adaptive learning rates

## Key Concepts Covered

- Regression (MSE, RMSE, MAE, R²)
- Overfitting vs Underfitting analysis
- Cross-validation (5-fold)
- Feature selection based on correlation

## Model Performance

| Model | Train R² | Test R² | Test RMSE | CV R² (mean±std) |
|-------|----------|---------|-----------|------------------|
| Linear Regression (Multivariate) | 0.743 | 0.710 | 4.650 | 0.688 ± 0.092 |
| Linear Regression (Feature Selection) | 0.687 | 0.651 | 5.099 | 0.651 ± 0.090 |
| Polynomial Regression (degree=3) | 0.549 | 0.583 | 5.577 | 0.491 ± 0.205 |
| Polynomial Regression (degree=2) | 0.536 | 0.567 | 5.679 | 0.483 ± 0.224 |
| Linear Regression (Univariate) | 0.489 | 0.458 | 6.355 | 0.452 ± 0.177 |

### Best Model
**Linear Regression (Multivariate)** with Test R² = 0.710

## Project Structure

```
.
├── notebooks/
│   ├── analyze.ipynb    # Data analysis
│   └── train.ipynb      # Model training
├── results/             # Model outputs (gitignored)
│   ├── *.joblib        # Trained models
│   ├── *.csv           # Data splits
│   ├── *.json          # Metrics
│   └── *.npy           # Predictions
├── pyproject.toml       # Dependencies
└── README.md
```

## Installation

```bash
# Install dependencies using UV
uv sync

# Or install manually
pip install -r requirements.txt
```

## Usage

1. **Data Analysis**
   ```bash
   jupyter notebook notebooks/analyze.ipynb
   ```

2. **Model Training**
   ```bash
   jupyter notebook notebooks/train.ipynb
   ```

## Requirements

- Python 3.12+
- pandas
- numpy
- scikit-learn
- jupyter
- kagglehub

## Results

All results are saved in the `results/` directory:
- Trained models (`.joblib`)
- Predictions (`.npy`)
- Metrics (`.json`)
- Model comparison (`.csv`)

## License

MIT License
