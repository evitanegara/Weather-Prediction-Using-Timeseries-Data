# Weather Forecasting for Agriculture Using Machine Learning and Deep Learning
## Project Overview
This project focuses on the development and evaluation of machine learning and deep learning models for time-series forecasting of solar radiation (SRAD), a key climatic factor influencing crop productivity. Leveraging 40 years of historical meteorological data from Rothamsted, England (1959–1999), the project aims to build accurate and interpretable models that can support agricultural planning, including irrigation scheduling and planting decisions. The project explores a comparative modeling approach using traditional machine learning algorithms (KNN, Random Forest, XGBoost, AdaBoost), deep learning techniques (MLP, LSTM), and hybrid ensemble methods (Voting, Stacking). Each model is rigorously optimized and evaluated using a consistent methodology, with results informing best practices for time-series forecasting in agricultural contexts.
## Dataset Overview

- **Source:** [e-RA Rothamsted Online Database (UK)](https://www.era.rothamsted.ac.uk/)
- **Period:** 1959–1999  
- **Frequency:** Daily observations (resampled to biweekly)  

| Column | Description | Type |
|--------|-------------|------|
| DATE | Date of observation (originally in DDMMYY format, converted to datetime) | Categorical (Datetime) |
| SRAD | Solar radiation (MJ/m²) | Continuous |
| TMAX | Maximum air temperature (°C) | Continuous |
| TMIN | Minimum air temperature (°C) | Continuous |
| RAIN | Daily precipitation (mm) | Continuous |

## Executive Summary
This project investigates multiple supervised learning approaches for time-series regression forecasting. Ensemble tree models and neural networks demonstrate strong accuracy, with the Voting Regressor achieving the best overall performance. Random Forest stands out for its low MAPE (0.1234) and high R² (0.9736), making it highly suitable for capturing complex relationships in climate data. XGBoost delivers similar results with efficient learning from difficult cases. MLP offers solid performance in modeling non-linear dependencies, while LSTM shows the best R² (0.9741), capturing temporal sequences well. The ensemble-based Voting Regressor achieves the lowest MAPE (0.1134), indicating it generalizes best across the evaluation folds. Visualizations show consistent model performance in replicating SRAD seasonality and trend, with future forecasts successfully projecting rising solar radiation into 2000.

## Project Workflow

### 1. Data Import and Library Setup
- Utilized libraries such as **Pandas**, **NumPy**, **Scikit-learn**, **Matplotlib**, **TensorFlow**, and **Keras** for data preprocessing, visualization, and model training.

### 2. Data Reading and Preprocessing
- Loaded 40 years of daily DSSAT-format weather data.
- Converted date strings (DDMMYY) into Python datetime objects.
- Resampled daily data into biweekly averages to reduce short-term volatility and highlight long-term trends.
  ![image](https://github.com/user-attachments/assets/2b5a1cdc-0730-45d2-a0c0-7c74a8c4f89e)


### 3. Visualization and Outlier Handling
- Visualized distributions of key features (SRAD, RAIN, TMAX, TMIN) using histograms and boxplots.
- Applied IQR-based capping** to handle rainfall outliers and reduce skew.
  ![image](https://github.com/user-attachments/assets/525682c5-afd1-4b5f-89d9-a8e3f309531f)


### 4. Data Splitting and Scaling
- Implemented walk-forward validation, using the last 26 weeks as a rolling test set while preserving temporal order.
- Standardized features using StandardScaler (Z-score normalization) for consistent model input.

### 5. Model Building and Hyperparameter Tuning
Trained and optimized the following models:
- K-Nearest Neighbors (KNN)  
- Random Forest Regressor  
- XGBoost Regressor  
- AdaBoost Regressor  
- Multi-Layer Perceptron (MLP)  
- Long Short-Term Memory (LSTM)  
- Voting Regressor (ensemble)  
- Stacking Regressor (meta-ensemble)

- GridSearchCV with 5-fold cross-validation was used to tune hyperparameters across 26 walk-forward iterations.

### 6. Future Forecasting
- Generated synthetic weekly data for the year 2000.
- Applied the same scaling transformations.
- Used trained models to generate forward predictions of solar radiation.

---

##  Model Performance Summary

| Model         | MAPE   | R²     | MAE    | RMSE   |
|---------------|--------|--------|--------|--------|
| Voting        | 0.1134 | 0.9715 | 0.8295 | 1.1655 |
| LSTM          | 0.1364 | 0.9741 | 0.8880 | 1.0635 |
| Random Forest | 0.1234 | 0.9736 | 0.8024 | 1.0748 |
| XGBoost       | 0.1168 | 0.9718 | 0.8102 | 1.1107 |
| MLP           | 0.1170 | 0.9687 | 0.8395 | 1.1691 |
| Stacking      | 0.1298 | 0.9677 | 0.8813 | 1.1882 |
| AdaBoost      | 0.1467 | 0.9643 | 0.9686 | 1.2500 |
| KNN           | 0.1893 | 0.9236 | 1.4012 | 1.8280 |

---

## Highlights

- **Voting Regressor** achieved the lowest MAPE (11.34%), offering the most accurate average forecasts.
- **LSTM** scored the highest R² (97.41%) and lowest RMSE, excelling at learning temporal dependencies.
- **Random Forest** and **XGBoost** offered strong balance between accuracy and interpretability.
- **MLP** showed solid performance on nonlinear patterns but was slightly more noise-sensitive.
- **KNN** was the weakest performer due to its sensitivity to noise and distance metrics in high-dimensional data.

## Forecast Visualization

- All models effectively captured the seasonal trend of solar radiation.
- For future prediction, all the models show an upward pattern, which makes sense because these models have effectively captured the seasonal and diurnal patterns present in the historical data. The upward trend observed in future predictions is likely a reflection of seasonal increases in solar radiation.

## Key Takeaways

- **Ensemble Learning Works Best:** Voting Regressor had the most balanced and reliable predictions.
- **Temporal Dynamics Matter:** LSTM’s ability to model sequences made it ideal for time-series forecasting.
- **Tree-Based Models Excel:** Random Forest and XGBoost offered both performance and transparency.
- **Data Quality & Scaling are Critical:** Outlier capping and feature standardization enhanced model consistency.

---

## Forecast Visualization

- All models effectively captured the seasonal trend of solar radiation.
- For future prediction, all the models show an upward pattern, which makes sense because these models have effectively captured the seasonal and diurnal patterns present in the historical data. The upward trend observed in future predictions is likely a reflection of seasonal increases in solar radiation.
  ![image](https://github.com/user-attachments/assets/2cbf12d5-7a7a-481c-bf7a-e1e722e92ab5)
  ![image](https://github.com/user-attachments/assets/e62f7f3c-763c-4ea2-b633-16e89e4e2a2c)

##  Contact

For questions, feedback, or collaboration inquiries: **evitanegara@gmail.com**
