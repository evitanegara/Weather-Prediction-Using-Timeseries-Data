# Weather Forecasting for Agriculture Using Machine Learning
## Project Overview
This project focuses on the development and evaluation of machine learning and deep learning models for time-series forecasting of solar radiation (SRAD), a key climatic factor influencing crop productivity. Leveraging 40 years of historical meteorological data from Rothamsted, England (1959–1999), the project aims to build accurate and interpretable models that can support agricultural planning, including irrigation scheduling and planting decisions. The project explores a comparative modeling approach using traditional machine learning algorithms (KNN, Random Forest, XGBoost, AdaBoost), deep learning techniques (MLP, LSTM), and hybrid ensemble methods (Voting, Stacking). Each model is rigorously optimized and evaluated using a consistent methodology, with results informing best practices for time-series forecasting in agricultural contexts.
## Dataset Overview

- **Source:** [e-RA Rothamsted Online Database (UK)](https://www.era.rothamsted.ac.uk/)
- **Period:** 1959–1999  
- **Frequency:** Daily observations (resampled to biweekly)  
- **Format:** DSSAT-compliant weather data format  
- **Missing Values:** None  
- **Outliers:** Rainfall outliers were detected using IQR and capped to reduce model distortion.

## Dataset Fields

| Column | Description | Type |
|--------|-------------|------|
| `DATE` | Date of observation (originally in DDMMYY format, converted to datetime) | Categorical (Datetime) |
| `SRAD` | Solar radiation (MJ/m²) | Continuous |
| `TMAX` | Maximum air temperature (°C) | Continuous |
| `TMIN` | Minimum air temperature (°C) | Continuous |
| `RAIN` | Daily precipitation (mm) | Continuous |


# Project objectives :
- This project aims to develop and compare models for weather forecasting, focusing on solar radiation (SRAD) as the target variable.
- Using a 40-year time series dataset (1959-1999), models like KNN, Random Forest, XGBoost, LSTM, and hybrid methods (stacking, voting) are evaluated.
- Hyperparameter tuning is applied to optimize model performance, improving predictive accuracy for agricultural decision-making.

# Project Overview 
This project involved several key steps to predict the weather in Agriculture. 
- **Data Import and Library Setup** : Utilized libraries such as Pandas, NumPy, Scikit-learn, Matplotlib, TensorFlow, and Keras for data manipulation, machine learning, deep learning, and visualization.
- **Data Reading and Preprocessing** : Consolidated 40 years of weather data, extracted key variables (SRAD, TMAX, TMIN, RAIN), converted dates to a standardized format. In addition, aggregated daily data into weekly averages for clearer trend analysis and noise reduction.
- **Data Visualization** : Created monthly and histogram visualizations for key variables to identify trends and detect outliers in the data.
- **Handling Outlier** : Handled outliers in the rainfall data using the Interquartile Range (IQR) method, ensuring model accuracy and reliability.
- **Data Splitting and Scaling** : Applied the rolling forecast method to split the dataset into training and testing sets, ensuring the chronological integrity of the time series data.
- **Scalling Data** : Standardized the features to ensure consistent scaling across all variables using Standard Scaller
- **Model Building and Hyperparameter Tuning** :
  - Developed models such as KNN, Random Forest, XGBoost, AdaBoost, MLP, and LSTM, including ensemble techniques like Voting and Stacking Regressors for improved predictions.
  - Fine-tuned hyperparameters using GridSearchCV to optimize model performance.
- **Future Data Prediction** : Generated artificial data for the year 2000 to predict solar radiation using biweekly intervals, extending the dataset for future predictions.
- **Evaluation Metric** : The models were evaluated based on several performance metrics: Mean Absolute Percentage Error (MAPE), R-squared (R2), Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) 

# Project Result 
- Random Forest: Achieved 12.34% MAPE and 97.36% R², providing accurate predictions and effectively handling complex data patterns.
- XGBoost and AdaBoost: XGBoost outperformed AdaBoost with 11.68% MAPE and 97.18% R², showing resilience to noise and capturing challenging patterns.
- MLP and LSTM: LSTM stood out with the highest R² (97.41%) and lowest RMSE (1.06), excelling in modeling temporal dependencies.
- Ensemble Methods: The Voting Regressor was the best performer (MAPE: 11.34%), while the Stacking Regressor was also competitive (MAPE: 12.98%), benefiting from combining multiple models for improved accuracy.
  
# Visualization
<img width="1083" alt="vis" src="https://github.com/user-attachments/assets/5d22f188-a79b-4538-8898-18a591947825">
<img width="527" alt="viss" src="https://github.com/user-attachments/assets/a3cc3edf-f449-4a10-b653-e2c8eb4bc514">

- Based on time series visualizations, all models successfully capture and predict patterns in the test data, demonstrating their robustness in handling time-series forecasting and understanding temporal dynamics.
- All models show an upward trend in future predictions, effectively capturing seasonal and diurnal patterns in the historical data.


# Conlusion 
This project applied machine learning, deep learning, and hybrid models to predict solar radiation. Voting Regressor and Random Forest offered reliable predictions, while LSTM excelled at capturing temporal patterns. Hyperparameter tuning further improved performance, highlighting the effectiveness of ensemble methods and deep learning for time-series forecasting in agricultural weather prediction.

# Contact 
For any questions or inquiries, please contact evitanegara@gmail.com
