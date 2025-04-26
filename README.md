# Cabin Price Prediction Model

This project implements a machine learning model to predict cabin booking prices using various features from historical booking data.

## Project Overview

The project uses a Random Forest Regressor to predict cabin booking prices based on historical data. The model takes into account various features including booking dates, cabin types, and other relevant attributes to make accurate price predictions.

## Files in the Project

- `cabin_price_prediction1.ipynb`: Jupyter notebook containing the data preprocessing, model training, and evaluation code
- `cabin_bookings.csv`: Dataset containing historical cabin booking information
- `price_prediction_model.pkl`: Trained model saved in pickle format

## Features Used

The model uses the following features for prediction:
- Booking dates (month and weekday)
- Cabin type
- Number of guests
- Various cabin amenities and characteristics

## Model Performance

The model uses the following evaluation metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared Score

## Requirements

To run this project, you'll need the following Python packages:
- pandas
- scikit-learn
- joblib

## Usage

1. Clone this repository
2. Install the required packages
3. Open the Jupyter notebook `cabin_price_prediction1.ipynb`
4. Run the cells to train the model or make predictions

## Data Preprocessing

The data preprocessing steps include:
- Handling missing values
- Converting date features
- Feature engineering
- Data scaling

## Model Training

The model training process includes:
- Data splitting into training and testing sets
- Feature scaling
- Hyperparameter tuning using GridSearchCV
- Model training with Random Forest Regressor
- Model evaluation using cross-validation

## Model Saving

The trained model is saved in pickle format (`price_prediction_model.pkl`) for future use.

## Future Improvements

Potential areas for improvement:
- Feature engineering
- Model selection
- Hyperparameter optimization
- Integration with a web interface
- Real-time prediction capabilities # cabin-price-prediction

