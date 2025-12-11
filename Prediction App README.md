# LinkedIn User Prediction App
A Streamlit application that predicts the likelihood that an individual uses LinkedIn based on demographic characteristics.

## Overview
This project builds and deploys a logistic regression model trained on survey data to estimate the probability that a person uses LinkedIn. Users adjust demographic inputs such as income, education, gender, age, marital status, and parent status. The app returns a prediction along with interactive visual insights based on the underlying dataset.

## Live App
https://linkedin-usage-prediction.streamlit.app/

## Project Structure
├── linkedIn.py                 # Streamlit application  
├── user_prediction_model.pkl   # Saved logistic regression model  
├── ss_cleaned.csv              # Cleaned dataset used for visualizations  
├── requirements.txt            # Python dependencies  
└── README.md                   # Documentation  

## How the Model Works
A logistic regression classifier was trained on a cleaned social media usage dataset.  
The target variable is LinkedIn usage (1 = uses LinkedIn, 0 = does not use LinkedIn).

Features included in the model:
- Income level  
- Education level  
- Parent status  
- Marital status  
- Gender  
- Age  

The model outputs:
- A predicted class (LinkedIn User or Not a User)  
- A probability score  

## Application Features

### 1. User Input Panel
Users select demographic attributes that the model uses to calculate a personalized LinkedIn usage probability.

### 2. Prediction Output
When the user clicks the Predict button, the app displays:
- Predicted class  
- Probability of LinkedIn usage  

### 3. Interactive Insights
A “Learn More” section displays Altair charts illustrating LinkedIn usage patterns by:
- Income  
- Education  
- Age  
- Parent status  
- Marital status  
- Gender  

These visualizations provide context for understanding how different demographic groups behave in the dataset.

## Technologies Used
- Python  
- Streamlit  
- scikit-learn  
- pandas  
- numpy  
- Altair  

## Installation Instructions

### 1. Clone the repository
git clone https://github.com/eahashie/linkedIn-user-prediction-app.git

cd linkedIn-user-prediction-app

### 2. Install dependencies

### 3. Run the application

## Dataset Information
The cleaned dataset `ss_cleaned.csv` includes the following variables:
- sm_li (LinkedIn user indicator)  
- income  
- education  
- parent  
- married  
- female  
- age  

These variables were created through preprocessing of the original survey dataset `social_media_usage.csv`.

## Model File
`user_prediction_model.pkl` contains the trained logistic regression model created using scikit-learn.

## Author
Eunice Ahashie  
Georgetown MSBA Candidate  
Data Analytics and Business Modeling  

## Future Enhancements
- Add alternative machine learning models  
- Integrate more advanced interpretability methods  
- Improve the user interface  
- Add batch prediction functionality  
