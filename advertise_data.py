import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

@st.cache_data
def load_data():
    df = pd.read_csv('Advertising.csv')  # Update with the correct path to your dataset
    return df

df = load_data()

st.title('Advertising Sales Prediction')

st.write("""
This app predicts sales based on advertising spending using different regression models.
""")

st.subheader('Advertising Dataset')
st.write(df.head())

st.sidebar.title('Select Advertising Expenditure')
TV = st.sidebar.number_input('TV Advertising Spend ($)', min_value=0.0)
Radio = st.sidebar.number_input('Radio Advertising Spend ($)', min_value=0.0)
Newspaper = st.sidebar.number_input('Newspaper Advertising Spend ($)', min_value=0.0)

X = df[['TV', 'radio', 'newspaper']]
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mse, r2

st.header('Linear Regression')
linear_model = LinearRegression()
linear_y_pred, linear_mse, linear_r2 = train_and_evaluate_model(linear_model)
st.write(f'Linear Regression Mean Squared Error: {linear_mse:.2f}')
st.write(f'Linear Regression R^2 Score: {linear_r2:.2f}')

fig, ax = plt.subplots()
ax.scatter(y_test, linear_y_pred, color='blue', alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax.set_xlabel('Actual Sales')
ax.set_ylabel('Predicted Sales')
ax.set_title('Linear Regression: Actual vs Predicted Sales')
st.pyplot(fig)

st.header('Lasso Regression')
lasso_model = Lasso(alpha=0.1)
lasso_y_pred, lasso_mse, lasso_r2 = train_and_evaluate_model(lasso_model)
st.write(f'Lasso Regression Mean Squared Error: {lasso_mse:.2f}')
st.write(f'Lasso Regression R^2 Score: {lasso_r2:.2f}')

fig, ax = plt.subplots()
ax.scatter(y_test, lasso_y_pred, color='green', alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax.set_xlabel('Actual Sales')
ax.set_ylabel('Predicted Sales')
ax.set_title('Lasso Regression: Actual vs Predicted Sales')
st.pyplot(fig)

st.header('Ridge Regression')
ridge_model = Ridge(alpha=0.1)
ridge_y_pred, ridge_mse, ridge_r2 = train_and_evaluate_model(ridge_model)
st.write(f'Ridge Regression Mean Squared Error: {ridge_mse:.2f}')
st.write(f'Ridge Regression R^2 Score: {ridge_r2:.2f}')

fig, ax = plt.subplots()
ax.scatter(y_test, ridge_y_pred, color='red', alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax.set_xlabel('Actual Sales')
ax.set_ylabel('Predicted Sales')
ax.set_title('Ridge Regression: Actual vs Predicted Sales')
st.pyplot(fig)

st.header('Predict Sales')
user_input = np.array([[TV, Radio, Newspaper]])
linear_prediction = linear_model.predict(user_input)[0]
lasso_prediction = lasso_model.predict(user_input)[0]
ridge_prediction = ridge_model.predict(user_input)[0]

st.write(f'Linear Regression Prediction: {linear_prediction:.2f}')
st.write(f'Lasso Regression Prediction: {lasso_prediction:.2f}')
st.write(f'Ridge Regression Prediction: {ridge_prediction:.2f}')

