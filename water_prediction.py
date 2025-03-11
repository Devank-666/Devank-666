import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data():
    df = pd.read_csv('water_potability.csv')  
    df = df.dropna()  
    return df

df = load_data()

st.title("Water Quality Analysis and Prediction")

st.write("""
This app analyzes water quality data and predicts whether water is drinkable or not based on its mineral content.
""")

st.subheader("Raw Data")
st.write(df.head())

st.subheader("Descriptive Statistics")
st.write(df.describe())

st.subheader("Distribution of Features")
features = df.columns[:-1]  

for feature in features:
    st.write(f"Distribution of {feature}")
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

X = df.drop('Potability', axis=1)
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)

st.subheader("Predict Water Potability")
st.write("Enter the following water quality parameters to predict if the water is drinkable or not.")

ph = st.number_input('pH ', min_value=0.0, max_value=14.0, value=7.0, step=0.1)
hardness = st.number_input('Hardness', min_value=0.0, value=100.0)
solids = st.number_input('Solids ', min_value=0.0, value=15000.0)
chloramines = st.number_input('Chloramines ', min_value=0.0, value=7.0)
sulfate = st.number_input('Sulfate ', min_value=0.0, value=250.0)
conductivity = st.number_input('Conductivity ', min_value=0.0, value=500.0)
organic_carbon = st.number_input('Organic Carbon ', min_value=0.0, value=15.0)
trihalomethanes = st.number_input('Trihalomethanes', min_value=0.0, value=80.0)
turbidity = st.number_input('Turbidity ', min_value=0.0, value=3.0)

input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
dt_prediction = dt_model.predict(input_data)[0]
lr_prediction = lr_model.predict(input_data)[0]

st.write(f"Decision Tree Prediction: {'Drinkable' if dt_prediction == 1 else 'Not Drinkable'}")
st.write(f"Logistic Regression Prediction: {'Drinkable' if lr_prediction == 1 else 'Not Drinkable'}")

st.write(f"Decision Tree Model Accuracy: {dt_accuracy:.2f}")
st.write(f"Logistic Regression Model Accuracy: {lr_accuracy:.2f}")
