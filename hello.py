import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.title("🏡 Housing Price Predictor")
st.write("This app predicts **Median House Value** based on **Total Rooms**.")

# Load dataset
df = pd.read_csv(r"C:\Users\91700\Downloads\Excl\housing.csv")
df = df.dropna(subset=['total_rooms', 'median_house_value'])

X = df[['total_rooms']]
y = df['median_house_value']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plot
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Actual')
ax.plot(X, y_pred, color='red', label='Prediction')
ax.set_xlabel("Total Rooms")
ax.set_ylabel("Median House Value")
ax.set_title("Linear Regression")
ax.legend()
st.pyplot(fig)

# Metrics
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

st.subheader("📊 Model Evaluation Metrics")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**R² Score:** {r2:.4f}")

# Predict a custom input
st.subheader("🔮 Predict House Value")
rooms_input = st.number_input("Enter total rooms:", value=5000)
if st.button("Predict"):
    prediction = model.predict([[rooms_input]])
    st.success(f"Predicted Median House Value: ${prediction[0]:,.2f}")
