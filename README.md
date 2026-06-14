# 🚀 HR Employee Attrition Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green.svg)

## 📌 Introduction
This is a Machine Learning web application built to predict employee attrition. It helps HR departments understand which employees are at a high risk of leaving the company, enabling them to take proactive retention measures.

## ✨ Key Features
* **Real-time Prediction:** Enter employee details to instantly get the attrition risk percentage.
* **Interactive UI:** Clean and user-friendly interface built with Streamlit.
* **Data Driven:** Powered by a robust Random Forest Classifier.

## 🛠️ Technologies Used
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Data Processing:** Pandas
* **Visualization:** Plotly

## 📁 Project Structure
* `app.py` : Main Streamlit web application script.
* `model_training.py` : Backend script used to clean data and train the ML model.
* `Model/` : Contains `.pkl` files (Saved model, scaler, and feature list).
* `Dataset/` : Contains the original HR dataset.
* `requirements.txt` : List of all Python dependencies.

## 🚀 How to Run the Project
1. Open your terminal.
2. Install all the required libraries by running this command:
   ```bash
   pip install -r requirements.txt


3. Start the Streamlit application by running:
   ```bash
   streamlit run app.py   

## 👨‍💻 Author
**Vishal Kashyap** *B.Tech CSE, UCER (AKTU)*