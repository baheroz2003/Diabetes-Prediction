import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Prediction function
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return '✅ The person is not diabetic' if prediction[0] == 0 else '⚠️ The person is diabetic'

def main():
    # Stylish top developer tag
    st.markdown("""
        <style>
        .developer-tag {
            background-color: #f1f3f6;
            color: #333;
            padding: 10px;
            border-radius: 10px;
            font-size: 16px;
            text-align: center;
            font-weight: bold;
            box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        </style>
        <div class="developer-tag">
            🚀 Developed by <span style='color:#ff4b4b;'>Baheroz Zeya</span>
        </div>
    """, unsafe_allow_html=True)

    # Stylish title
    st.markdown("<h1 style='text-align:center; color:#0066cc;'>🧠 Smart Diabetes Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Input form
    st.subheader("🔍 Please enter the following details:")
    Pregnancies = st.text_input('👶 Number of Pregnancies')
    Glucose = st.text_input('🍭 Glucose Level')
    BloodPressure = st.text_input('💓 Blood Pressure value')
    SkinThickness = st.text_input('📏 Skin Thickness value')
    Insulin = st.text_input('💉 Insulin Level')
    BMI = st.text_input('⚖️ BMI value')
    DiabetesPedigreeFunction = st.text_input('🧬 Diabetes Pedigree Function value')
    Age = st.text_input('🎂 Age of the Person')

    diagnosis = ''

    if st.button('🔎 Get Diabetes Test Result'):
        diagnosis = diabetes_prediction([
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ])
        st.success(diagnosis)

    st.markdown("<br><hr><p style='text-align:center; color:gray;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
