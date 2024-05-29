import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('trained_svc_pipeline-0.1.0.pkl','rb'))

def predict_banknote(variance, skewness, curtosis, entropy):

    input = np.array([[variance, skewness, curtosis, entropy]]).astype(np.float64)
    prediction = model.predict(input)
    
    return int(prediction)

def main():
    st.title("Banknote Authentication Classifier")
    variance = st.text_input("variance", placeholder="Type Here")
    skewness = st.text_input("skewness", placeholder="Type Here")
    curtosis = st.text_input("curtosis", placeholder="Type Here")
    entropy = st.text_input("entropy", placeholder="Type Here")

    if st.button("Get Prediction"):
        output = predict_banknote(variance, skewness, curtosis, entropy)
        st.success(f'Result: {output}.')
        st.write('1 = banknote is genuine, 0 = banknote is forged')

if __name__=='__main__':
    main()
