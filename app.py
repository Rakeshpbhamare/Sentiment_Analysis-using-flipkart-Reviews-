import streamlit as st
import pickle
from PIL import Image

st.write("""
         # Sentiment Analysis(Based on flipcart Reviews)
         """
         )

Positive = Image.open('positive_emoji_2018.xl')
Negative = Image.open('negative_emoji_2018.xl')

# https://docs.streamlit.io/library/api-reference/widgets

sentence = st.input('Enter your Review')


# print(test_data)
@st.cache(allow_output_mutation = True)
def cache_model(model_name):
    model = pickle.load(open(model_name, 'rb'))
    return (model)

sentimate_model = cache_model("model.pkl")

import tensorflow as tf


if st.button('Predict'):
    

    pred_prob = sentimate_model.predict([sentence])
    pred_label = tf.squeeze(tf.round(pred_prob)).numpy()
    print(f"Pred: {pred_label}", "(Positive)" if pred_label > 0 else "(Negative)", f"Prob: {pred_prob[0][0]}")
    print(f"Text:\n{sentence}")


    # st.write(test_data)
    # st.write(result)

    if pred_label > 0:
        
        st.image(Positive, width  = 300)
        #st.write('## Please visit a cardiologist')

    else:

        st.image(Negative, width = 300)
        #st.write('## Keep up the good lifestyle! You are not at risk')
