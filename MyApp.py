import streamlit as st
import numpy as np
from Functions import load_data

# load data
df,x,y,seed = load_data()

st.title("Titanic Survival App")
st.write("""
         # Hello worl- *You!*
         \nThis page hosts a prediction model to the titanic survivability famous problem/dataset.
         """)

if st.checkbox('pre-visu'):
    st.write("\nThe dataset looks like this: ")
    n_rows = st.slider('Number or rows in pre-visualizaion')
    st.dataframe(df.head(n_rows))

select_features = st.selectbox('Select features to pre-visualization', df.columns)


st.text_input("input data", key="name")

hist_values = np.histogram(df['Age'], bins=30, range=(0,24))[0]
st.bar_chart(hist_values)

with st.form(key='passenger_input'):
             age = st.number_input("Passenger's Age")
             fare = st.number_input("Passenger's Fare value")
             st.form_submit_button('Submit passenger')




