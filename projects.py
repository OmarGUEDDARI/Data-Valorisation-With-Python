import pandas
import numpy
import streamlit
Projects=dict({'Project_Name':['After Sales Analysis','Object Detection'] ,'Tags' : ['NLP','DL']})
streamlit.selectbox('Projects',options=Projects['Project_Name'])
