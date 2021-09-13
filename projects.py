import pandas
import numpy
import streamlit
import mysql.connector
import pyspark
import sklearn
from pyspark import SparkContext
from apiclient.discovery import build
connexion=mysql.connector.connect(user='sql11436700', database='sql11436700',password='rPVRabqddG',port='3306',host='sql11.freesqldatabase.com')
Projects=dict({'Project_Name':['Data Science','Youtube Comments','Object Detection'] ,'Tags' : ['Discovery','NLP','DL']})
ML_Steps=['Preprocessing','Feature Selection','Model Selection','Model Tuning','Deployment']
step=''
url=''
youtube = build('youtube','v3',developerKey="AIzaSyBTPzZunEN9bWskxKh2xoUvBreUoB4QHhk")
comments=pandas.DataFrame(columns=['comment'])
iterator=1
def youtube_extraction(url):
    global comments
    global iterator
    comments=pandas.DataFrame()
    video_id=url.split('?v=')[1][0:11]
    video_response=youtube.commentThreads().list(
    part='snippet,replies',
    videoId=video_id).execute()
    while iterator<100 :
      #while video_response :  
        for item in video_response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments = comments.append(pandas.DataFrame(data=comment,inplace=True,columns=['comment'])
            print(comment)
            iterator+=1
    return comments 
streamlit.button('Hi There Welcome to This App about Data Science Click on this Button to Start!!')
Project=streamlit.selectbox('Projects',options=Projects['Project_Name'])
if 'Data' in Project :
    step=streamlit.radio(label='Steps',options=ML_Steps)
if 'Comments' in Project :
    url=streamlit.text_input("Youtube Video URL", '')
    youtube_extraction(url)
    streamlit.write(1)
