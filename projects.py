import pandas
import numpy
import streamlit
import mysql.connector
import pyspark
import sklearn
import emoji
import nltk
from pyspark import SparkContext
from apiclient.discovery import build
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
#connexion=mysql.connector.connect(user='sql11436700', database='sql11436700',password='rPVRabqddG',port='3306',host='sql11.freesqldatabase.com')
Projects=dict({'Project_Name':['Data Science','Youtube Comments','Object Detection'] ,'Tags' : ['Discovery','NLP','DL']})
ML_Steps=['Preprocessing','Feature Selection','Model Selection','Model Tuning','Deployment']
step=''
url=''
pies=plt.subplots()
youtube = build('youtube','v3',developerKey="AIzaSyBTPzZunEN9bWskxKh2xoUvBreUoB4QHhk")
comments,Feel_viz,Sentiment,Feelings=pandas.Series(),pandas.DataFrame(),pandas.DataFrame(),pandas.DataFrame(columns=['Negative','Positive','Neutral','Compound'])
iterator=1
nltk.download('punkt')
nltk.download('vader_lexicon')
sia=SentimentIntensityAnalyzer()
def youtube_extraction(url):
    global comments
    global iterator
    comments=pandas.DataFrame(columns=['id','comment'])
    video_id=url.split('?v=')[1][0:11]
    video_response=youtube.commentThreads().list(
    part='snippet,replies',
    videoId=video_id).execute()
    while iterator<1000 :  
        for item in video_response['items']:
            comment =dict({'id':[iterator],'comment':[item['snippet']['topLevelComment']['snippet']['textDisplay']]})
            comments=comments.append(comment,ignore_index=True)
            iterator+=1
    return comments 
def youtube_comments_processing(comments):
    global Sentiment
    comments['comment_demojized']=comments['comment'].apply(lambda x : emoji.demojize(x[0]))
    comments['comment_demojized']=comments['comment_demojized'].str.lower()
    comments['Sentiment']=comments['comment_demojized'].apply(lambda x :  sia.polarity_scores(x))
    Sentiment=comments[['comment','Sentiment']]
    return Sentiment
def youtube_sentiment_viz(Sentiment):
    global Feelings
    global Feel_viz
    global pies
    for feeling in Sentiment['Sentiment']:
       Feelings=Feelings.append({'Negative':feeling['neg'],'Positive':feeling['pos'],'Neutral':feeling['neu'],'Compound':feeling['compound']},ignore_index=True)
    Feelings['Feeling']=Feelings.idxmax(axis=1)
    Feel_viz=pandas.Series([1 for i in range (len(Feelings['Feeling']))],index=Feelings['Feeling'],name='counts')
    Feel_viz=Feel_viz.groupby(level=0).sum()
    fig1, ax1 = plt.subplots()
    ax1.pie(Feel_viz, labels = Feel_viz.index, autopct='%1.1f%%',
        shadow=True, startangle=90)
    streamlit.pyplot(fig1)
    return Feelings ,Feel_viz
streamlit.title('Welcome to This App about Data Science Click Choose from the menu below!!')
Project=streamlit.selectbox('Projects',options=Projects['Project_Name'])
if 'Data' in Project :
    step=streamlit.radio(label='Steps',options=ML_Steps)
if 'Comments' in Project :
    url=streamlit.text_input("Youtube Video URL", '')
    if len(url)==0 or "https://www.youtube.com/watch?v=" not in url:
        streamlit.write('Please enter a valid Youtube Video URL')
    else:
        youtube_extraction(url)
        youtube_comments_processing(comments)
        youtube_sentiment_viz(Sentiment)
