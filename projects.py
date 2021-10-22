import pandas
import numpy
import streamlit
import graphviz
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
graph=graphviz.Digraph()
#connexion=mysql.connector.connect(user='sql11436700', database='sql11436700',password='rPVRabqddG',port='3306',host='sql11.freesqldatabase.com')
Projects=dict({'Project_Name':['Data Science','Youtube Comments'] ,'Tags' : ['Discovery','NLP','DL']})
ML_Steps=['Project Understanding','Preprocessing','Feature Selection','Model Selection','Model Tuning','Deployment']
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
def steps_description(step):
    if 'Project' in step:
        streamlit.write('Understanding the business project and translating it into an adequat data science project is an essential step.')
        streamlit.write('-Translating business issue and metrics into a type of data science subjects (regression,classification,clustering) is the first thing to do . ')
        streamlit.write('-Exchanging with business partners and information system collegues in order to identify data sources.')
        streamlit.write('-Exploring the different data sources , their frequency , quality in order to audit data .')
        streamlit.write('-Creating an ETL for the project data acquisition can help in the deployment.')
        streamlit.write("Once the project defined , it fills generally in one of those cases , other real life problems are part of reinforcement learning or semi-supervised learning , they are not present in the next tree.")
        streamlit.title('Machine Learning :Supervised & Unsupervised')
        graph.edge('Machine Learning','Supervised Learning',label='Mapping a function from features to target')
        graph.edge('Machine Learning','Unsupervised Learning', label='Finding Patterns in the data set')
        graph.edge('Supervised Learning','Regression',label='The Target is numeric.')
        graph.edge('Supervised Learning','Classification',label='The target variable is categorical.')
        graph.edge('Unsupervised Learning','Clustering',label='structuring the data set into homogeneous clusters')
        graph.edge('Unsupervised Learning','Association',label='Find rules between the features')
        streamlit.graphviz_chart(graph)
    if 'Preprocessing' in step :
        streamlit.write('Preprocessing data is a necessary step , the idea is to transform the data set  from raw data to the standards of the machine learning algorithm they are inserted in .')
        streamlit.write('Understanding the project , exploring the data set and setting the algorithms to run defines the steps of preprocessing to be done . ')
        streamlit.write('Here are some of the steps :')
        streamlit.title('Standardization')
        streamlit.write('Some Machine Learning algorithms , are sensitive to input values scales , by using Mean Removal and Variance Scaling , Feature importance can be put to adequate standard')
        streamlit.title('Encoding Categorical Features')
        streamlit.write('Changing variable type from text to numeric , it can be done using OneHotEncoder method or a specific built in function')
        streamlit.title('Discretization')
        streamlit.write('Changing continuous variables into discrete ones with defined values')
        streamlit.title('Imputation of missing values')
        streamlit.write('it can be done in a simple way using the mean of variable values , or in a more sofisticated way  using k-means for example of the k neighbors to the missing value')
        streamlit.write('marking the imputed values can be added in training ML models for quality purpose')
    if 'Feature' in step :
        streamlit.write('Features that are generally empty or are have constant values are useless.')
        streamlit.title('PCA')
        streamlit.write('For memory purpose in some projects , dimension reduction can be useful')
        streamlit.write('PCA stands for Principal Component Analysis it tends to decompose a multivariate dataset into components that explains the major part of the variance.')
    if 'Model Selection' in step:
        streamlit.write('In order to choose the right algorithm , criterions have to be defined , some common metrics are : score , timing .')
        streamlit.title('Score')
        streamlit.write('Each type of project got a hand of accuracy metrics that help judge the performance , this metric can be personnalized .')
        streamlit.title('Time')
        streamlit.write('Running time is important once the algorithms need to be putted in production.')
        streamlit.title('Some Popular Machine Learning Algorithms')
        streamlit.title('Supervised Learning : Regression')
        streamlit.write('Linear Regression-Decision Tree-Random Forest-Gradient Boosting Tree')
        streamlit.title('Supervised Learning : Classification')
        streamlit.write('Logistic Regression-Decision Tree-Random Forest-Gradient Boosting Tree-Support Vector Machine-Naive Bayes')
        streamlit.title('Unsupervised Learning : Clustering')
        streamlit.write('Hierarchical-K-means-DBSCAN-Isolation Forest')
    if 'Model Tuning' in step :
        streamlit.write('Each algorithm got a set of parameters that define it\'s behaviour .')
        streamlit.write('Tuning those parameters is essential so that the algorithm respond correclty to the given task .')
        streamlit.write('Tuning is generally done with cross validation technique to evaluate the generalisation capacity under a given set of parameters.')
        streamlit.write('The search of the optimal parameters can be done with one of the following methods.')
        streamlit.title('Grid Search')
        streamlit.write('given a set of values for each hyperparameter ,all combinations are tested and the one with the best performance evaluated for example by a cross validation is chosen.')
        streamlit.title('Randomized Search')
        streamlit.write('given a set of values for each hyperparameter , some combination randomly are tested and the one with the best performance is chosen.')
        streamlit.title('Bayesian Search')
        streamlit.write('given a set of values for each hyperparameter, by predicting the outcome of the evaluation function , a smart search in the set of hyperparameters is done iteratively.' )
    if 'Deployment' in step :
        streamlit.write('-Creating a pipeline of all the transformation happening in the data from the source to the final outcome .')
        streamlit.write('-Pushing the pipeline script in cloud or a local server with a specific time for running')
        streamlit.write('-Monitor the running by creating log files or using existing tools')
streamlit.title('Welcome to This App about Data Science Choose from the menu below!!')
Project=streamlit.selectbox('Projects',options=Projects['Project_Name'])
if 'Data' in Project :
    step=streamlit.sidebar.radio(label='Steps',options=ML_Steps)
    steps_description(step)
if 'Comments' in Project :
    url=streamlit.text_input("Youtube Video URL", '')
    if len(url)==0 or "https://www.youtube.com/watch?v=" not in url:
        streamlit.write('Please enter a valid Youtube Video URL')
    else:
        youtube_extraction(url)
        youtube_comments_processing(comments)
        youtube_sentiment_viz(Sentiment)
