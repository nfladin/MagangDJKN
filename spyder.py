import pandas as pd
import re
import nltk
import numpy as np
from sklearn.model_selection import LeaveOneOut
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
#from sklearn import preprocessing
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
#from sklearn.naive_bayes import GaussianNB
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import  RandomForestClassifier
from sklearn.svm import SVC
#from numpy import reshape
#from array import *
#import re 
import requests
from bs4 import BeautifulSoup 
import mysql.connector 
#from textblob import TextBlob
# from googletrans import Translator


tokenized = []
word_index_map = {}
cur_index = 0
data = []            
#i=0
keyword = np.genfromtxt("keyword.txt", delimiter = ",", dtype="str") 
df = pd.read_csv('data500.csv') 
cleanr = re.compile('<.*?>') 
nltk.download('punkt')

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="nopalganteng",
  database="scrap"
) 

def process(s):
    stopwords = StopWordRemoverFactory().get_stop_words()
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    s = s.lower()
    s = re.sub(r'\d+', '', s)
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

def tokens_vector(tokens, label):
    k = np.zeros(len(word_index_map)+1)
    for t in tokens:
        if t in word_index_map:
            i = word_index_map[t]
            k[i] += 1 
    if k.sum() > 0:
        k = k / k.sum()
    k[-1] = label
    return k

#CRAWLING DATA
def titletrack():
  url = 'https://finance.detik.com/indeks'
  response = requests.get(url) 
  soup = BeautifulSoup(response.text, 'lxml')
  title = soup.find_all('h3', {'class':'media__title'})[0].find('a').text
  return title
  #print(title)
def timetrack():
  url = 'https://finance.detik.com/indeks'
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'lxml')
  time = soup.find_all('div', {'class':'media__date'})[0].find('span')
  return time['title']
def urltrack():
  url = 'https://finance.detik.com/indeks'
  response = requests.get(url) 
  soup = BeautifulSoup(response.text, 'lxml')
  r = soup.find_all('a', class_="media__link", href=True, text=True)[0]
  return r['href']

for i in df.index:
    lb = str(df['KATEGORI'][i])
    x = str(df['ISI'][i]) 
    tokens = process(x) 
    tokenized.append({'token':tokens,'label':lb})
    for token in tokens:
        #print(token)
        if token not in word_index_map:
            word_index_map[token] = cur_index
            cur_index += 1 

for tokens in tokenized:
    y = tokens_vector(tokens["token"], tokens["label"])
    data.append(y.tolist()) 
    #i += 1
    
data = np.array(data)
X = data[:,:-1]  
y = data[:,-1] 
loo = LeaveOneOut()
loo.get_n_splits(X)
#model = LogisticRegression() #0,63 
#model = GaussianNB() #0,65 0,78
#model = KNeighborsClassifier() #0,62
#model = RandomForestClassifier(n_estimators=20) #0,63 0.77
model = SVC() #0,65 0,79 
scores = [] 
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train,y_train) 
    scores.append(model.score(X_test,y_test))
print(np.array(scores).sum()/len(scores))


mycursor = mydb.cursor()
sql = "INSERT INTO scrap (title, date, url, content, author, tag, category, sentiment, label, encode) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
data_crwl = [] 
title1 = str(titletrack())
#time1 = str(timetrack())
url1 = str(urltrack())

while True:
    title2 = str(titletrack())
    #time2 = str(timetrack())
    import datetime
    time2 = str(datetime.date.today())
    url2 = str(urltrack())
    if title1 != title2 and url1 != url2 :
        response = requests.get(url2) 
        soup = BeautifulSoup(response.text, 'lxml')
        konten = str(soup.find_all('p'))
        konten2 = soup.find('a', class_="btn btn--blue-base btn--sm mgb-24",href=True)
        if konten2 == None:
            konten2 = ''
        else:
            urlv2 = konten2['href']
            responsev2 = requests.get(urlv2) 
            soupv2 = BeautifulSoup(responsev2.text, 'lxml')
            konten2 = str(soupv2.find_all('p'))
        author = str(soup.find_all('div', {'class':'detail__author'}))
        tag = str(soup.find_all('div', {'class':'nav'}))
        kategori = str(soup.find_all('div', {'class':'page__breadcrumb'}))
        #cleanr = re.compile('<.*?>')
        konten = re.sub(cleanr,'',konten)
        konten2 = re.sub(cleanr,'',konten2) 
        author = re.sub(cleanr,'',author)
        tag = re.sub(cleanr,'',tag)
        tag = re.sub('\n','',tag)
        kategori = re.sub(cleanr,'',kategori)
        kategori = re.sub('\n','',kategori)
        #data_crwl.extend((title2,time2,url2,konten,konten2,author,tag,kategori))
        data_crwl.append(title2)
        data_crwl.append(time2)
        data_crwl.append(url2)
        data_crwl.append(konten)
        data_crwl.append(konten2)
        data_crwl.append(author)
        data_crwl.append(tag)
        data_crwl.append(kategori)
        title1 = title2
        #time1 = time2
        url1 = url2
        x = str(data_crwl[3] + data_crwl[4])
        x = x.lower()
        for key in keyword:
            if (x.find(key) != -1):
                print(x)
                print(key)
                tokens = process(x)
                tokenized.append({'token':tokens,'label':1})
                data_test = []
                for tokens in tokenized:
                    y = tokens_vector(tokens["token"], tokens["label"])
                    data_test.append(y.tolist())
                data_test = np.array(data_test)
                X_crwl = data_test[:,:-1]  
                y_crwl = data_test[:,-1]
                hasil = model.predict(X_crwl)
                hasil = hasil[len(hasil)-1]
                if hasil == 0.0:
                    label = 'NEGIATIF'
                    sentiment = str(sum(X_crwl[len(X_crwl)-1])*-1)
                    encode = '-1'
                elif hasil == 1.0:
                    label = 'NETRAL'
                    sentiment = '0'
                    encode = '0'
                elif hasil == 2.0:
                    label = 'POSITIF' 
                    sentiment = str(sum(X_crwl[len(X_crwl)-1]))
                    encode = '1'
           
                val = (data_crwl[0], data_crwl[1], data_crwl[2], x, data_crwl[5], data_crwl[6], data_crwl[7], sentiment, label, encode)
                mycursor.execute(sql, val)
                mydb.commit()
                #data_crwl = []
                print("######CLEAR######")
        data_crwl = [] 
        print("LOAD NEW NEWS") 




