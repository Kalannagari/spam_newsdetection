import pandas as pd
import numpy as np
fake=pd.read_csv('Fake.csv')
true=pd.read_csv('True.csv')
true['lable']=0
fake['lable']=1
dataset1=true[['text','lable']]
dataset2=fake[['text','lable']]
dataset=pd.concat([dataset1,dataset2])
dataset=dataset.sample(frac=1)
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stopwords=stopwords.words('english')
ps=WordNetLemmatizer()
def clear_row(row):
    row=row.lower()
    row=re.sub('[^a-zA-Z]',' ', row)
    token=row.split()
    news=[ps.lemmatize(word) for word in token if not word in  stopwords]
    cnews=' '.join(news)
    return cnews
dataset['text']= dataset['text'].apply(lambda x :clear_row(x))
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer= TfidfVectorizer(max_features=50000,lowercase=False,ngram_range=(1,2))
x=dataset.iloc[:3500,0]
y=dataset.iloc[:3500,1]    
from sklearn.model_selection import train_test_split
train_data , test_data, train_lable, test_lable = train_test_split(x,y,test_size=0.2,random_state=0)
vec=vectorizer.fit_transform(train_data)
vec=vec.toarray()
vec1=vectorizer.fit_transform(test_data)
vec1=vec1.toarray()
train_data = pd.DataFrame(vec, columns=vectorizer.get_feature_names_out())
test_data = pd.DataFrame(vec1, columns=vectorizer.get_feature_names_out())
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(train_data,train_lable)
y_pred=clf.predict(test_data)
from sklearn.metrics import accuracy_score
accuracy_score(test_lable,y_pred)



