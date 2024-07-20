#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head(1)


# In[4]:


movies=movies.merge(credits,on='title')


# In[5]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[6]:


movies.head()


# In[7]:


movies.isnull().sum()


# In[8]:


movies.dropna(inplace=True)


# In[9]:


movies.duplicated().sum()


# In[10]:


import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[11]:


movies['genres'] = movies['genres'].apply(convert)


# In[12]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[13]:


movies.head(1)


# In[14]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[15]:


movies['cast'] = movies['cast'].apply(convert3)


# In[16]:


movies.head(1)


# In[17]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[18]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[19]:


movies.head(1)


# In[20]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[21]:


movies.head(1)


# In[22]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[23]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[24]:


movies.head(1)


# In[25]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[26]:


movies.head(1)


# In[27]:


new_df = movies[['movie_id','title','tags']]


# In[28]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[29]:


new_df['tags'][0]


# In[30]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[35]:


new_df.head()


# In[36]:


import nltk


# In[37]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[38]:


def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[39]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[40]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[41]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[42]:


from sklearn.metrics.pairwise import cosine_similarity


# In[43]:


similarity = cosine_similarity(vectors)


# In[44]:


similarity[0]


# In[45]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x: x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[50]:


recommend('Avatar')


# In[92]:


import pickle


# In[95]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[96]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




