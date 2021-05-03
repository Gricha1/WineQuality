#!/usr/bin/env python
# coding: utf-8

# In[90]:


import pandas as pd
import numpy as np
from math import ceil


# In[161]:


data = pd.read_csv("winequalityN.csv")


# In[92]:


data


# In[93]:


data.info()


# Сразу же чекним наши данные на Nan

# In[94]:


data.isna().sum().sum()


# 38 относительно 6497 довольно мало, значит просто выкенем обьекты с Nan

# In[162]:


data = data.dropna()


# In[163]:


features = data.columns[:-1]
target = data.columns[-1]


# Ну по заветам Шевниной начинаем первый этап, это визуализация данных гистограммы все дела

# In[97]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[98]:


fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.hist(data[target])
ax.set(xlabel='quality')
None


# Как же мало обьектов 3го качества и 9го в разбиении на трейн и тест надо будет учесть

# In[99]:


print(data[data[target]==3].shape,data[data[target]==9].shape)


# In[100]:


n=ceil(np.sqrt(len(list(features))))
fig=plt.figure(figsize=(15,15))
for ind,col in enumerate(features):
    ax=fig.add_subplot(n,n,ind+1)
    ax.hist(data[col])
    


# Да вродебы шикарные данные, видимо тут что то простое совсем придется обучать
# 
# Наш следующий этап - будем исследовать зависимости между признаками и отбирать те, что нужны для модели

# In[164]:


data['type'].unique() #Кекв делаем onehot


# In[165]:


data[features]


# In[166]:


data= pd.concat([data[features[1:]],pd.get_dummies(data['type']),data[target]],axis=1)


# In[167]:


data.corr().iloc[:,-1]


# Есть то, от чего наши данные не плохо зависят думаю 0.2 будет граница для взятия 

# In[168]:


(abs(data.corr().iloc[:,-1])>=0.2)


# alcohol, density, chlorides, volatile acidity

# In[169]:


features = ["alcohol", "density", "chlorides", "volatile acidity"]


# In[170]:


data = data[features+[target]]


# Ну и обязательно глянем не зависимы ли между собой наши признаки(отобранные)

# In[139]:


data.corr().style.background_gradient(cmap='coolwarm')


# Ех, к сожалению наши признаки все таки как то зависят друг от друга
# 
# Теперь разобьем данные
# Придумаем себе тестовую выборку, чтобы она была каждый раз одна и та же зададим random_seed=42

# In[140]:


from sklearn.model_selection import train_test_split


# In[171]:


X_train,X_test,Y_train,Y_test = train_test_split(data[features],data[target],train_size=0.7,random_state=42)


# In[148]:


plt.hist(Y_train)
plt.hist(Y_test)
None


# Ну норм разбило данные в принципе классы в трейне у нас все есть, причем в том нормальном количестве, будем обучать

# In[190]:


from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer


# In[151]:


criterion = ["gini", "entropy"]
max_depth = [5,10,20,30,50]
min_samples_split = [2,5,8]
grid_params = {"criterion":criterion,"max_depth":max_depth,"min_samples_split":min_samples_split}


# In[227]:


def sc(y_true,y_pred):
    roc_auc = roc_auc_score(y_true,y_pred,multi_class='ovr')
    return roc_auc
    


# In[228]:


score = make_scorer(sc)


# In[231]:


gridCV = GridSearchCV(DTC(),grid_params,error_score='raise',cv=5,refit=True,n_jobs=-1)


# In[232]:


gridCV.fit(data[features],data[target])
None


# In[215]:


y_pred = gridCV.best_estimator_.predict_proba(X_test)


# In[218]:


roc_auc_score(Y_test,y_pred,multi_class="ovr")


# In[236]:


n_neighbors = [5,8,10,15,20,30]
metric = ["euclidean","minkowski","chebyshev","manhattan"]
leaf_size = [18,23,30,35,39,40,50]
grid_params = {"n_neighbors":n_neighbors,"metric":metric,"leaf_size":leaf_size}


# In[237]:


gridCV = GridSearchCV(KNN(),grid_params,error_score='raise',cv=5,refit=True,n_jobs=-1)


# In[238]:


gridCV.fit(data[features],data[target])


# In[239]:


y_pred = gridCV.best_estimator_.predict_proba(X_test)
roc_auc_score(Y_test,y_pred,multi_class="ovr")


# In[240]:


gridCV.best_params_


# Замечаем, что малое количество размера листьев нам помогло, попробуем его еще больше уменьшить

# In[274]:



sizes = list(range(7,30))
metrics=[]
for size in sizes:
    clf=KNN(leaf_size= size, metric= 'manhattan', n_neighbors= 30)
    clf.fit(X_train,Y_train)
    y_pred = clf.predict_proba(X_test)
    roc = roc_auc_score(Y_test,y_pred,multi_class='ovr')
    metrics.append(roc)
plt.plot(sizes,metrics)
None


# In[276]:


max(metrics)


# На данный момент наилучшую метрику выдает KNN с параметрами {'leaf_size': 18, 'metric': 'manhattan', 'n_neighbors': 30}
# у которого roc = 0.8602177441458556
