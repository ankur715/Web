#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
time_start = time.time()


# In[2]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('testing_dataset.csv')
df.drop('Unnamed: 0', inplace = True, axis = 1)
df.head()


# In[4]:


## Check NA's
df.isnull().sum()


# In[5]:


df['starbucks_flag'].value_counts()


# In[6]:


for i in range(len(df)):
    wd = df.loc[i,'weekday_tran']
    wk = df.loc[i,'weekend_tran']
    m = round(np.average([wd,wk], weights=[5,2]),0)
    df.loc[i,'tran_mean'] = m 


# In[7]:


df.head(7)


# In[8]:


ct = df[['starbucks_flag','city']]
ct.groupby(['city','starbucks_flag']).size()


# In[9]:


pd.options.display.float_format = '{:.2f}'.format
dg = df.drop(['CT','star_count'], axis = 1)
dg.groupby(['starbucks_flag','city']).mean()


# In[10]:


dg.groupby(['starbucks_flag','city']).median()


# In[11]:


dg.groupby(['starbucks_flag','city']).std()


# In[12]:


dg1 = df.drop(['CT','star_count','city'], axis = 1)
dg1.groupby(['starbucks_flag']).mean()


# In[13]:


dg1.groupby(['starbucks_flag']).median()


# In[14]:


df[['total_population', 'median_age',
       'total_households', 'median_hh_income', 'median_rent',
       'median_home_value', 'percent_workers', 'percent_leave_7_9',
       'perc_hs_dipl', 'perc_bach_deg', 'perc_masters_deg',
       'perc_walk_to_work', 'perc_car_to_work', 'perc_pub_tran',
       'perc_bicycle_to_work', 'perc_work_from_home', 'time_to_work',
       'weekday_tran', 'weekend_tran','tran_mean']].skew()


# In[15]:


def box_plot(xx):
    plt.subplots(figsize=(12,10))
    plt.subplot(211)
    plt.title(xx)
    sns.boxplot(x="starbucks_flag", y=xx,
            hue="city",
            data=df)
    plt.subplot(212)
    plt.title(xx)
    sns.boxplot(x="starbucks_flag", y=xx,
            data=df)


# In[16]:


col_list = ['total_population', 'median_age',
       'total_households', 'median_hh_income', 'median_rent',
       'median_home_value', 'percent_workers', 'percent_leave_7_9',
       'perc_hs_dipl', 'perc_bach_deg', 'perc_masters_deg',
       'perc_walk_to_work', 'perc_car_to_work', 'perc_pub_tran',
       'perc_bicycle_to_work', 'perc_work_from_home', 'time_to_work',
       'weekday_tran', 'weekend_tran','tran_mean']


# In[17]:


for i in col_list:
    box_plot(i)


# In[18]:


### Correlation plot
number = df[['total_population', 'median_age',
       'total_households', 'median_hh_income', 'median_rent',
       'median_home_value', 'percent_workers', 'percent_leave_7_9',
       'perc_hs_dipl', 'perc_bach_deg', 'perc_masters_deg',
       'perc_walk_to_work', 'perc_car_to_work', 'perc_pub_tran',
       'perc_bicycle_to_work', 'perc_work_from_home', 'time_to_work',
       'weekday_tran', 'weekend_tran','star_count','tran_mean']]
cor = number.corr()
f,ax = plt.subplots(figsize = (18,18))
plt.title(' Correlation Heatmap')
sns.heatmap(cor, annot = True, linewidths = .5, fmt='.3f', ax = ax)


# ### Train with manhattan and philadelphia, test on chicago

# In[19]:


df1 = df.copy()
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
lc = LabelEncoder()
df1['class'] = lc.fit_transform(df1["starbucks_flag"])


# In[20]:


cot = df1['class'].value_counts()
print(cot)
print('% non starbucks', cot[0]/cot.sum())
print('% starbucks', cot[1]/cot.sum())


# In[21]:


df1.drop('total_households', inplace = True, axis = 1)


# In[22]:


df1.head(2)


# In[23]:


test1 = df1[df1['city'] == 'chicago']
train1 = df1[df1['city'] != 'chicago']
train1['class'].value_counts()


# In[24]:


# df_0 = train1[train1['class'] == 0]
# df_1 = train1[train1['class'] == 1]
# df_under = df_0.sample(300 ,random_state=19)
# train1 = pd.concat([df_under, df_1], axis = 0)
# train1['class'].value_counts()


# In[25]:


train1.reset_index(inplace = True)
train1.drop('index', inplace = True, axis = 1)
train = train1.drop(['CT','tract_name', 'city','starbucks_flag','star_count','weekend_tran','weekday_tran'], axis = 1)
train.head()


# In[26]:


test1.reset_index(inplace = True)
test1.drop('index', inplace = True, axis = 1)
test = test1.drop(['CT','tract_name', 'city','starbucks_flag','star_count','weekend_tran','weekday_tran'], axis = 1)
test.tail()


# In[27]:


X_train = train.drop('class', axis = 1)
y_train = train[['class']]
X_test = test.drop('class', axis = 1)
y_test = test[['class']]


# In[28]:


sc = MinMaxScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# In[29]:


from sklearn.model_selection import GridSearchCV
log = LogisticRegression(random_state=0)
penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)
clf = GridSearchCV(log, hyperparameters, cv=5, verbose=1,n_jobs= -1,scoring = 'roc_auc' )
best_model = clf.fit(X_train_scaled, y_train)


# In[30]:


print('Best Parameters:', best_model.best_params_)
print('Best Score:',best_model.best_score_)


# In[31]:


log1 = LogisticRegression(C = 2.7825594022071245,penalty = 'l1' ,random_state=0)
log1.fit(X_train_scaled, y_train)


# In[32]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = log1.predict(X_train_scaled)
cm = confusion_matrix(y_train, pred)


# In[33]:


print('accuracy:', accuracy_score(y_train,pred))
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm, annot =True, linewidth = 0.5, linecolor = "red", fmt = ".0f", ax =ax)
plt.title("Training Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.yticks( va="center")


# In[34]:


pred_chi = log1.predict(X_test_scaled)
cm_Chi = confusion_matrix(y_test, pred_chi)
print('accuracy:', accuracy_score(y_test,pred_chi))
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm_Chi, annot =True, linewidth = 0.5, linecolor = "red", fmt = ".0f", ax =ax)
plt.title("Test Dataset (Chicago)")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.yticks( va="center")


# In[35]:


prob_chi = log1.predict_proba(X_test_scaled)
chi_prob = pd.DataFrame(prob_chi, columns = ['no_starbucks', 'starbucks'])
chi = pd.concat([test1, chi_prob] ,axis = 1)
chi.head()


# In[36]:


non_star = chi[chi['starbucks_flag'] == 'Non-Starbucks']
non_star2 = non_star[['CT', 'tract_name','no_starbucks', 'starbucks']]
max(non_star['starbucks'])


# In[37]:


chi_top5 = non_star2.nlargest(5, ['starbucks'])
chi_top5


# In[38]:


top5 = set(chi_top5['CT'])
res1 = test1[test1.CT.isin(top5)]
res1


# In[39]:


test1.groupby('city').mean()


# In[40]:


chi['new_starbucks'] = np.where(chi['CT'].isin(res1['CT']),'New_location',chi['starbucks_flag'] )
chi['new_starbucks2'] = np.where(chi['CT'].isin(res1['CT']),chi['tract_name'],chi['starbucks_flag'] )
chi[chi['CT'].isin(res1['CT'])]


# In[41]:


chi.to_csv('chi_prob.csv')
chi.head()


# ### Train with manhattan and chicago, test on philadelphia

# In[42]:


test2 = df1[df1['city'] == 'philadelphia']
train2 = df1[df1['city'] != 'philadelphia']
print('Train')
print(train2['class'].value_counts())
print('Test')
print(test2['class'].value_counts())


# In[43]:


train2.reset_index(inplace = True)
train2.drop('index', inplace = True, axis = 1)
train = train2.drop(['CT','tract_name', 'city','starbucks_flag','star_count','weekend_tran','weekday_tran'], axis = 1)
train.head()


# In[44]:


test2.reset_index(inplace = True)
test2.drop('index', inplace = True, axis = 1)
test = test2.drop(['CT','tract_name', 'city','starbucks_flag','star_count','weekend_tran','weekday_tran'], axis = 1)
test.head()


# In[45]:


X_train = train.drop('class', axis = 1)
y_train = train[['class']]
X_test = test.drop('class', axis = 1)
y_test = test[['class']]


# In[46]:


sc = MinMaxScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# In[47]:


log22 = LogisticRegression(random_state=0)
penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)
clf = GridSearchCV(log22, hyperparameters, cv=5, verbose=1,n_jobs= -1,scoring = 'roc_auc' )
best_model = clf.fit(X_train_scaled, y_train)


# In[48]:


print('Best Parameters:', best_model.best_params_)
print('Best Score:',best_model.best_score_)


# In[49]:


log2 = LogisticRegression(C = 1.0,penalty = 'l2' ,random_state=0)
log2.fit(X_train_scaled, y_train)


# In[50]:


pred2 = log2.predict(X_train_scaled)
cm2 = confusion_matrix(y_train, pred2)
print('accuracy:', accuracy_score(y_train,pred2))
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm2, annot =True, linewidth = 0.5, linecolor = "red", fmt = ".0f", ax =ax)
plt.title("Training Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.yticks( va="center")


# In[51]:


pred_phi = log2.predict(X_test_scaled)
cm_phi = confusion_matrix(y_test, pred_phi)
print('accuracy:', accuracy_score(y_test,pred_phi))
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm_Chi, annot =True, linewidth = 0.5, linecolor = "red", fmt = ".0f", ax =ax)
plt.title("Test Dataset (Chicago)")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.yticks( va="center")


# In[52]:


prob_phi = log2.predict_proba(X_test_scaled)
phi_prob = pd.DataFrame(prob_phi, columns = ['no_starbucks', 'starbucks'])
phi = pd.concat([test2, phi_prob] ,axis = 1)
phi.head()


# In[53]:


non_star = phi[phi['starbucks_flag'] == 'Non-Starbucks']
non_star2 = non_star[['CT', 'tract_name','no_starbucks', 'starbucks']]
max(non_star['starbucks'])


# In[54]:


phi_top5 = non_star2.nlargest(5, ['starbucks'])
phi_top5


# In[55]:


p_top5 = set(phi_top5['CT'])
res2 = test2[test2.CT.isin(p_top5)]
res2.head()


# In[56]:


test2.groupby('city').mean()


# In[57]:


phi['new_starbucks'] = np.where(phi['CT'].isin(res2['CT']),'New_location',phi['starbucks_flag'] )
phi['new_starbucks2'] = np.where(phi['CT'].isin(res2['CT']),phi['tract_name'],phi['starbucks_flag'] )
phi[phi['CT'].isin(res2['CT'])]


# In[58]:


phi.to_csv('phi_prob.csv')
phi.head()


# ### Train with philadelphia and chicago, test on manhattan

# In[59]:


test3 = df1[df1['city'] == 'manhattan']
train3 = df1[df1['city'] != 'manhattan']
print('Train')
print(train3['class'].value_counts())
print('Test')
print(test3['class'].value_counts())


# In[60]:


train3.reset_index(inplace = True)
train3.drop('index', inplace = True, axis = 1)
train = train3.drop(['CT','tract_name', 'city','starbucks_flag','star_count','weekend_tran','weekday_tran'], axis = 1)
train.tail()


# In[61]:


test3.reset_index(inplace = True)
test3.drop('index', inplace = True, axis = 1)
test = test3.drop(['CT','tract_name', 'city','starbucks_flag','star_count','weekend_tran','weekday_tran'], axis = 1)
test.tail()


# In[62]:


X_train = train.drop('class', axis = 1)
y_train = train[['class']]
X_test = test.drop('class', axis = 1)
y_test = test[['class']]


# In[63]:


sc = MinMaxScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# In[64]:


log33 = LogisticRegression(random_state=0)
penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)
clf = GridSearchCV(log33, hyperparameters, cv=5, verbose=1,n_jobs= -1,scoring = 'roc_auc' )
best_model = clf.fit(X_train_scaled, y_train)


# In[65]:


print('Best Parameters:', best_model.best_params_)
print('Best Score:',best_model.best_score_)


# In[66]:


log3 = LogisticRegression(C = 1.0,penalty = 'l2' ,random_state=0)
log3.fit(X_train_scaled, y_train)


# In[67]:


import pickle

pickle.dump(log3, open('model.pkl','wb'))


# In[68]:


pred3 = log3.predict(X_train_scaled)
cm3 = confusion_matrix(y_train, pred3)
print('accuracy:', accuracy_score(y_train,pred3))
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm3, annot =True, linewidth = 0.5, linecolor = "red", fmt = ".0f", ax =ax)
plt.title("Training Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.yticks( va="center")


# In[69]:


pred_man  = log3.predict(X_test_scaled)
cm_man = confusion_matrix(y_test, pred_man)
print('accuracy:', accuracy_score(y_test,pred_man))
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm_Chi, annot =True, linewidth = 0.5, linecolor = "red", fmt = ".0f", ax =ax)
plt.title("Test Dataset (Chicago)")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.yticks( va="center")


# In[70]:


prob_man = log3.predict_proba(X_test_scaled)
man_prob = pd.DataFrame(prob_man, columns = ['no_starbucks', 'starbucks'])
man = pd.concat([test3, man_prob] ,axis = 1)
man.head()


# In[71]:


non_star = man[man['starbucks_flag'] == 'Non-Starbucks']
non_star2 = non_star[['CT', 'tract_name','no_starbucks', 'starbucks']]
max(non_star['starbucks'])


# In[72]:


man_top5 = non_star2.nlargest(5, ['starbucks'])
man_top5


# In[73]:


m_top5 = set(man_top5['CT'])
res3 = test3.loc[test3.CT.isin(m_top5)]
res3


# In[74]:


test3.groupby('city').mean()


# In[75]:


man['new_starbucks'] = np.where(man['CT'].isin(res3['CT']),'New_location',man['starbucks_flag'] )
man['new_starbucks2'] = np.where(man['CT'].isin(res3['CT']),man['tract_name'],man['starbucks_flag'] )
man[man['CT'].isin(res3['CT'])]


# In[76]:


man.to_csv('man_prob.csv')
man.tail()


# In[77]:


time_stop = time.time()-time_start
time_stop


# In[81]:


len(X_train_scaled)


# In[84]:


len(X_train.columns)


# In[91]:


X_train.mean()


# In[88]:


X_train.columns

