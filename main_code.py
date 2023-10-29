#!/usr/bin/env python
# coding: utf-8

# # Network Disruptions Detection
# 
# The goal of the problem is to predict a network's fault severity at a time at a particular location based on the log data available. Each row in the main dataset (train.csv, test.csv) represents a location and a time point. They are identified by the "id" column, which is the key "id" used in other data files. 
# 
# Fault severity has 3 categories: 0,1,2 (0 meaning no fault, 1 meaning only a few, and 2 meaning many). 
# 
# Different types of features are extracted from log files and other sources: event_type.csv, log_feature.csv, resource_type.csv, severity_type.csv. 
# 
# Note: “severity_type” is a feature extracted from the log files (in severity_type.csv). Often this is a severity type of a warning message coming from the log. "severity_type" is categorical. It does not have an ordering. “fault_severity” is a measurement of actual reported faults from users of the network and is the target variable (in train.csv).
# 
# File descriptions
# - train.csv - the training set for the fault severity
# - test.csv - the test set for fault severity
# - sample_submission.csv – a sample of the correct format for the input
# - event_type.csv: type of event related to the main dataset
# - log_feature.csv - features extracted from log files
# - resource_type.csv: resource type related to the main dataset
# - severity_type.csv: severity type of a warning message coming from the log
# 
# # We import libraries and data:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[2]:


event= pd.read_csv('event_type.csv')
event.head()


# In[3]:


log_feature= pd.read_csv('log_feature.csv')
log_feature.head()


# In[4]:


resource_type= pd.read_csv('resource_type.csv')
resource_type.head()


# In[5]:


severity_type= pd.read_csv('severity_type.csv')
severity_type.head()


# In[6]:


attack_intensity = pd.read_csv('train.csv')
attack_intensity.head()


# In[7]:


#to see the shapes:
print('Attack Intensity: ', attack_intensity.shape)
print ('severity_type: ', severity_type.shape)
print ('event: ', event.shape)
print ('resource_type: ', resource_type.shape)
print ('log_feature: ', log_feature.shape)


# gathering for **train**:

# In[8]:


df = attack_intensity.merge(severity_type, how = 'left', left_on='id', right_on='id')
df.head()


# In[9]:


df = df.merge(event, how = 'left', left_on='id', right_on='id')
df.head()


# In[10]:


df = df.merge(resource_type, how= 'left', left_on='id', right_on='id')
df.head()


# In[11]:


df = df.merge(log_feature, how='left', left_on='id', right_on='id')
df.head()


# We can see that there are repeated records, so we proceed to eliminate them according to the `id`.

# In[12]:


df.drop_duplicates(subset='id', inplace=True)


# In[13]:


df.reset_index(inplace=True, drop=True)


# In[14]:


df.head()


# In[15]:


df.shape


# In[16]:


df.info()


# ## **Merging Test Data**

# In[17]:


test_data = pd.read_csv('test.csv')
test_data.head()


# In[18]:


#to see the shapes:
print('Test_data')
print ('severity_type')
print ('event')
print ('resource_type')
print ("log_feature")


# In[19]:


test_df = test_data.merge(severity_type, how= 'left', left_on='id', right_on='id')
test_df .head()


# In[20]:


test_df = test_df.merge(event, how= 'left', left_on='id', right_on='id')
test_df.head()


# In[21]:


test_df = test_df.merge(resource_type, how= 'left', left_on='id', right_on='id')
test_df .head()


# In[22]:


test_df = test_df .merge(log_feature, how='left', left_on='id', right_on='id')
test_df .head()


# Dropping Duplicate Values.

# In[23]:


test_df.drop_duplicates(subset='id', inplace=True)


# In[24]:


test_df.reset_index(inplace=True, drop=True)


# In[25]:


test_df.head()


# In[26]:


test_df.shape


# In[27]:


test_df.info()


# In[28]:


df.info()


# ### **Checking If there are any null values avaliable**

# In[29]:


#Let's see if there are missing values
df.isnull().sum()


# ### **Data Preprocessing (Converting String to Integer)**

# In[30]:


df['location'] = df['location'].str.split(' ')
df['location'] = df['location'].str.get(1)

df['severity_type'] = df['severity_type'].str.split(' ')
df['severity_type'] = df['severity_type'].str.get(1)

df['event_type'] = df['event_type'].str.split(' ')
df['event_type'] = df['event_type'].str.get(1)

df['log_feature'] = df['log_feature'].str.split(' ')
df['log_feature'] = df['log_feature'].str.get(1)

df['resource_type'] = df['resource_type'].str.split(' ')
df['resource_type'] = df['resource_type'].str.get(1)


# In[31]:


df = df.apply(pd.to_numeric)


# ## **Data Visualizations**

# In[32]:


import bokeh
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.plotting import figure, show


# In[33]:


#with a seaborn heatmap
plt.figure(figsize = (12,10))
corr = df.corr()
sns.heatmap(data = corr, yticklabels=True, cbar=True, cmap="viridis", annot = True)
plt.show()


# ## **Exploratory Data Analysis.**

# In[34]:


#To view unique values ​​by column:
print ('unique for location: ', df.location.unique)
print ('unique for fault_severity: ', df.fault_severity.unique)
print ('unique for severity_type: ', df.severity_type.unique)
print ('unique for event_type: ', df.event_type.unique)
print ('unique for resource_type: ', df.resource_type.unique)
print ('unique for log_feature: ', df.log_feature.unique)
print ('unique for volume: ', df.volume.unique)


# ## **A general view.**
# 
# To see the number of cases according to **fault_severity**:

# In[35]:


#To display the data according to l2
df_fs = df.groupby('fault_severity', sort=False, as_index=False).size().sort_values(by="size",ascending=False)
df_fs


# In[36]:


plt.figure(figsize=(10,5))

data = [4784, 1871, 726]
keys = ['Fault Severity 0','Fault Severity 1', 'Fault Severity 2']
  
# declaring exploding pie
explode = [0.1, 0, 0]
# define Seaborn color palette to use
palette_color = sns.color_palette('YlGnBu')
  
# plotting data on chart
plt.pie(data ,labels=keys, colors=palette_color,
        explode=explode, autopct='%.0f%%')
  
# displaying chart
plt.show()


# In[37]:


plt.figure(figsize=(10,5))
sns.countplot(data = df, x='fault_severity', lw=1, edgecolor="black", palette=Spectral6, color = "#007ACC")
plt.title ('Count of Intensity of the Attack')
plt.xlabel('Fault Severity')
plt.ylabel('Count')
plt.show()


# In[38]:


plt.figure(figsize = (12,10))
sns.pairplot(df, palette=Spectral6)
plt.show()


# We can see that there are more non-failures than failures. 1 o 2.
# 
# To see the number of cases according to **severity_type**:

# In[39]:


df_st = df.groupby('severity_type', sort=False, as_index=False).size().sort_values(by="size",ascending=False)
df_st


# In[40]:


plt.figure(figsize=(10,5))
sns.countplot(data = df, x='severity_type', hue = 'fault_severity',lw=1, edgecolor="black",palette=Spectral6, color = "#007ACC")
plt.title ('Number of cases by Type of Severity (Severity Type)')
plt.xlabel('Severity Type')
plt.ylabel('Quantity')
plt.show()


# In[41]:


plt.figure(figsize=(10,5))
sns.countplot(data = df, x='severity_type',lw=1, edgecolor="black",palette=Spectral6, color = "#007ACC")
plt.title ('Number of cases by Type of Severity (Severity Type)')
plt.xlabel('Severity Type')
plt.ylabel('Count')
plt.show()


# In[ ]:





# With this information we can see that the faults that occur most are those of type 1 and 2. Those that severities type 4, 5 and 3 occur very little (compared to types 1 and 2).
# 
# To see the number of cases according to **resource_type**:

# In[42]:


df_rt = df.groupby('resource_type', sort=False, as_index=False).size().sort_values(by="size",ascending=False)
df_rt


# In[43]:


plt.figure(figsize=(15,5))
sns.barplot(data = df_rt, x='resource_type', y= 'size',lw=1, edgecolor="black", palette=Spectral6, color = "#007ACC")
plt.title ('Number of cases by Type of Resource(Resource Type)')
plt.xlabel('Resource Type')
plt.ylabel('Quantity')
plt.show()


# The most used types of resources are: 2 and 8. While the rest of the resources have relatively few cases.
# 
# To see the number of cases by location:

# In[44]:


df_loc = df.groupby('location', sort=False, as_index=False).size().sort_values(by="size",ascending=False)
df_loc


# In[45]:


plt.figure(figsize=(15,5))
sns.barplot(df_loc['location'][:5],df_loc['size'][:5], lw=1, edgecolor="black", palette=Spectral6, color = "#007ACC")
plt.title ('Count of Intensity of Attacks by Location(Top 5)')
plt.xlabel('Resource Type')
plt.ylabel('Quantity')
plt.show()


# In[46]:


plt.figure(figsize=(15,5))
sns.barplot(df_loc['location'],df_loc['size'], lw=1, edgecolor="black", palette=Spectral6, color = "#007ACC")
plt.title ('Intensity of Attacks in each and every location')
plt.xlabel('Resource Type')
plt.ylabel('Quantity')
plt.show()


# The 5 locations that have the most cases are: 821, 1107, 734, 126 and 1008.
# 
# ## fault_severity = 1
# 
# Let's analyze type 1 faults to see what we can find

# In[47]:


faults_type_1= df[df.fault_severity == 1]


# In[48]:


faults_type_1.head()


# In[49]:


faults_type_1.shape


# In[50]:


ft1_et = faults_type_1.groupby('event_type', sort=False, as_index=False).size().sort_values(by="size",ascending=False)
ft1_et


# In[51]:


plt.figure(figsize=(15,5))
sns.barplot(data = ft1_et, x='event_type', y='size', lw=1, edgecolor="black", palette=Spectral6, color = "#007ACC")
plt.title ('Number of cases by Event Type (Event Type)')
plt.xlabel('Event type')
plt.xticks(rotation=90)
plt.ylabel('Quantity')
plt.show()


# ## fault_severity = 2
# 
# Let's analyze type 1 faults to see what we can find

# In[52]:


faults_type_2= df[df.fault_severity == 2]


# In[53]:


faults_type_2.head()


# In[54]:


ft2_loc = faults_type_2.groupby('location', sort=False, as_index=False).size().sort_values(by="size",ascending=False)
ft2_loc


# In[55]:


plt.figure(figsize=(15,5))
sns.barplot(data = ft2_loc[1:15], x='location', y='size', lw=1, edgecolor="black", palette = Spectral6, color = "#007ACC")
plt.title ('15 Locations with the most cases of type 1 faults')
plt.xlabel('Locations')
plt.xticks(rotation=90)
plt.ylabel('Quantity')
plt.show()


# In[56]:


ft2_et = faults_type_2.groupby('event_type', sort=False, as_index=False).size().sort_values(by="size",ascending=False)
ft2_et


# In[57]:


plt.figure(figsize=(15,5))
sns.barplot(data = ft2_et, x='event_type', y='size', lw=1, edgecolor="black", palette=Spectral6, color = "#007ACC")
plt.title ('Number of cases by Event Type (Event Type)')
plt.xlabel('Event type')
plt.xticks(rotation=90)
plt.ylabel('Quantity')
plt.show()


# In[58]:


type(df)


# ## **Data Transformation**

# In[59]:


df.head()


# In[60]:


target_var = df['fault_severity']
independent_features = df.drop(columns = ['id','fault_severity'])


# In[61]:


df = pd.get_dummies(independent_features, columns = ['severity_type', 'resource_type'])
df.head()


# In[62]:


sc = StandardScaler()
col_to_scale = ['location', 'event_type', 'log_feature', 'volume']
df[col_to_scale] = sc.fit_transform(df[col_to_scale])


# In[63]:


df.head()


# # Resampling

# In[64]:


from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot


# In[65]:


X = df
y = target_var


# In[66]:


# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


# In[67]:


print(X.shape)


# In[68]:


print(y.shape)


# In[69]:


counter = Counter(y)
for k,v in counter.items():
	per = v / len(y) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.show()


# In[ ]:





# ## **Model Building**

# In[70]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)


# In[71]:


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# In[72]:


def resultados (y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, cbar= True,  square= True, annot=True, fmt= '.0f',
           cmap= 'YlGnBu', linewidths=.5);
    plt.title('Confusion matrix')
    plt.ylabel('Predicted Values')
    plt.xlabel('Actual Values')
    plt.show()


# # Support Vector Machine

# In[73]:


svc_clf = SVC()
svc_clf.fit(X_train, y_train)
y_pred_svc = svc_clf.predict(X_test)

print_score(svc_clf, X_train, y_train, X_test, y_test, train=True)
print_score(svc_clf, X_train, y_train, X_test, y_test, train=False)


# In[74]:


resultados(y_test, y_pred_svc)


# # Decision Tree

# In[75]:


dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
y_pred_tree = dt_clf.predict(X_test)

print_score(dt_clf, X_train, y_train, X_test, y_test, train=True)
print_score(dt_clf, X_train, y_train, X_test, y_test, train=False)


# In[76]:


resultados(y_test, y_pred_tree)


# # Random Forest Classifier

# In[77]:


rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)


# In[78]:


resultados(y_test, y_pred_rf)


# # KNearest Neighbour

# In[79]:


knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
y_pred_knn = knn_clf.predict(X_test)

print_score(knn_clf, X_train, y_train, X_test, y_test, train=True)
print_score(knn_clf, X_train, y_train, X_test, y_test, train=False)
resultados(y_test, y_pred_knn)


# # GaussianNB

# In[80]:


gnb_clf = GaussianNB()
gnb_clf.fit(X_train, y_train)
y_pred_gnb = gnb_clf.predict(X_test)

print_score(gnb_clf, X_train, y_train, X_test, y_test, train=True)
print_score(gnb_clf, X_train, y_train, X_test, y_test, train=False)
resultados(y_test, y_pred_gnb)


# In[ ]:





# In[ ]:





# # Improving the Best Model

# In[81]:


#Make the grid for Grid Search:
param_grid = {'n_estimators': [200, 300, 400, 500],   
              'min_samples_split': [2,3,4,5],    
              'min_samples_leaf':[1,3,5]}   


# In[82]:


model = GridSearchCV(rf_clf, param_grid=param_grid, cv=5)


# In[83]:


model.fit(X_train, y_train)


# In[84]:


print("Best parameters: "+str(model.best_params_))
print("Best Score: "+str(model.best_score_)+'\n')


# In[85]:


rf_clf_2 = RandomForestClassifier(min_samples_leaf=3, min_samples_split=3, n_estimators=300)


# In[86]:


rf_clf_2.fit(X_train, y_train)


# In[87]:


y_pred_forest2= rf_clf_2.predict(X_test)


# In[88]:


rf_clf_2 = RandomForestClassifier()
rf_clf_2.fit(X_train, y_train)
y_pred_rf2 = rf_clf_2.predict(X_test)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
resultados(y_test, y_pred_rf2)


# In[89]:


cols_model = X_train.columns
feature_importance= pd.DataFrame(list(zip(cols_model, rf_clf_2.feature_importances_.transpose())), columns = ['Col','Importance']).sort_values(by="Importance",ascending=False)
feature_importance


# In[ ]:




