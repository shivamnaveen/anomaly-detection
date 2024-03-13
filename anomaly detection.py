#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the libaries to work with for EDA
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


data = pd.read_csv('Online Payment Fraud Detection.csv',encoding='unicode-escape')


# In[3]:


data.shape


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.describe()


# In[7]:


data.columns


# In[8]:


data.info()


# In[9]:


data.dtypes


# In[10]:


data.isna().sum()


# In[11]:


data.nameDest.unique()


# In[12]:


data.nameOrig.unique()


# In[13]:


data.nameOrig.value_counts()


# In[14]:


data.nameDest.value_counts()


# In[15]:


data.amount.max()


# In[16]:


labels = data['type'].astype('category').cat.categories.tolist()
counts = data['type'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()


# In[17]:


data.type.value_counts()


# In[18]:


top_ten = data.groupby('nameOrig').type.sum().sort_values(ascending=False)[:10]
top_ten


# In[19]:


data['amount'].mean()


# In[20]:


sns.boxplot(y=data.step)
plt.title('Time of Transaction Profile')
plt.ylim(0,100)
plt.show()


# In[21]:


sns.boxplot(y=data.amount)
plt.title('Amounts Transacted Profile')
plt.ylim(0,1000000)
plt.show()


# In[22]:


sns.boxplot(y=data.isFraud)
plt.title('Fraud Profile')
plt.ylim(-1,1)
plt.show()


# In[23]:


Online_Payment_layout = sns.PairGrid(data, vars = ['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'], hue = 'isFraud')

Online_Payment_layout.map_diag(plt.hist, alpha = 0.6)
Online_Payment_layout.map_offdiag(plt.scatter, alpha = 0.5)
Online_Payment_layout.add_legend()


# In[24]:


sns.barplot(x='amount', y='type', hue= 'isFraud', data=data)
plt.show()


# In[25]:


sns.catplot(data=data,kind='box')

plt.ylim(0,2000000)


# In[26]:


labels = data['isFraud'].astype('category').cat.categories.tolist()
counts = data['isFraud'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()


# In[27]:


Fraudulent_Transaction = data[data.isFraud ==1]
Not_Fraudulent_Transaction = data[data.isFraud ==0]


# In[28]:


print('Fraudulent Transaction: {}'.format(len(Fraudulent_Transaction)))
print('Not Fraudulent Transaction: {}'.format(len(Not_Fraudulent_Transaction)))
   


# In[29]:


Not_Fraudulent_Transaction.amount.describe()


# In[30]:


Fraudulent_Transaction.amount.describe()


# In[31]:


data.groupby('isFraud').mean()


# In[32]:


Non_Fraudulent_Sample = Not_Fraudulent_Transaction.sample(n=1142)


# In[33]:


new_dataset = pd.concat([Non_Fraudulent_Sample, Fraudulent_Transaction], axis=0)


# In[34]:


new_dataset.head()


# In[35]:


new_dataset.tail()


# In[36]:


new_dataset['isFraud'].value_counts()


# In[37]:


new_dataset.shape


# In[38]:


new_dataset.groupby('isFraud').mean()


# In[39]:


from sklearn.preprocessing import OneHotEncoder


# In[40]:


encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, drop=None,)


# In[41]:


encoder_df =  pd.get_dummies(new_dataset, columns=['type','nameOrig','nameDest'], prefix=['type','nameOrig','nameDest'])


# In[42]:


encoder_df


# In[43]:


encoder_df.shape


# In[44]:


encoder_df.head()


# In[45]:


encoder_df.tail()


# In[46]:


Y = encoder_df['isFraud']


# In[47]:


features = encoder_df.drop('isFraud', axis=1)


# In[48]:


X = features


# In[49]:


Y.head()


# In[50]:


X.head()


# In[51]:


from sklearn.model_selection import train_test_split


# In[52]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=2)


# In[53]:


print('\n',X_train.head(2))

print('\n',X_test.head(2))

print('\n',Y_train.head(2))

print('\n',Y_test.head(2))


# In[54]:


from sklearn.linear_model import LogisticRegression


# In[55]:


model = LogisticRegression()


# In[56]:


model.fit(X_train, Y_train)


# In[57]:


model_pred = model.predict(X_test)


# In[58]:


probs = model.predict_proba(X_test)


# In[59]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, precision_recall_curve, average_precision_score, roc_auc_score


# In[60]:


print('\nClassification Report:')
print(classification_report(Y_test, model_pred))


# In[61]:


pd.DataFrame(confusion_matrix(Y_test, model_pred), 
             columns=['Predicted Negative(0) ', 'Predicted Positive(1)'], 
             index=['Actually Negative(0)', 'Actually Positive(1)'])


# In[62]:


pd.DataFrame(confusion_matrix(Y_test, model_pred), 
             columns=['Predicted Not Fraud(0) ', 'Predicted Fraud(1)'], 
             index=['Actually Not Fraud(0)', 'Actually Fraud(1)'])


# In[63]:


print('Accuracy:',accuracy_score(Y_test, model_pred))


# In[64]:


average_precision = average_precision_score(Y_test, model_pred)
average_precision


# In[65]:


y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)


#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# In[66]:


print('AUC Score:')
print(roc_auc_score(Y_test, probs[:,1]))


# In[67]:


from sklearn.ensemble import RandomForestClassifier


# In[68]:


model = RandomForestClassifier(random_state=5, n_estimators=20)


# In[69]:


model.fit(X_train,Y_train)


# In[70]:


model_pred = model.predict(X_test)


# In[71]:


probs = model.predict_proba(X_test)


# In[72]:


from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, precision_recall_curve, average_precision_score, roc_auc_score


# In[73]:


print('Classification_Report:\n',classification_report(Y_test, model_pred))


# In[74]:


pd.DataFrame(confusion_matrix(Y_test, model_pred), 
             columns=['Predicted Negative(0) ', 'Predicted Positive(1)'], 
             index=['Actually Negative(0)', 'Actually Positive(1)'])


# In[75]:


pd.DataFrame(confusion_matrix(Y_test, model_pred), 
             columns=['Predicted Not Fraud(0) ', 'Predicted Fraud(1)'], 
             index=['Actually Not Fraud(0)', 'Actually Fraud(1)'])


# In[76]:


print('Accuracy:',accuracy_score(Y_test, model_pred))


# In[77]:


average_precision = average_precision_score(Y_test, model_pred)
average_precision


# In[78]:


y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)


#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()


# In[79]:


print('AUC Score:')
print(roc_auc_score(Y_test, probs[:, 1]))


# In[ ]:




