#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def map_score(dataframe,customers,prob):
    dataframe['Propensity'] = 0
    for i in range(len(dataframe)):
        idx = dataframe.index[i]
        for j in range(len(customers)):
            if dataframe.loc[idx,'Client'] == customers[j]:
                dataframe.loc[idx, "Propensity"] = prob[j]


# In[66]:


df_kbc_bank = pd.read_csv("F:\Spark_Files\bank.csv")
df_kbc_bank.head() 
#df_kbc_bank.shape
df_kbc_bank.to_csv('bankdata.csv',index=False)


# In[4]:


df_kbc_bank['Client'].nunique()


# In[5]:


cust_req = pd.DataFrame({'Category': {0: 'Credit', 1: 'Loan',2:'Mutual Fund'}, 'Sum': {0: float(df_kbc_bank[["Sale_CC"]].sum()),
1:float(df_kbc_bank[["Sale_CL"]].sum()),2:float(df_kbc_bank[["Sale_MF"]].sum()) }})


# In[6]:


cust_req.head()


# In[7]:


sns.barplot(x = 'Sum', y = 'Category', data = cust_req)
plt.title('Requirement of Customers')
plt.xlabel('Count')
plt.ylabel('Products for Marketing')


# In[8]:


df_kbc_bank = df_kbc_bank.dropna()
corr = df_kbc_bank.corr()
corr.sort_values(["Sale_CL"], ascending = False, inplace = True)
print(corr.Sale_CL)


# In[9]:


data_loan = df_kbc_bank[['Tenure','TransactionsCred_CA','Count_CA','TransactionsCred','TransactionsDebCash_Card','ActBal_CC','Age','Client','Sale_CL','Revenue_CL']]
#len(data_loan)
data_loan.head()
#data.shape()


# In[12]:


X = data_loan.loc[:,'Tenure':'Client'].values # as_matrix() is deprecated hence replaced with .values 
y = data_loan.Sale_CL.values
#X.shape
#len(y)


# In[15]:


Corr_Loan = data_loan.corr()
plt.figure(figsize=(10,10))
sns.heatmap(Corr_Loan, vmax=1, square=True,annot=True,cmap='bwr')
plt.title('Correlation between the selected Features')


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=42)


# In[17]:


logistic = linear_model.LogisticRegression(penalty='l2',C=1,class_weight='balanced')
logistic = logistic.fit(X_train, y_train)
print("Accuracy = "+str(logistic.score(X_test,y_test)))


# In[18]:


pscore = logistic.predict_proba(X_test)[:,1]  # The predicted propensities by the model
pscore


# In[19]:


Clients = X_test[:,-1]


# In[20]:


data_loan.head()


# In[21]:


map_score(data_loan,Clients,pscore)


# In[22]:


# Remove records where the Target sale value is 0
data_loan =data_loan[(data_loan.Sale_CL != 0)]
# Remove cases where probability is 0
data_loan = data_loan[(data_loan.Propensity != 0)]
# Remove the non essential fields 
data_loan = data_loan[['Client','Sale_CL','Revenue_CL','Propensity']]
# Sort the values by Decreasing order of Propensity, so that the clients with the highest propensity can be targeted first
Clients_loan = data_loan.sort_values(by ='Propensity',ascending=False)
# Export the results to a CSV file
Clients_loan.to_csv('Clients_loan.csv',index=False)


# In[23]:


data_loan.head()


# In[24]:


corr.sort_values(["Sale_CC"], ascending = False, inplace = True)
print(corr.Sale_CC)


# In[25]:


data_credit = df_kbc_bank[['ActBal_SA','ActBal_CA','Count_SA','VolumeDeb_PaymentOrder','TransactionsDebCash_Card','ActBal_CC','Count_CC','Client','Sale_CC','Revenue_CC']]
data_credit.head()


# In[26]:


X = data_credit.loc[:,'ActBal_SA':'Client'].values
y = data_credit.Sale_CC.values


# In[27]:


Corr_Credit = data_credit.corr()
plt.figure(figsize=(10,10))
sns.heatmap(Corr_Credit, vmax=1, square=True,annot=True,cmap='Dark2')
plt.title('Correlation between the selected Features')


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=42)
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))


# In[29]:


pscore = clf.predict_proba(X_test)[:,1]


# In[30]:


map_score(data_credit,Clients,pscore)


# In[31]:



# Remove records where the Target sale value is 0
data_credit =data_credit[(data_credit.Sale_CC != 0)]
# Remove cases where probability is 0
data_credit = data_credit[(data_credit.Propensity != 0)]
# Remove the non essential fields 
data_credit = data_credit[['Client','Sale_CC','Revenue_CC','Propensity']]
# Sort the values by Decreasing order of Propensity, so that the clients with the highest propensity can be targeted first
Clients_credit = data_credit.sort_values(by ='Propensity',ascending=False)
# Export the results to a CSV file
Clients_credit.to_csv('Clients_credit.csv',index=False)


# In[32]:


Clients_credit.head()


# In[33]:


corr.sort_values(["Sale_MF"], ascending = False, inplace = True)
print(corr.Sale_MF)


# In[34]:


data_mutual_fund = df_kbc_bank[['Count_MF','TransactionsDebCashless_Card','TransactionsDeb','TransactionsCred_CA'             
,'TransactionsCred','TransactionsDeb_CA','ActBal_MF',
'TransactionsDeb_PaymentOrder','VolumeCred_CA','Client','Sale_MF','Revenue_MF']]
data_mutual_fund.head()
#len(data_mutual_fund)


# In[35]:


X = data_mutual_fund.loc[:,'Count_MF':'Client'].values
y = data_mutual_fund.Sale_MF.values


# In[36]:


Corr_Mutual_Fund = data_mutual_fund.corr()
plt.figure(figsize=(10,10))
sns.heatmap(Corr_Mutual_Fund, vmax=1, square=True,annot=True,cmap='Accent')
plt.title('Correlation between the selected Features')


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=42)
clf = GaussianNB()
clf.fit(X_train,y_train)


# In[38]:


print(clf.score(X_test,y_test))


# In[39]:


pscore = clf.predict_proba(X_test)[:,1]
#pscore


# In[40]:


Clients = X_test[:,-1]
map_score(data_mutual_fund,Clients,pscore)


# In[41]:


data_mutual_fund.head()


# In[42]:


# Remove records where the Target sale value is 0
data_mutual_fund =data_mutual_fund[(data_mutual_fund.Sale_MF != 0)]
# Remove cases where probability is 0
data_mutual_fund = data_mutual_fund[(data_mutual_fund.Propensity != 0)]
# Remove the non essential fields 
data_mutual_fund = data_mutual_fund[['Client','Sale_MF','Revenue_MF','Propensity']]
# Sort the values by Decreasing order of Propensity, so that the clients with the highest propensity can be targeted first
Clients_mutual_fund = data_mutual_fund.sort_values(by ='Propensity',ascending=False)
# Export the results to a CSV file
Clients_mutual_fund.to_csv('Clients_mutual_fund.csv',index=False)
Clients_mutual_fund.head()


# In[43]:


data_credit.rename(columns={'Revenue_CC': 'Revenue'}, inplace=True)
data_loan.rename(columns={'Revenue_CL': 'Revenue'}, inplace=True)
data_mutual_fund.rename(columns={'Revenue_MF': 'Revenue'}, inplace=True)
print("Revenue From sale of Credit Card   = " +str(data_credit.Revenue.sum()))
print("Revenue From sale of Consumer Loan = " +str(data_loan.Revenue.sum()))
print("Revenue From sale of Mutual Fund   = " +str(data_mutual_fund.Revenue.sum()))


# In[44]:


print("Total best case Revenue = " +str(data_credit.Revenue.sum() + data_loan.Revenue.sum() + data_mutual_fund.Revenue.sum()))


# In[45]:


Consolidated_data = data_credit[["Client","Revenue",'Propensity']].copy()
Consolidated_data['Category'] = 'Credit'
Consolidated_data = Consolidated_data.append(data_loan, sort = True)
Consolidated_data = Consolidated_data[['Category','Client','Revenue','Propensity']]
Consolidated_data = Consolidated_data.fillna('Loan')
Consolidated_data = Consolidated_data.append(data_mutual_fund)
Consolidated_data = Consolidated_data[['Category','Client','Revenue','Propensity']]
Consolidated_data = Consolidated_data.fillna('mutual_fund')
Consolidated_data['Propensity'] = pd.to_numeric(Consolidated_data['Propensity'])


# In[51]:


Consolidated_data['Category'].value_counts()


# In[52]:


sns.FacetGrid(Consolidated_data, hue="Category", height=5)    .map(plt.scatter, "Propensity", "Revenue")    .add_legend()


# In[53]:


sns.violinplot(x="Category", y="Revenue", data=Consolidated_data, size=6)


# In[54]:


sns.pairplot(Consolidated_data.drop("Client", axis=1), hue="Category", height=3)


# In[55]:


sns.pairplot(Consolidated_data.drop("Client", axis=1), hue="Category", height=3, diag_kind="kde")


# In[62]:


CallTheseClients = Consolidated_data[(Consolidated_data.Propensity >= 0.60)]
CallTheseClients = CallTheseClients.sort_values(by =['Category','Propensity'],ascending=False)
CallTheseClients


# In[63]:


Expected_Revenue = CallTheseClients.groupby(by=['Category'],as_index=False)['Revenue'].sum()
print("Expected Revenue From sale of Credit Card   = " +str(Expected_Revenue['Revenue'].iloc[0]))
print("Expected Revenue From sale of Consumer Loan = " +str(Expected_Revenue['Revenue'].iloc[1]))
print("Expected Revenue From sale of Mutual Fund   = " +str(Expected_Revenue['Revenue'].iloc[2]))


# In[64]:


print("Combined Expected Revenue = " +str(CallTheseClients['Revenue'].sum()))


# In[65]:


CallTheseClients.to_csv('CallTheseClients.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:




