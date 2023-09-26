#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


data = pd.read_csv(r"C:\Users\Shan Jacob\Downloads\ACME-HappinessSurvey2020.csv")


# In[3]:


data.head()


# In[4]:


from scipy import stats


mean = np.mean(data)
print(mean)


# In[8]:


median = np.median(data['Y'])
print(median)


# In[9]:


print("\nSummary Statistics:")
print(data.describe())


# In[10]:


variance = np.var(data)
print(variance)


# In[11]:


std_dev = np.std(data)
print(std_dev)


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.countplot(x='Y', data=data)
plt.title("Distribution of Target Variable (Y)")
plt.show()


# In[14]:


selected_columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
for col in selected_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()


# In[15]:


selected_columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
subset_data = data[selected_columns]

# Calculate the covariance matrix
covariance_matrix = np.cov(subset_data, rowvar=False)

# Create a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(covariance_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()

# Add title and labels
plt.title("Covariance Matrix")
plt.xticks(range(len(selected_columns)), selected_columns, rotation=45)
plt.yticks(range(len(selected_columns)), selected_columns)

# Show the plot


# In[40]:


data.head(20)


# In[71]:


X1_5 = data['X1'].head(10).tolist()
Y_5=data['Y'].head(10).tolist()


# In[80]:


X2_5 = data['X2'].head(10).tolist()
X3_5 = data['X3'].head(10).tolist()


# In[ ]:





# In[76]:


import matplotlib.pyplot as plt


# Create a line plot for "This year"

plt.scatter(X1_5,[1,2,3,4,5,6,7,8,9,10])
# Create a bar plot for "Last year"
plt.plot(Y_5)

# Set labels and title
plt.xlabel('X1')
plt.ylabel('Y')
plt.title('X vs Y')
plt.legend()

# Show the plot
plt.show()


# In[81]:


plt.scatter(X1_5,[1,2,3,4,5,6,7,8,9,10])
plt.scatter(X2_5,[1,2,3,4,5,6,7,8,9,10])
plt.scatter(X3_5,[1,2,3,4,5,6,7,8,9,10])
# Create a bar plot for "Last year"
plt.plot(Y_5)

# Set labels and title
plt.xlabel('X2')
plt.ylabel('Y')
plt.title('X vs Y')
plt.legend()

# Show the plot
plt.show()


# In[18]:


correlation_matrix = subset_data.corr()

plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()

# Add a title and labels
plt.title("correlation matrix")
plt.xticks(range(len(selected_columns)), selected_columns, rotation=45)
plt.yticks(range(len(selected_columns)), selected_columns)

# Show the plot
plt.show()


# In[19]:


missing_values = data.isnull().sum()

# Print out any columns with missing values
print(missing_values)


# In[21]:


X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']]
Y = data['Y']


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[49]:


# Initialize the XGBoost classifier
model = xgb.XGBClassifier()

# Train the model
model.fit(X_train, y_train)


# In[51]:


y_pred =model.predict(X_test)


# In[52]:


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# In[53]:




from sklearn.metrics import confusion_matrix


# Assuming y_pred and y_test are defined
conf_mat = confusion_matrix(y_test, y_pred)

# Create a figure and a set of subplots
plt.figure(figsize=(6, 4))
sns.set(font_scale=1.2)  # Adjust the font scale for better readability
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Unhappy', 'Happy'],
            yticklabels=['Unhappy', 'Happy'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[56]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

y_probabilities = model.predict_proba(X_test)[:, 1]

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probabilities)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()


# In[58]:


from sklearn.linear_model import LogisticRegression
modelL = LogisticRegression()

# Train the model
modelL.fit(X_train, y_train)

# Predict on the test set
y_predL = modelL.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_predL)
print(f'Accuracy: {accuracy}')


# In[83]:


print(y_pred)


# In[86]:


yp_10=y_pred[:10]
yt_10=y_test[:10]
print(yt_10,yp_10)


# In[90]:


plt.scatter(yp_10,[1,2,3,4,5,6,7,8,9,10])
# Create a bar plot for "Last year"
plt.plot(yt_10,[1,2,3,4,5,6,7,8,9,10])

# Set labels and title
plt.xlabel('X2')
plt.ylabel('Y')
plt.title('X vs Y')
plt.legend()

# Show the plot
plt.show()


# In[ ]:




