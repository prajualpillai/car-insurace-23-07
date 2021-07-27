# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import LabelEncoder


# %%
df = pd.read_csv("train.csv")
df.head(2)

# %% [markdown]
# # Checking the type of data in the dataset

# %%
df.info()

# %% [markdown]
# # Checking for null values

# %%
df.isnull().sum()

# %% [markdown]
# As we see that there are no null values in our dataset, we can now carry on without imputations

# %%
df.drop(['id'],axis = 1,inplace=True)


# %%
df.describe(include='all')

# %% [markdown]
# # Considering one feature at a time
# %% [markdown]
# ## Gender

# %%
df['Gender'].unique()

# %% [markdown]
# Has only two values 'Male' and 'Female'.
# Next lets, check the count of each and the ratio of male and female in the data.

# %%
for i in df.columns:
    if df[i].dtype == object:
        print(i)
        plt.hist(df[i])
        plt.show()

# %% [markdown]
# Apart from vehicle age all the other categorical datas are almost evenly distributed

# %%
print(len(df[df['Vehicle_Age']=='> 2 Years']))

# %% [markdown]
# # Checking for correlation of data

# %%
y = df['Response']
x = df.drop(['Response'],axis=1)


# %%
plt.rcParams['agg.path.chunksize'] = 100000
plt.plot(df['Response'],df['Region_Code'],'ro')
plt.show()

# %% [markdown]
# ## Linear Correlatiom

# %%
df.corr(method='spearman')

# %% [markdown]
# ## Scatter plot

# %%
fig,axes = plt.subplot(nrows = len(df), ncols = 1)
for i in df.columns:
    axes[j][0].scatter()

# %% [markdown]
# # Label Encoding

# %%

df1 = df.copy()
lab = LabelEncoder()
for i in df.columns:
    if df[i].dtype == object:
        df1[i] = lab.fit_transform(df[i]).astype('float64')
df1

# %% [markdown]
# # One Hot Encoding

# %%
df1 = df.copy()
for i in df1.columns:
    if df1[i].dtype == object:
        d = pd.get_dummies(df[i],prefix = i)
        df1.drop(i,axis=1,inplace=True)
        df1 = df1.join(d)
df1


# %%



