#%%
import pandas as pd
import numpy as np
# %%
df = pd.read_csv("train.csv")
df.describe()
# %%
id = df.id
df = df.drop("id",axis=1)
df.describe()
# %%
for i in df.columns:
    print(sum(df[i].isnull()))
# %%
import matplotlib.pyplot as plt
for i in df.columns:
    print(i)
    plt.plot(id,df[i])
    plt.show()
# %%
from sklearn.preprocessing import OneHotEncoder as ohe
enc = ohe(sparse = "false")
for i in df.columns:
    if df[i].dtype == "object":
        X = df[i]
# %%
for i in df.columns:
    if df[i].dtype == "object":
        X = df[i]
        enc.fit(X)
        print(enc.categories_)
# %%
x = df["Gender"]
from keras.utils import to_categorical as tc
x = np.array(x)
en = tc(x)
print(en)
# %%

# %%

# %%
