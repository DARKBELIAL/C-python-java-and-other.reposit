from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=fetch_california_housing()
dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names
dataset['price']=df.target
dataset.head()
X=dataset.iloc[:,:-1]##separating the independent features
y=dataset.iloc[:,-1]##separating the dependent features
