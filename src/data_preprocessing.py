import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle
import os 

df=pd.read_csv('data/house_data.csv')

os.makedirs('models', exist_ok=True)

df.fillna(0, inplace=True)

le=LabelEncoder()
df['location']=le.fit_transform(df['location'])

X=df.drop('price', axis=1)
y=df['price']