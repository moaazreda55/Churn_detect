import pandas as pd
import os


df = pd.read_csv("Customer-Churn-Records.csv")

df.drop(columns=['RowNumber', 'CustomerId', 'Surname','Card Type','Complain'], axis=1, inplace=True)

df.drop(index=df[df['Age'] > 80].index.tolist(), axis=0, inplace=True)

df.to_csv('data_prep.csv',index=False)