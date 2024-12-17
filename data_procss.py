import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector


df=pd.read_csv('data_prep.csv')

X = df.drop(columns=['Exited'], axis=1)
y = df['Exited']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=45)


num_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
categ_cols = ['Gender', 'Geography']
ready_cols = list(set(X_train.columns.tolist()) - set(num_cols) - set(categ_cols))

num_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(num_cols)),
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ])


categ_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(categ_cols)),
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('ohe', OneHotEncoder(drop='first', sparse_output=False))
                    ])

ready_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(ready_cols)),
                        ('imputer', SimpleImputer(strategy='most_frequent'))])

all_pipeline = FeatureUnion(transformer_list=[
                                    ('numerical', num_pipeline),
                                    ('categorical', categ_pipeline),
                                    ('ready', ready_pipeline)])  


all_pipeline.fit(X_train)


out_categ_cols = all_pipeline.named_transformers['categorical'].named_steps['ohe'].get_feature_names_out(categ_cols)                                    


X_train_final = pd.DataFrame(all_pipeline.transform(X_train), columns=num_cols + list(out_categ_cols) + ready_cols)
X_test_final = pd.DataFrame(all_pipeline.transform(X_test), columns=num_cols + list(out_categ_cols) + ready_cols)


X_train_final.to_csv('Train_data.csv',index=False)
X_test_final.to_csv('Test_data.csv',index=False)
y_train.to_csv('Train_label.csv',index=False)
y_test.to_csv('Test_label.csv',index=False)

