import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os,json
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, roc_curve, auc
from PIL import Image




X_train_final=pd.read_csv('Train_data.csv')
X_test_final=pd.read_csv('Test_data.csv')
y_train=pd.read_csv('Train_label.csv')
y_test=pd.read_csv('Test_label.csv')

y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]

vals_count = 1 - (np.bincount(y_train) / len(y_train))
vals_count = vals_count / np.sum(vals_count)  ## normalizing

dict_weights = {}
for i in range(2):  ## 2 classes (0, 1)
    dict_weights[i] = vals_count[i]

over = SMOTE(sampling_strategy=0.7)
X_train_resmapled, y_train_resampled = over.fit_resample(X_train_final, y_train)

def train_model(X_train, y_train, plot_name='', class_weight=None):
    
    global clf_name

    clf = RandomForestClassifier(n_estimators=344, max_depth=12, random_state=45, class_weight=class_weight)
    # clf = LogisticRegression(C=2.5, max_iter=1000, random_state=45, class_weight=class_weight)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test_final)
    

    f1_test = f1_score(y_test, y_pred_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    clf_name = clf.__class__.__name__

    ## Plot the confusion matrix 
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cbar=False, fmt='.2f', cmap='Blues')
    plt.title(f'{plot_name}')
    plt.xticks(ticks=np.arange(2) + 0.5, labels=[False, True])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=[False, True])

    ## Save the plot locally
    plt.savefig(f'{plot_name}.png', bbox_inches='tight', dpi=300)
    plt.close()  


    ## Results
    new_results = {f'f1-score-{plot_name}': f1_test, f'accuracy-{plot_name}': acc_test}

    ## Dump to json file
    with open('metrics.json', 'a') as f:
        json.dump(new_results, f)  # Append new_results to the JSON file

    return True


## 1. without considering the imabalancing data
train_model(X_train=X_train_final, y_train=y_train, plot_name='without-imbalance')

## 2. with considering the imabalancing data using class_weights
train_model(X_train=X_train_final, y_train=y_train, plot_name='with-class-weights', class_weight=dict_weights)

## 3. with considering the imabalancing data using oversampled data (SMOTE)
train_model(X_train=X_train_resmapled, y_train=y_train_resampled, plot_name=f'with-SMOTE')

## Combine all conf matrix in one
confusion_matrix_paths = [f'./without-imbalance.png', f'./with-class-weights.png', f'./with-SMOTE.png']

## Load and plot each confusion matrix
plt.figure(figsize=(15, 5))  # Adjust figure size as needed
for i, path in enumerate(confusion_matrix_paths, 1):
    im = Image.open(path)
    plt.subplot(1, len(confusion_matrix_paths), i)
    plt.imshow(im)
    plt.axis('off')  # Disable axis for cleaner visualization


## Save combined plot locally
plt.suptitle(clf_name, fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'conf_matrix.png', bbox_inches='tight', dpi=300)

for path in confusion_matrix_paths:
    os.remove(path)

with open('metrics.json', 'r') as file:
    data = file.read()

json_objects = data.split('}')
json_objects = [obj + '}' for obj in json_objects if obj]

combined_data = {}
for obj in json_objects:
    obj_data = json.loads(obj)
    combined_data.update(obj_data)


## Dump the combined data back to the same file
with open('metrics.json', 'w') as f:
    json.dump(combined_data, f)