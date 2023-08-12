# Random Forest Classifier
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('iris.csv')

species_dic = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

df = df.replace(['Setosa', 'Versicolor' , 'Virginica'],[0, 1, 2])

X = df.iloc[:, 0:-1] 
y = df.iloc[:, -1] 

rfc = RandomForestClassifier() 
rfc.fit(X, y) 

# Function for classification based on inputs
def classify(a, b, c, d):
    arr = np.array([a, b, c, d]) 
    arr = arr.astype(np.float64) 
    query = arr.reshape(1, -1) 
    prediction =species_dic[rfc.predict(query)[0]]
    proba=rfc.predict_proba(query)[0] 
    return prediction, proba 