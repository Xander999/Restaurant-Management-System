#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:48:43 2019

@author: xander999
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def preh(xns):
    print(xns)
    dataset = pd.read_csv('zomato.csv',encoding="ISO-8859-1")
    y = dataset.iloc[:,-2].values

    dataset=dataset.drop(columns='Address')
    dataset=dataset.drop(columns=['Locality','Locality Verbose'])
    dataset=dataset.drop(columns=['Rating color','Rating text','Longitude','Latitude'])
    dataset=dataset.drop(columns=['Currency','Restaurant Name','Cuisines','City','Is delivering now','Switch to order menu'])
    X = dataset.iloc[:, 2:8].values

    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    X[:,1] = labelencoder_X.fit_transform(X[:,1])
    X[:,2] = labelencoder_X.fit_transform(X[:,2])
    
    X=X.astype('float64')

    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(X[:, 0:8])
    X[:, 0:8] = imputer.transform(X[:, 0:8])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    xx=sc_X.transform(xns)
    # Fitting K-NN to the Training set
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    '''
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)'''
    
    ans=classifier.predict(xx)
    return ans

if __name__=="__main__":
    a=[[6000.0,1.0,0.0,4.0,4.9,621.0]]
    print(preh(a))