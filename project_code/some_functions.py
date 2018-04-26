#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: doris xiang
"""
import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import explained_variance_score, mean_absolute_error,mean_squared_error,r2_score
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import validation_curve
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

def load_csv(file_path):
    data = pd.read_csv(file_path)
    data = data.replace('$Failed',np.nan)
    data2 = data.replace('NaN',np.nan).replace('na',np.nan)
    data2 = data2.dropna()
    data2 = data2.astype(float)
    #create new features by aggregating by seasons
    data2['spring'] = data2['Mar']+data2['Apr']+data2['May']
    data2['summer'] = data2['Jun']+data2['Jul']+data2['Aug']
    data2['autumn'] = data2['Sept']+data2['Oct']+data2['Nov']
    data2['winter'] = data2['Dec']+data2['Jan']+data2['Feb']
    
    data3 = data2.iloc[:,4:]
    label = data2.iloc[:,3]   
    return data3,label
    
def standarlize(data,label):
    scaler = MinMaxScaler()
    data3 = scaler.fit_transform(data)
    if len(label)==1:
        y =0
    else:
        y = label/label.max()
    return data3, y
    
def min_max(series):
    mi = np.min(series)
    series2 = series-mi
    return series2


def load_prediction(file_path):
    data = pd.read_csv(file_path)
    data = data.replace('$Failed',np.nan)
    data2 = data.replace('NaN',np.nan).replace('na',np.nan)
    '''
    impute = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    X = data2.iloc[:,3:]
    
    X1 = pd.DataFrame(impute.fit_transform(X))
    X1 = X1.astype(float)
    #create new features by aggregating by seasons
    X1.columns = X.columns
    X1['spring'] = X1['Mar']+X1['Apr']+X1['May']
    X1['summer'] = X1['Jun']+X1['Jul']+X1['Aug']
    X1['autumn'] = X1['Sept']+X1['Oct']+X1['Nov']
    X1['winter'] = X1['Dec']+X1['Jan']+X1['Feb']
    
    zipcodes = data.iloc[:,0]
    '''
    data2 = data2.dropna()
    zipcodes = data2['zipcode']
    data2 = data2.iloc[:,3:]
    data2 = data2.astype(float)
    #create new features by aggregating by seasons
    data2['spring'] = data2['Mar']+data2['Apr']+data2['May']
    data2['summer'] = data2['Jun']+data2['Jul']+data2['Aug']
    data2['autumn'] = data2['Sept']+data2['Oct']+data2['Nov']
    data2['winter'] = data2['Dec']+data2['Jan']+data2['Feb']
    
    
    return zipcodes,data2
    
''' 
def get_data_from_svmlight(svmlight_file):
    data_train = load_svmlight_file(svmlight_file,n_features=29)
    X_train = data_train[0]
    Y_train = data_train[1]
    return X_train, Y_train
    
    
    
def plot_validation_curve(X,y):
    param_range = np.logspace(-6, -1, 5)
    train_scores, test_scores = validation_curve(
    linear_model.LogisticRegression(), X, y, param_name="C", param_range=param_range,
    cv=10, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.title("Validation Curve with SVM")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
''' 