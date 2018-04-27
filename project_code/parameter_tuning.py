#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: dorisx
"""
import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor



def ridge_plot(X_train,y_train,X_holdout,y_holdout):
    lamda = np.logspace(-4,4,num=60)
    MSE_train = []
    MSE_holdout = []
    for val in lamda:
        model = linear_model.Ridge(alpha = val)
        model.fit(X_train, y_train)
        y_pre_train = model.predict(X_train)
        y_pre_holdout = model.predict(X_holdout)
        MSE_train.append(mean_squared_error(y_train,y_pre_train))
        MSE_holdout.append(mean_squared_error(y_holdout,y_pre_holdout))
              
    lam = lamda[np.argmin(MSE_holdout)]
    MSE = np.min(MSE_holdout)
    
    train_scores_mean = np.mean(MSE_train)
    train_scores_std = np.std(MSE_train)
    test_scores_mean = np.mean(MSE_holdout)
    test_scores_std = np.std(MSE_holdout)
        
    plt.title("mean squared error with ridge regression")
    plt.xlabel("$\\alpha$")
    plt.ylabel("error")
    plt.ylim(0.0, 0.01)
    lw = 2
    plt.semilogx(lamda, MSE_train, label="Training error",
             color="darkorange", lw=lw)
    plt.fill_between(lamda, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
    plt.semilogx(lamda, MSE_holdout, label="Holdout error",
             color="navy", lw=lw)
    plt.fill_between(lamda, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1,
                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
 
    return lam,MSE

def lasso_plot(X_train,y_train,X_holdout, y_holdout):
    alpha = np.logspace(-7,1,num=60)
    MSE_train = []
    MSE_holdout = []
    for val in alpha:
        model = linear_model.Lasso(alpha = val)
        model.fit(X_train, y_train)
        y_pre_train = model.predict(X_train)
        y_pre_holdout = model.predict(X_holdout)
        MSE_train.append(mean_squared_error(y_train,y_pre_train))
        MSE_holdout.append(mean_squared_error(y_holdout,y_pre_holdout))
              
    al = alpha[np.argmin(MSE_holdout)]
    MSE = np.min(MSE_holdout)
    
    train_scores_mean = np.mean(MSE_train)
    train_scores_std = np.std(MSE_train)
    test_scores_mean = np.mean(MSE_holdout)
    test_scores_std = np.std(MSE_holdout)
        
    plt.title("mean squared error with lasso regression")
    plt.xlabel("$\\alpha$")
    plt.ylabel("error")
    plt.ylim(0.0, 0.01)
    lw = 2
    plt.semilogx(alpha, MSE_train, label="Training error",
             color="darkorange", lw=lw)
    plt.fill_between(alpha, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
    plt.semilogx(alpha, MSE_holdout, label="Holdout error",
             color="navy", lw=lw)
    plt.fill_between(alpha, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1,
                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
 
    return al,MSE

def ada_boost_plot(X_train, y_train, X_holdout, y_holdout):
    alpha = np.logspace(0,3,num=10)
    MSE_train = []
    MSE_holdout = []
    alpha = [int(x) for x in alpha]
    for val in alpha:
        model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=val, random_state=137)
        model.fit(X_train, y_train)
        y_pre_train = model.predict(X_train)
        y_pre_holdout = model.predict(X_holdout)
        MSE_train.append(mean_squared_error(y_train,y_pre_train))
        MSE_holdout.append(mean_squared_error(y_holdout,y_pre_holdout))
              
    al = alpha[np.argmin(MSE_holdout)]
    MSE = np.min(MSE_holdout)
    
    train_scores_mean = np.mean(MSE_train)
    train_scores_std = np.std(MSE_train)
    test_scores_mean = np.mean(MSE_holdout)
    test_scores_std = np.std(MSE_holdout)
        
    plt.title("mean squared error of Decision Tree regression with Adaboost")
    plt.xlabel("n_estimators")
    plt.ylabel("error")
    plt.ylim(0.0, 0.02)
    lw = 2
    plt.semilogx(alpha, MSE_train, label="Training error",
             color="darkorange", lw=lw)
    plt.fill_between(alpha, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
    plt.semilogx(alpha, MSE_holdout, label="Holdout error",
             color="navy", lw=lw)
    plt.fill_between(alpha, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1,
                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
 
    return al,MSE
    

def gradient_boost_plot(X_train, y_train, X_holdout, y_holdout):
    alpha = np.logspace(0,3,num=10)
    MSE_train = []
    MSE_holdout = []
    alpha = [int(x) for x in alpha]
    for val in alpha:
        model = GradientBoostingRegressor(max_depth=4,
                          n_estimators=val)
        model.fit(X_train, y_train)
        y_pre_train = model.predict(X_train)
        y_pre_holdout = model.predict(X_holdout)
        MSE_train.append(mean_squared_error(y_train,y_pre_train))
        MSE_holdout.append(mean_squared_error(y_holdout,y_pre_holdout))
              
    al = alpha[np.argmin(MSE_holdout)]
    MSE = np.min(MSE_holdout)
    
    train_scores_mean = np.mean(MSE_train)
    train_scores_std = np.std(MSE_train)
    test_scores_mean = np.mean(MSE_holdout)
    test_scores_std = np.std(MSE_holdout)
        
    plt.title("mean squared error of gradient boost")
    plt.xlabel("n_estimators")
    plt.ylabel("error")
    plt.ylim(0.0, 0.02)
    lw = 2
    plt.semilogx(alpha, MSE_train, label="Training error",
             color="darkorange", lw=lw)
    plt.fill_between(alpha, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
    plt.semilogx(alpha, MSE_holdout, label="Holdout error",
             color="navy", lw=lw)
    plt.fill_between(alpha, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1,
                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
 
    return al,MSE

def kernel_ridge(X_train,y_train):
    al = [0.0001,0.001,0.01,0.1,1,10]
    gamma_range = [0.0001,0.001,0.01,0.1,1,10]
    parameters = {'kernel':('linear', 'rbf','polynomial'), 'alpha':al,'gamma':gamma_range}
    kernel = KernelRidge()
    clf = GridSearchCV(kernel, parameters)
    clf.fit(X_train, y_train)
    print clf.best_params_
    return clf.best_params_

def k_nearest_plot(X_train, y_train, X_holdout, y_holdout):
    alpha = np.logspace(0,3,num=10)
    MSE_train = []
    MSE_holdout = []
    alpha = [int(x) for x in alpha]
    for val in alpha:
        model = KNeighborsRegressor(n_neighbors=val)
        model.fit(X_train, y_train)
        y_pre_train = model.predict(X_train)
        y_pre_holdout = model.predict(X_holdout)
        MSE_train.append(mean_squared_error(y_train,y_pre_train))
        MSE_holdout.append(mean_squared_error(y_holdout,y_pre_holdout))
              
    al = alpha[np.argmin(MSE_holdout)]
    MSE = np.min(MSE_holdout)
    
    train_scores_mean = np.mean(MSE_train)
    train_scores_std = np.std(MSE_train)
    test_scores_mean = np.mean(MSE_holdout)
    test_scores_std = np.std(MSE_holdout)
        
    plt.title("mean squared error of k nearest neighbors regressor")
    plt.xlabel("n_neighbors")
    plt.ylabel("error")
    plt.ylim(0.0, 0.02)
    lw = 2
    plt.semilogx(alpha, MSE_train, label="Training error",
             color="darkorange", lw=lw)
    plt.fill_between(alpha, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
    plt.semilogx(alpha, MSE_holdout, label="Holdout error",
             color="navy", lw=lw)
    plt.fill_between(alpha, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1,
                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
 
    return al,MSE
    
    
    
