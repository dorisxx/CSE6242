#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: doris xiang
"""

import itertools
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor



def show_metrics(RegressorName,y_true,y_pre):
    MAE = mean_absolute_error(y_true, y_pre)
    MSE = mean_squared_error(y_true,y_pre)
    R2 = r2_score(y_true,y_pre) 
    print "______________________________________________"
    print "Regressor: "+RegressorName
    print "R2: "+str(R2)
    print "Mean Absolute Error: "+str(MAE)
    print "Mean Squared Error: "+str(MSE)
    print "______________________________________________"
    print ""
    return (R2,MSE)


def least_square(X_train,y_train,X_test,y_test):
    reg_2 = linear_model.LinearRegression()
    reg_2.fit (X_train,y_train)
    y_pre = reg_2.predict(X_test)
    r2, mse = show_metrics('Least Squares',y_test,y_pre)
    return r2,mse
    
def ridge(X_train,y_train,X_test,y_test,val):
    model = linear_model.Ridge(alpha = val)
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    r2, mse = show_metrics('Ridge Regression',y_test,y_pre)
    return r2,mse

def lasso(X_train,y_train,X_test,y_test,val):
    model = linear_model.Lasso(alpha = val)
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    r2, mse = show_metrics('Lasso Regression',y_test,y_pre)
    return r2,mse

def ada_boost(X_train,y_train,X_test,y_test,val):
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=val, random_state=137)
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    r2, mse = show_metrics('Ada Boost',y_test,y_pre)
    return r2,mse

def gradient_boost(X_train,y_train,X_test,y_test,val):
    model = GradientBoostingRegressor(max_depth=4,n_estimators=val)
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    r2, mse = show_metrics('Gradient Boost',y_test,y_pre)
    return r2,mse

def kernel_ridge(X_train,y_train,X_test,y_test,val):
    kernel = KernelRidge(kernel = val['kernel'],alpha = val['alpha'], gamma = val['gamma'])
    kernel.fit(X_train,y_train)
    y_pre = kernel.predict(X_test)
    r2, mse = show_metrics('Kernel Ridge',y_test,y_pre)
    return r2,mse
    
def k_nearest(X_train,y_train,X_test,y_test,val):
    kernel = KNeighborsRegressor(n_neighbors=val)
    kernel.fit(X_train,y_train)
    y_pre = kernel.predict(X_test)
    r2, mse = show_metrics('k nearest neighbors regressor',y_test,y_pre)
    return r2,mse

def kernel_ridge_pre(X_train,y_train,X_pre,val):
    kernel = KernelRidge(kernel = val['kernel'],alpha = val['alpha'], gamma = val['gamma'])
    kernel.fit(X_train,y_train)
    y_pre = kernel.predict(X_pre)
    return y_pre

    
    
