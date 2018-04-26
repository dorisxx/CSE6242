#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: doris xiang
"""
import some_functions
import parameter_tuning
import run_models
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd
from sklearn.metrics import mean_squared_error


#my favorite prime number which is also the mysterious Fine structure constant
RANDOM_STATE = 137 


def main():
    #concatenate sheets
    file_path = ['data/Sheet%d.csv' %(i+2) for i in range(20)]
    X_list = [some_functions.load_csv(f)[0] for f in file_path]
    y_list = [some_functions.load_csv(f)[1] for f in file_path]
    X = pd.concat(X_list,ignore_index=True)
    y = pd.concat(y_list,ignore_index=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= RANDOM_STATE, shuffle = True)
    X_train2, X_holdout,y_train2, y_holdout = train_test_split(X_train,y_train, test_size = 0.3, random_state = RANDOM_STATE, shuffle = True)
    X_train,y_train = some_functions.standarlize(X_train,y_train)
    X_train2,y_train2 = some_functions.standarlize(X_train2,y_train2)
    X_test,y_test = some_functions.standarlize(X_test,y_test)
    X_holdout,y_holdout = some_functions.standarlize(X_holdout,y_holdout)
    
    #tune parameters     
    ridge_lamda,ridge_mse = parameter_tuning.ridge_plot(X_train2,y_train2,X_holdout,y_holdout)
    lasso_alpha,lasso_mse = parameter_tuning.lasso_plot(X_train2,y_train2,X_holdout,y_holdout)
    ada_n,ada_mse = parameter_tuning.decision_tree_plot(X_train2,y_train2,X_holdout,y_holdout)
    gradient_n,gradient_mse = parameter_tuning.gradient_boost_plot(X_train2,y_train2,X_holdout,y_holdout)
    kernel_ridge_params = parameter_tuning.kernel_ridge(X_train2,y_train2)
    k_nearest_k, k_nearest_mse = parameter_tuning.k_nearest_plot(X_train2,y_train2,X_holdout,y_holdout)
    
    #test models using best parameters
    r2_1_,mse_1_= run_models.least_square(X_train,y_train,X_train,y_train)
    r2_1,mse_1= run_models.least_square(X_train,y_train,X_test,y_test)
    
    r2_2_,mse_2_= run_models.ridge(X_train,y_train,X_train,y_train,ridge_lamda)
    r2_2,mse_2= run_models.ridge(X_train,y_train,X_test,y_test,ridge_lamda)
    
    r2_3_,mse_3_= run_models.lasso(X_train,y_train,X_train,y_train,lasso_alpha)
    r2_3,mse_3= run_models.lasso(X_train,y_train,X_test,y_test,lasso_alpha)
    
    r2_4_,mse_4_= run_models.ada_boost(X_train,y_train,X_train,y_train,ada_n)
    r2_4,mse_4= run_models.ada_boost(X_train,y_train,X_test,y_test,ada_n)
    
    r2_5_,mse_5_= run_models.gradient_boost(X_train,y_train,X_train,y_train,gradient_n)
    r2_5,mse_5= run_models.gradient_boost(X_train,y_train,X_test,y_test,gradient_n)
    
    r2_6_,mse_6_= run_models.kernel_ridge(X_train,y_train,X_train,y_train,kernel_ridge_params)
    r2_6,mse_6= run_models.kernel_ridge(X_train,y_train,X_test,y_test,kernel_ridge_params)
    
    r2_7_,mse_7_= run_models.k_nearest(X_train,y_train,X_train,y_train,k_nearest_k)
    r2_7,mse_7= run_models.k_nearest(X_train,y_train,X_test,y_test,k_nearest_k)
    
    #make the prediction on final model
    zipcodes, texas_data = some_functions.load_prediction('data/texas_features.csv')
    texas_data = some_functions.standarlize(texas_data,0)[0]
    
    y_pre2 = run_models.kernel_ridge_pre(X_train,y_train,texas_data, kernel_ridge_params)
    y_pre3 = some_functions.min_max(y_pre2)
    output = pd.DataFrame()
    output['zipcodes'] = zipcodes
    output['risk score'] = y_pre3
    output.to_csv("risk_score.csv",sep = ',', index = False)
    
if __name__ == '__main__':
    main()
    
    
    


    

        
    
        
    