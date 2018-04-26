## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to recognize seizure from EEG brain wave signals

import numpy as np
import pandas as pd
import time 

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

######################################### Reading and Splitting the Data ###############################################

# Read in all the data.
data = pd.read_csv('seizure_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data. DO NOT CHANGE.
random_state = 100 # DO NOT CHANGE

# XXX
# TODO: Split each of the features and labels arrays into 70% training set and
#       30% testing set (create 4 new arrays). Call them x_train, x_test, y_train and y_test.
#       Use the train_test_split method in sklearn with the paramater 'shuffle' set to true 
#       and the 'random_state' set to 100.
# XXX

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state= random_state, shuffle = True)


# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
# XXX
linear = LinearRegression()
linear.fit(X_train,y_train)

# XXX
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Use y_predict.round() to get 1 or 0 as the output.
# XXX

acc_test1 = accuracy_score(y_test,linear.predict(X_test).round())
acc_train1 = accuracy_score(y_train,linear.predict(X_train).round())
print "linear regression train accuracy %s \n test accuracy %s" %(acc_train1,acc_test1)


# ############################################### Multi Layer Perceptron #################################################
# XXX
# TODO: Create an MLPClassifier and train it.
# XXX
MLPC = MLPClassifier()
MLPC.fit(X_train,y_train)

# XXX
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX
acc_test2 = accuracy_score(y_test,MLPC.predict(X_test))
acc_train2 = accuracy_score(y_train,MLPC.predict(X_train))
print "MLP train accuracy %s \n test accuracy %s" %(acc_train2,acc_test2)



# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
# XXX

ran_forest = RandomForestClassifier()
ran_forest.fit(X_train,y_train)
# XXX
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX

acc_test3 = accuracy_score(y_test,ran_forest.predict(X_test))
acc_train3 = accuracy_score(y_train,ran_forest.predict(X_train))
print "Random Forest train accuracy %s \n test accuracy %s" %(acc_train3,acc_test3)

# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth' 
#       and select the combination that gives the best testing accuracy.
#       After fitting, print the best params, using .best_params_, and print the best score, using .best_score_.
# XXX

param_ran = {'n_estimators':(10,100,500,1000),'max_depth' : (None,4,5,6,7,8)}

gscv = GridSearchCV(ran_forest,param_ran,cv=10)
gscv.fit(X_train,y_train)
print "Random Forest paramter tuning",gscv.best_params_,"Best score",gscv.best_score_

acc_test3_tunning = accuracy_score(y_test,gscv.predict(X_test))
print "Random Forest test accuracy after tunning",acc_test3_tunning


# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Create a SVC classifier and train it.
# XXX
svc = SVC()
svc.fit(X_train, y_train)

# XXX
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX


print "SVM test accuracy",accuracy_score(y_test,svc.predict(X_test))
print "SVM train accuracy",accuracy_score(y_train,svc.predict(X_train))

# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear) 
#       and select the set of parameters that gives the best testing accuracy.
#       After fitting, print the best params, using .best_params_, and print the best score, using .best_score_.
# XXX
C_range = [0.001,0.01,0.1,1,10,100,1000]
param_svm = {'kernel':('rbf','linear'), 'C':C_range}

svm_cv = GridSearchCV(svc,param_svm,cv=10)  #Set cross-validation folds to 2


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
svm_cv.fit(X_train_scaled,y_train)
print "SVM paramter tuning",svm_cv.best_params_,"Best score",svm_cv.best_score_

svc2 = SVC(C = 1000, kernel = 'rbf')
svc2.fit(X_train,y_train)
print "SVM test accuracy after tuning",accuracy_score(y_test,svc2.predict(X_test))
# XXX 
# ########## PART C ######### 
# TODO: Print your CV's best mean testing accuracy and its corresponding mean training accuracy and mean fit time.
# 		State them in report.txt
# XXX
results = svm_cv.cv_results_
index = np.argmax(results['mean_test_score']) 
print "highest mean testing score",results['mean_test_score'][index]
print "corresponding mean training score",results['mean_train_score'][index]
print "correspoding mean fit time",results['mean_fit_time'][index]

