DESCRIPTION - 
This package includes the necessary components for recreating our model. It specifies the required software and libraries, and includes scripts for pulling training records and data for predictions. While our method includes pulling data for over 40,000 locations across the US, we have included a toy data set for demonstrative purposes. 

Software requirements:
Mathematica
Python 2.7
Python 2.7 libraries:
Scikit-learn
Matplotlib
Pandas
numpy
Included files and descriptions:
-	toy_data.xlsx - a toy data set that determines the locations for which GetData_training.nb and GetData_prediction.nb pull data
-	GetData_training.nb - Mathematica script for collecting weather data for over 40,000 training records for the building the model (based on the locations provided in toy_data.xlsx
-	GetData_prediction.nb - Mathematica script for collecting features in 2017 for the locations needed for the prediction (based on an input of zip codes)
-	main.py - primary script that reads data from toy_data.xlsx to split data in various training and test data sets, and performs functions in parameter_tuning.py, run_models.py, and some_function.py
-	parameter_tuning.py (used in main.py)
-	run_models.py  (used in main.py)
-	some_function.py  (used in main.py)

INSTALLATION - 
Ensure that Mathematica and Python 2.7 are installed. Install all the packages on Python. Put all the data files in a single folder under same directory with the scripts listed above in order to run them.

EXECUTION - 
To run a demo on your code, please perform the following steps:
1.	Install everything in INSTALLATION
2.	Put all data files in the same folder
3.	Run GetData_training.nb
4.	Run GetData_prediction.nb
5.	Run main.py using the output from steps 3 and 4
