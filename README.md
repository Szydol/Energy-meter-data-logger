# Energy meter data logger
Data logger implementation for the master thesis using IEM2155/A9MEM2155 energy meter and raspberry Pi. The data was afterwards used in order to perform energy usage predictions and classification.

A9MEM2155.py - Script for logging the data from the meter into SQLite database.  
scan_wifi.py - Script for logging the user presence in home as supplementary data to the energy readings.  

random_forest_classifier.py - energy usage classification using sklearn RandomForestClassifier.  
multi_layer_perceptron.py - energy usage classification using sklearn MLPClassifier.  
logistic_regression.py - energy usage classification using sklearn LogisticRegression.  
LSTM.py - energy usage prediction using tensorflow LSTM.
