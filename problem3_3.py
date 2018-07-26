from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
import numpy as np
import csv

data = np.genfromtxt('C:\\Users\\DENVER\\edX\\AI\\Classification\\input3.csv', delimiter = ",")
X_data = data[:, :2]
y_data = data[:, 2]
cv = StratifiedShuffleSplit(n_splits = 5, test_size = 0.4, random_state = 42)
results = [['svm_linear'], ['svm_polynomial'], ['svm_rbf'], ['logistic'],\
           ['knn'], ['decision_tree'], ['random_forest']]


                        

