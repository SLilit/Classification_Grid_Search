from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
import numpy as np
import csv

data = np.genfromtxt('C:\\Users\\DENVER\\edX\\AI\\Classification\\input3.csv', delimiter = ",")
X_data = data[:, :2]
y_data = data[:, 2]
cv = StratifiedShuffleSplit(n_splits = 5, test_size = 0.4, random_state = 42)
results = [['svm_linear'], ['svm_polynomial'], ['svm_rbf'], ['logistic'],\
           ['knn'], ['decision_tree'], ['random_forest']]

def classifier(estimator, parameters, cv, X, y):
    clf = GridSearchCV(estimator, parameters, cv = cv, return_train_score = True )
    clf.fit(X, y)
    scores = clf.cv_results_
    max_train = 0
    max_test = 0
    for i in range(5):
        train = 'split' + str(i) + '_train_score'
        test = 'split' + str(i) + '_test_score'
        if max(scores[train]) > max_train:
            max_train = max(scores[train])
        if max(scores[test]) > max_test:
            max_test = max(scores[test])
    
    return [max_train, max_test]






                        

