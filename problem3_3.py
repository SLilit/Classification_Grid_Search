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


from sklearn.svm import SVC

est = SVC(kernel = 'linear')
prm = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
result = classifier(est, prm, cv, X_data, y_data)
results[0] = results[0] + result

est = SVC(kernel = 'poly')
prm = {'C': [0.1, 1, 3], 'degree': [4, 5, 6], 'gamma': [0.1, 0.5]}
result = classifier(est, prm, cv, X_data, y_data)
results[1] = results[1] + result

est = SVC(kernel = 'rbf')
prm = {'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10]}
result = classifier(est, prm, cv, X_data, y_data)
results[2] = results[2] + result

from sklearn.linear_model import LogisticRegression

est = LogisticRegression()
prm = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
result = classifier(est, prm, cv, X_data, y_data)
results[3] = results[3] + result
print('LogisticRegression')

from sklearn.neighbors import KNeighborsClassifier

est = KNeighborsClassifier()
prm = {'n_neighbors': [3, 5, 10, 20, 35, 50], 'leaf_size': [5, 10, 15, 30, 60]}
result = classifier(est, prm, cv, X_data, y_data)
results[4] = results[4] + result
print('KNeighborsClassifier')



                        
with open ('C:\\Users\\DENVER\\edX\\AI\\Classification\\output3.csv', 'w') as f:
    writer = csv.writer(f, delimiter = ',')
    for row in results:
        writer.writerow(row)
