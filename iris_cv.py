from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pylab as plt

# load data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# kNN
knn = KNeighborsClassifier(n_neighbors=2) #n_neighbors=3,4,5
knn.fit(X, y)
print('kNN score')
print(knn.score(X, y))

####################################
# DummyClassifier
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='stratified',random_state=0) #stratified, most_frequent
dummy.fit(X, y)
print('dummy score')
print(dummy.score(X, y))

####################################
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
knn.fit(X, y)
print('knn accuracy score')
print(accuracy_score(knn.predict(X), y))

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# find k value for kNN
# load data
iris = datasets.load_iris()
X, y = iris.data, iris.target

result = []
score = []
k_range = range(8, 20)
for k in k_range:
    result.append(k)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    print('k is %d, score is %f' %(k, knn.score(X, y)))
    score.append(knn.score(X, y))

def draw(x, y, title='K value for kNN'):
    plt.plot(x, y, label='k value')
    plt.title(title)
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend(loc='best', framealpha=0.5, prop={'size':'small'})
    plt.tight_layout(pad=1)
    plt.gcf().set_size_inches(8,4)
    plt.show()

draw(result, score, 'kNN')
print("Score's mean %f" % (sum(score)/len(k_range)))

# train_test_split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

k_range = range(3, 20)
train_scores = []
test_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_score = knn.score(X_train, y_train)
    train_scores.append(train_score)
    test_score = knn.score(X_test, y_test)
    test_scores.append(test_score)
    print('k is %d, train score=%f, test score=%f' % (k, train_score, test_score))

print("Train score's mean %f" % (sum(train_scores)/len(k_range)))
print("Test score's mean %f" % (sum(test_scores)/len(k_range)))
    
plt.plot(k_range, train_scores, linestyle='-', label='train_score')
plt.plot(k_range, test_scores, linestyle='--', label='test_score')
plt.title('train score vs. test score')
plt.xlabel('k')
plt.ylabel('Score')
plt.grid(True)
plt.legend(loc='best', framealpha=0.5, prop={'size':'small'})
plt.tight_layout(pad=1)
plt.show()

####################################
# KFold
from sklearn.cross_validation import KFold
folds = 5
kf = KFold(len(y), n_folds=folds, indices=True)

k_range = range(3, 20)
score_means = []
for k in k_range:
    test_scores = []
    knn = KNeighborsClassifier(n_neighbors=k)
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        test_scores.append(score)
#         print('k is %d, test score is %f' % (k, score))
    score_means.append(sum(test_scores)/folds)
    print("K is %d, test score's mean %f" % (k, sum(test_scores)/folds))

draw(k_range, score_means)

####################################
# LeaveOneOut
from sklearn.cross_validation import LeaveOneOut
loo = LeaveOneOut(len(y))
print(loo)
knn = KNeighborsClassifier(n_neighbors=5)
scores = []
for train, test in loo:
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    knn.fit(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    scores.append(test_score)
#     print('test score=%f' % (test_score))
print('mean score %f' % (sum(scores)/150))

####################################
# cross_val_score
from sklearn.cross_validation import cross_val_score, KFold
kf = KFold(len(y), n_folds=5)
print(kf)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=kf)
print(sum(scores)/5)

####################################
# grid_search
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score, KFold, ShuffleSplit
kf = KFold(len(y), n_folds=5)
# kf = ShuffleSplit(n = len(X), n_iter=10)

k_range = range(3, 20)
#parameters = {'n_neighbors':k_range}
parameters = {'n_neighbors':k_range, 'weights':('uniform', 'distance')}
knn_base = KNeighborsClassifier()
grid_search = GridSearchCV(knn_base, parameters, cv=kf)
grid_search.fit(X, y)
print(grid_search)
print(grid_search.best_params_)
print(grid_search.grid_scores_)


####################################
# try GridSearchCV with KFold

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, KFold, ShuffleSplit
from sklearn.grid_search import GridSearchCV

iris = datasets.load_iris()
X, y = iris.data, iris.target

kf = KFold(len(y), n_folds=5)

## KNN
estimators = [('pca', PCA()), ('knn', KNeighborsClassifier())]
parameters = {'pca__n_components':(2, 3), 'knn__n_neighbors':(6,7,8,9,10,11,12,13), 'knn__weights':('uniform', 'distance')}

## SVM
# estimators = [('pca', PCA()), ('svm', SVC())]
# parameters = {'pca__n_components':(2, 3), 'svm__C':(0.4, 1, 1.5, 2, 3)}

## LogisticRegression
# estimators = [('pca', PCA()), ('lr', LogisticRegression())]
# parameters = {'pca__n_components':(2, 3), 'lr__penalty':( 'l1', 'l2')}

pipeline = Pipeline(estimators)
print(pipeline)
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=kf)
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
grid_search.fit(X, y)
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

