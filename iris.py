import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# load data
iris = datasets.load_iris()
print(iris.DESCR)
print(iris.target_names)
X, y = iris.data, iris.target
print(X[100:120, :])
print('Size of data : %s' % (X.shape, ))
print('Target value : %s' % np.unique(y))


sample = [[6, 4, 5.5, 2],]
#sample = [[6, 4, 5, 2],]
# try 1.
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
predicted_value = knn.predict(sample)
print('Try 1. -----')
print(iris.target_names[predicted_value])
print(knn.predict_proba(sample))

# try 2.
knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
knn.fit(X, y)
predicted_value = knn.predict(sample)
print('Try 2. -----')
print(iris.target_names[predicted_value])
print(knn.predict_proba(sample))

from sklearn.grid_search import GridSearchCV

print('Try 3 with GridSearchCV')
parameters = {'n_neighbors':(1, 3, 10), 'weights':('uniform', 'distance')}
knn_base = KNeighborsClassifier()
grid_search = GridSearchCV(knn_base, parameters)
grid_search.fit(X, y)
print(grid_search)
print(grid_search.best_params_)
print(grid_search.grid_scores_)

print('Try 4 with GridSearchCV and pipeline')
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

## KNN
estimators = [('pca', PCA()), ('knn', KNeighborsClassifier())]
parameters = {'pca__n_components':(2, 3), 'knn__n_neighbors':(1, 3, 10), 'knn__weights':('uniform', 'distance')}

## SVM
#estimators = [('pca', PCA()), ('svm', SVC())]
#parameters = {'pca__n_components':(2, 3), 'svm__C':(0.4, 1, 1.5, 2, 3)}


## LogisticRegression
#estimators = [('pca', PCA()), ('lr', LogisticRegression())]
#parameters = {'pca__n_components':(2, 3), 'lr__penalty':( 'l1', 'l2')}



pipeline = Pipeline(estimators)
print(pipeline)
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
grid_search.fit(X, y)
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
print(grid_search.best_estimator_.predict(sample))
print(iris.target_names[grid_search.best_estimator_.predict(sample)])
