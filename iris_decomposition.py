import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


## laod data
iris = load_iris()
X, y = iris.data, iris.target

# print(iris.DESCR)

iris_df = pd.DataFrame(X, columns=list(iris.feature_names))
print(iris.feature_names)
# print(iris_df.describe())
iris_df.boxplot()
iris_df.plot()
# print(iris_df['sepal length (cm)'])
# plt.scatter(iris_df['sepal length (cm)'], iris_df['sepal width (cm)'], c=y)
plt.show()


## get correlation
y1 = np.vstack(y)
iris_df = pd.DataFrame(np.hstack((X, y1)),
        columns=iris.feature_names+['target'])
iris_df.corr()


## feature selection

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, KFold, ShuffleSplit
from sklearn.grid_search import GridSearchCV

iris = datasets.load_iris()
X, y = iris.data, iris.target  # 0.953
X = iris_df.iloc[:, [0,1]] # 0.733
# X = iris_df.iloc[:, [2,3]] # 0.947
kf = KFold(len(y), n_folds=10)

## KNN
estimators = [('knn', KNeighborsClassifier())]
parameters = {'knn__n_neighbors':(7,), 'knn__weights':('distance',)}

## SVM
# estimators = [('pca', PCA()), ('svm', SVC())]
# parameters = {'pca__n_components':(2, 3), 'svm__C':(0.4, 1, 1.5, 2, 3)}

## LogisticRegression
# estimators = [('pca', PCA()), ('lr', LogisticRegression())]
# parameters = {'pca__n_components':(2, 3), 'lr__penalty':( 'l1', 'l2')}

pipeline = Pipeline(estimators)
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=kf)
grid_search.fit(X, y)
print("Best score: %0.3f" % grid_search.best_score_)


## feature extraction
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.lda import LDA

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

lda = LDA(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

fa = FactorAnalysis(n_components=2)
X_r3 = fa.fit(X).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
              % str(pca.explained_variance_ratio_))
print(sum(pca.explained_variance_ratio_))
plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
plt.legend(loc="best")
plt.title('PCA of IRIS dataset')

plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
plt.legend(loc="best")
plt.title('LDA of IRIS dataset')

plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r3[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
plt.legend(loc="lower right", framealpha=0.5)
plt.title('Factor Analysis of IRIS dataset')

plt.show()

## find best prediction
from datetime import datetime

from sklearn import preprocessing
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.lda import LDA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, KFold, ShuffleSplit
from sklearn.grid_search import GridSearchCV

iris = datasets.load_iris()
X, y = iris.data, iris.target
kf = KFold(len(y), n_folds=10)

# X = preprocessing.normalize(X, norm='l2')
# X = preprocessing.scale(X)
# X = preprocessing.StandardScaler().fit_transform(X)
# X = preprocessing.MinMaxScaler().fit_transform(X)

def grid_search_cv(estimators, parameters, title):
    print("============================================================================")
    print(title)
    
    pipeline = Pipeline(estimators)
    # print(pipeline)
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=kf)
#     print("Performing grid search...")
#     print("pipeline:", [name for name, _ in pipeline.steps])
    tstart = datetime.now()
    grid_search.fit(X, y)
    t = datetime.now() -datetime.now()
    print("time : %f" % t.microseconds)
    print("Best score: %0.3f" % grid_search.best_score_)
#     print("Best parameters set:")
#     best_parameters = grid_search.best_estimator_.get_params()
#     for param_name in sorted(parameters.keys()):
#         print("\t%s: %r" % (param_name, best_parameters[param_name]))

## KNN
knn_estimators = [('knn', KNeighborsClassifier())]
knn_parameters = {'knn__n_neighbors':(7,), 'knn__weights':('distance',)}
grid_search_cv(knn_estimators, knn_parameters, 'Only KNN')

# KNN with PCA
knn_pca_estimators = [('pca', PCA()), ('knn', KNeighborsClassifier())]
knn_pca_parameters = {'knn__n_neighbors':(7,), 'knn__weights':('distance',)}
# knn_pca_parameters = {'pca__n_components':(2, 3), 'knn__n_neighbors':(6,7,8,9,10,11,12,13), 'knn__weights':('uniform', 'distance')}
grid_search_cv(knn_pca_estimators, knn_pca_parameters, 'KNN with PCA')

# KNN with FA
knn_fa_estimators = [('fa', FactorAnalysis()), ('knn', KNeighborsClassifier())]
knn_fa_parameters = {'knn__n_neighbors':(7,), 'knn__weights':('distance',)}
# knn_fa_parameters = {'fa__n_components':(2, 3), 'knn__n_neighbors':(6,7,8,9,10,11,12,13), 'knn__weights':('uniform', 'distance')}
grid_search_cv(knn_fa_estimators, knn_fa_parameters, 'KNN with FA')

# LDA
knn_lna_estimators = [('lda', LDA()), ('knn', KNeighborsClassifier())]
knn_fa_parameters = {'knn__n_neighbors':(7,8,9,10), 'knn__weights':('distance',)}
# knn_lna_parameters = {'lda__n_components':(2, 3), 'knn__n_neighbors':(6,7,8,9,10,11,12,13), 'knn__weights':('uniform', 'distance')}
grid_search_cv(knn_lna_estimators, knn_fa_parameters, 'KNN with LDA')

# LDA
lda_estimators = [('lda', LDA())]
lda_parameters = {'lda__n_components':(2, 3)}
grid_search_cv(lda_estimators, lda_parameters, 'only LDA')

# LDA
lda_lda_estimators = [('lda', LDA()), ('lda_c', LDA())]
lda_lda_parameters = {'lda__n_components':(2, 3)}
grid_search_cv(lda_lda_estimators, lda_lda_parameters, 'LDA with LDA')

# SVM
svm_estimators = [('svm', SVC())]
svm_parameters = {'svm__C':(0.4, 1, 1.5, 2, 3), 'svm__kernel':('linear', 'rbf', 'poly')}
grid_search_cv(svm_estimators, svm_parameters, 'only SVM')

# SVM
svm_estimators = [('pca', PCA()), ('svm', SVC())]
svm_parameters = {'pca__n_components':(2, 3), 'svm__C':(0.4, 1, 1.5, 2, 3), 'svm__kernel':('linear', 'rbf', 'poly')}
grid_search_cv(svm_estimators, svm_parameters, 'SVM with PCA')

## LogisticRegression
lr_estimators = [('pca', PCA()), ('lr', LogisticRegression())]
lr_parameters = {'pca__n_components':(2, 3), 'lr__penalty':( 'l1', 'l2')}
grid_search_cv(lr_estimators, lr_parameters, 'LR with PCA')
