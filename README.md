# scikit-learn을 활용한 기계학습

모바일로 문자를 보낼 때 다음 쓸만한 단어를 추천해 주고, 페이스북에 찍은 사진을 올리면 별도로 정보를 알리지 않아도 내 주변 사람 중 누구인지 인식한다. 아마존과 같은 인터넷 서점에 가면 내가 살만한 책이 화면 옆에 계속 따라 다린다. 잠들기 전, 어떻게 내가 쓸 단어, 주변 사람들, 기호를 이 기계들이 알고 있는지 불연듯 궁금해지며 의심이 간다. 쉽게 접할 수 있는 "빅데이터"라는 단어를 생각해 내고 그냥 넘어가려 해도 개인 정보가 유출된 것은 아닌지 영 찜찜하다. 이런 현상이 일어나게 된 이유는 여러가지일 수 있지만 몇 가지 혁신적인 일들로 가능해졌다.

가장 먼저 말할 수 있는 점은 컴퓨팅 환경에 변화이다. 모바일, 자동차 GPS, 서버 로그등 우리 주변의 다양한 하드웨어에서 데이터가 쏟아지고 있다. 네트워크나 여러 기계상에서 하는 행위들은 누구에게는 중요한 정보가 된다. 또한 저장 공간은 점점 더 저렴해지고 네트워트는 점점 더 빨라지고 있어, 이러한 데이터는 점점 더 축척된다. 다양한 데이터들이 생성될 수 있는 환경과 더불어 이를 저장하는데 많은 비용이 들지 않는다는 점이다. 다음은 기계 학습의 발전이다. 기계 학습이란 말 그대로, 엄마가 아이를 가르치듯이 기계를 학습시키는 일이다. 기계에게 우리가 가지고 있는 데이터로 학습을 시킨 후 기계에 학습한 데이터와 유사한 데이터를 입력하면 그와 관련된 정보를 예측하도록 하게 한다. 마치 선생님이 학생들에게 시험 범위안에서 학습을 시킨 후 시험을 보는 것과 유사하다. 가장 쉽게 접할 수 있는 예는 이메일 박스에 자동으로 분류하는 스팸 메일이다. 메일 시스템은 이전에 가지고 있던 스팸 이메일 정보를 가지고 나에게 새로 보내 온 이메일이 스팸인지 아닌지 분류한다. 확률과 통계학을 기반의 기계 학습의 발전 또한 우리 생활에 큰 영향을 미치게 되었다.

이 글은 간단한 예제를 통해 기계 학습에 대한 전반적인 이해를 높이도록 하는데 목적이 있다. 예제는 파이썬을 기반으로 하며 기계 학습을 쉽게 적용할 수 있도록 scikit-learn 라이브러리를 사용하고자 한다.
 
## 파이썬과 scikit 스택
파이썬은 비교적 배우기 쉬운 컴퓨터 언어로서 초기 구글이 시스템을 구축할 때 사용했고 유튜브의 추천 시스템을 만든 언어로 유명하다. 신생 다은 언어보다 다양하고 적용하기 쉬운 라이브러리를 가지고 있어 다양한 분야에서 사용되고 있으며 특히 공학자나 과학자들에게 인기가 높다. scikit 계열 라이브러리는 공학자나 과학자들이 파이썬을 사용하게 용이하도록 하였다. numpy는 matlab과 같이 매트리스 연산을 쉽게 할 수 있게 한다. scikit 라이브러리는 이산이나 공학에서 잘 사용하는 계산이나 함수를 모아 최적화하여 쉽게 쓰고 적용하도록 하겠다. 다음은 표는 주요 기능이다.

표

이와 더불어 pandas는 R의 데이터프레임과 유사하도록 구현되어 데이터를 손 쉽게 처리할 수 있게 하였다. 이러한 라이브러리는 모두 numpy를 기반으로 하기 때문에 numpy를 사용한다면 다른 라이브러리와 호환이 가능하여 기능 확장을 하거나 통합하는데 문제가 되지 않도록 되어 있다. 

## 기계학습(Machine Learnin) 소개
기계 학습은 기존 데이터를 사용하여 새로운 데이터의 정보를 유추할 수 있도록 한다. 이러한 기계 학습은 일반적으로 몇단계를 걸쳐 실행된다.

1. 데이터 정리 및 이해
1. 모델의 학습
1. 모델의 평가

가장 먼저 해야 할 일은 데이터의 충분한 이해이다. 데이터를 이해하여 어떤 데이터가 유효한 것인지 어떤 데이터가 유효하지 않은지 판단해야 한다. 이를 기반으로 데이터를 추상할 기법을 선택해야 한다. 하지만 너무 부담을 가질 필요는 없다. 분명 잘못된 판단으로 나중에 좀 더 나은 결과를 얻을 수 있다. 그런 이유 때문에 파이썬을 선택한다. 파이썬은 기민하여 다양하게 데이터를 변경하거나 추가, 삭제 할 수 있다.
다음은 데이터에 대한 모델의 학습 단계이다. 데이터를 가장 잘 추상화 할 수 있는 기법을 선택하여 기계를 학습시킨다. 이를테면, 신경망, 서포트 벡터 머신, K평균등 다양한 기법으로 모델을 생성하게 된다. 모델에는 각기 다른 매개 변수가 있기 때문에 같은 입력과 같은 모델이더라도 매개 변수를 변경하여 전혀 다른 결과를 얻을 수도 있다.
마지막으로 학습한 모델의 평가가 필요하다. 다양한 모델과 각 모델의 여러 조건으로 데이터를 예측할 수 있고 이 예측이 얾마나 정확한가를 평가해야 더 나은 모델이나 기법을 선택할 수 있게 된다.

기계 학습 알고리즘을 학습 방법에 따라 크게 두가지로 나눌 수 있다. 지도 학습(supervised learning)은 데이터에 예측하고자 하는 목적 속성(target feature)이 있어  예측 모델(predictive model)을 구축하는 반면, 비지도 학습(unsupervised learning)은 목적 속성이 없이 기술 모델(descriptive model)을 구축한다. 전자의 대표적인 예로서, 스팸 메일 분류를 말할 수 있다. 스팸 메일과 햄 메일(정상 메일)로 구분할 수 있는 목적 속성을 가진 데이터를 가지고 학습한 후 스팸 메일을 분류한다. 반면, 후자의 예로서 영화 추천과 같이 나와 기호가 같은 사용자가 본 영화를 찾아 추천해 주는 시스템이다. 

## scikit-learn 소개
scikit-learn은 2007년 구글 썸머 코드에서 처음 구현되기 시작하여 이제는 파이썬으로 구현된 가장 유명한 기계 학습 라이브러리가 되었다. scikit-learn의 장점은 라이브러리 외적으로는 scikit 스택을 사용하고 있기 때문에 다른 라이브러리와 호환성이 매우 좋다. 또한 라이브러리 내적으로는 통일된 인터페이스를 가지고 있기 때문에 매우 간단하게 여러 기법을 적용할 수 있어 최상의 결과를 얻을 수 있게 한다.
라이브러리의 구성은 크게 지도 학습, 비지도 학습, 모델 선택 및 평가, 데이터 변환으로 나눌 수 있다(scikit-learn 사용자 가이드 참조). 지도 학습에는 서포트 벡터 머신, 나이브 베이즈, 결정 트리등이 있으며 비지도 학습에는 군집화, 이상치 검출등이 있다. 모델 선택 및 평가에는 교차 검증(cross-validation), 파이프라인(pipeline)등 있으며 마지막으로 데이터 변환에는 속성 추출, 선처리등이 있다.
클래스별로 보자면 다음과 같은 클래스가 있다.
각 기법들이 가지고 있어야 하는 가장 기본 BaseEstimator가 있으며 기법의 공통적인 부분을 가지고 있는 ClassifierMixin, 
RegressorMixin, ClusterMixin들이 있어 기법들은 각각의 기반 클래스를 상속 받아 구현된다. 대부분의 클래스는 기존 데이터를 적합화하는 fit 메쏘드와 데이터를 예측하는 predict 메쏘드를 가지고 있다.

## 예제
지금부터 간단한 예제를 통해 기계 학습이 어떻게 적용될 수 있는지를 알아보도록 하겠다. 기계 학습에서 자주 사용하는 1936년에 만들어진 피셔경의 붓꽃(iris) 데이터를 사용하겠다. 이 데이터를 사용하는 이유는 데이터를 구하기 매우 쉽고 데이터 크기와 속성이 적기 때문이다. 좀 더 나가기 앞서, scikit 스택과 scikit-learn(코드에서는 sklearn으로 사용한다.)을 설치하도록 하자.
윈도우 
http://sourceforge.net/projects/numpy/
http://sourceforge.net/projects/scikit-learn/
http://sourceforge.net/projects/scipy/
에서 각각의 라이브러리를 내려 받아 설치 한 후 파이썬 셀에서 다음을 실행해 본다. 잘 설치되었다면 아무 메시지가 나오지 않는다.
import numpy as np
import scipy
import sklearn

iris 데이터는 scikit-learn을 설치하면 기본으로 포함되어 있다. 

from sklearn import datasets

iris = datasets.load_iris()
print(iris.DESCR)
print(iris.target_names)

iris.DESCR을 하면 iris 데이터의 기본 정보가 출력된다. 

Data Set Characteristics:
    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
    :Summary Statistics:
    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20  0.76     0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

150개의 데이터와 4개의 속성인 꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비가 나타나며 각 속성의 최소값, 최대값, 평균등을 볼 수 있다. 속성 속성은 0, 1, 2이며 즉, Setosa, Versicolour, Virginica로 세개의 꽃 종류가 된다.
가장 위에 있는 10개의 데이터를 살펴보면 다음과 같다.

print(X[0:10, :])

[[ 5.1  3.5  1.4  0.2]
 [ 4.9  3.   1.4  0.2]
 [ 4.7  3.2  1.3  0.2]
 [ 4.6  3.1  1.5  0.2]
 [ 5.   3.6  1.4  0.2]
 [ 5.4  3.9  1.7  0.4]
 [ 4.6  3.4  1.4  0.3]
 [ 5.   3.4  1.5  0.2]
 [ 4.4  2.9  1.4  0.2]
 [ 4.9  3.1  1.5  0.1]]

iris 객체는 data와 target 속성을 가지고 있고 data는 iris 데이터의 속성들이며, target은 목적 속성이다.
데이터를 편의상 X, y로 지정하도록 하자.

X, y = iris.data, iris.target
print('Size of data : %s' % (X.shape, )) 
print('Target value : %s' % np.unique(y))

Size of data : (150, 4)
Target value : [0 1 2]

이로써 데이터는 준비되었고 sample이라는 임의에 데이터가 어떤 꽃에 속하는지 찾아 보도록 하자.  이를 위해 기계 학습 기법을 선택해야 할 차례이다. 

sample = [[3, 5, 4, 2],]


### KNN 소개
k 최근접 이웃 알고리즘(kNN,  k-Nearest Neighbors algorithm, http://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)은 매우 단순한 알고리즘이다. kNN은 예측하고자 하는 데이터에 가까이 있는 주변에 있는 k개의 데이터를 보고 데이터를 예측하는 방법이다. 이를테면, k가 3이라면 예측하고자 하는 데이터에 가까이 있는 주변 3개 데이터의 목적 속성에 따라 예측하게 된다. 이 알고리즘에는 k뿐만 아니라 몇가지 매개 변수(parameter)가 있는데 주변 데이터의 중요도에 따른 가중치와 주변 데이터를 계산할 알고리즈등이 있다. scikit-learn에 있는 KNeighborsClassifier를 사용한다. 초기 k값인 n_neighbors는 1로 설정한다. 즉 가장 가까운 하나의 데이터를 보고 예측하겠다는 의미이다. fit메쏘드에 X, y를 입력하여 적합화하고 predict 메쏘드에 sample을 넣고 예측값을 확인해 보자.

from sklearn.neighbors import KNeighborsClassifier
# try 1.
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
predicted_value = knn.predict(sample)
print(iris.target_names[predicted_value])
print(knn.predict_proba(sample))

['virginica']
[[ 0.  0.  1.]]

virginica로 예측했다. 확률도 k가 1이기 없기 때문에 virginica가 100%가 된다. 다음 코드 조각과 같이 k를 3으로, weight를 'distance'로 변경하고 실행해 보자.

knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(X, y)
predicted_value = knn.predict(sample)
print(knn.predict_proba(sample))
print(iris.target_names[predicted_value])

['versicolor']
[[ 0.          0.66473133  0.33526867]]


이번에는 versicolor로 예측했고 확률적으로 versicolor는 66%이고, virginica는 33%이다. virginica보다 versicolor이 확률적으로 높기 때문에 versicolor로 예측하게 되었다. 여기에서는 예측값에 대한 정확도를 측정하지 않겠다. 더 알아보기를 참고하도록 하자. 앞서 언급했듯이 기계 학습 알고리즘도 중요하지만 매개 변수의 선택도 매우 중요하다. scikit-learn 라이브러리에는 이 부분을 자동화 할 수 있도록 GridSearchCV 클래스를 제공한다. GridSearchCV의 매개 변수는 알고리즘과 관련 매개 변수들이다. 다음 코드 조각은 k인 n_neighbors을 1, 3, 10로 설정하고 weights를 uniform, distance로 설정하여 각각의 매개 변수 조합으로 최적의 조건을 찾는다.
 
parameters = {'n_neighbors':(1, 3, 10), 'weights':('uniform', 'distance')}
knn_base = KNeighborsClassifier()
grid_search = GridSearchCV(knn_base, parameters)
grid_search.fit(X, y)
print(grid_search)
print(grid_search.best_params_)
print(grid_search.grid_scores_)


{'n_neighbors': 3, 'weights': 'uniform'}
[mean: 0.96000, std: 0.00000, params: {'n_neighbors': 1, 'weights': 'uniform'}, mean: 0.96000, std: 0.00000, params: {'n_neighbors': 1, 'weights': 'distance'}, mean: 0.96667, std: 0.01886, params: {'n_neighbors': 3, 'weights': 'uniform'}, mean: 0.96667, std: 0.01886, params: {'n_neighbors': 3, 'weights': 'distance'}, mean: 0.95333, std: 0.00943, params: {'n_neighbors': 10, 'weights': 'uniform'}, mean: 0.96667, std: 0.02494, params: {'n_neighbors': 10, 'weights': 'distance'}]

다른 조건에서 같은 결과가 나온다면 적은 복잡도를 만드는 매개 변수를 선택한다. 결과를 보자면 {'n_neighbors': 3, 'weights': 'uniform'} 조건일 때 가장 예측력이 좋다.

### 더 읽어보기
scikit-learn의 좀 더 생산적인 기능으로 pipeline과 라이브러리의 공통적인 인터페이스이다. pipeline은 기계 학습 알고리즘을 적용하기 전에 전처리와 같은 처리를 연결해서 구현할 수 있도록 한다. 이를테면 iris 데이터의 차원을 줄여 기계 학습 알고리즘에 적용할 수도 있다. 차원의 저주라는 말이 있듯이 높은 차원은 좋은 결과를 주지 못하는 경우가 많다. pipeline을 사용하면 PCA를 적용하고 기계 학습 알고리즘을 바로 적용할 수 있다.

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

## KNN
estimators = [('pca', PCA()), ('knn', KNeighborsClassifier())]
parameters = {'pca__n_components':(2, 3), 'knn__n_neighbors':(1, 3, 10), 'knn__weights':('uniform', 'distance')}

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


Pipeline(knn=KNeighborsClassifier(algorithm=auto, leaf_size=30, metric=minkowski,
           n_neighbors=5, p=2, weights=uniform),
     knn__algorithm=auto, knn__leaf_size=30, knn__metric=minkowski,
     knn__n_neighbors=5, knn__p=2, knn__weights=uniform,
     pca=PCA(copy=True, n_components=None, whiten=False), pca__copy=True,
     pca__n_components=None, pca__whiten=False)
Performing grid search...
('pipeline:', ['pca', 'knn'])
Fitting 3 folds for each of 12 candidates, totalling 36 fits
[Parallel(n_jobs=-1)]: Done   1 jobs       | elapsed:    0.0s
[Parallel(n_jobs=-1)]: Done  30 out of  36 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=-1)]: Done  36 out of  36 | elapsed:    0.0s finished
Best score: 0.973
Best parameters set:
    knn__n_neighbors: 3
    knn__weights: 'uniform'
    pca__n_components: 2
[1]
['versicolor']

pipeline으로 작업을 연결할 수 있으며 주어진 매개 변수를 적용하여 최적의 결과를 찾는다. grid_search.best_estimator_.predict(sample)을 통해 최적의 조건으로 versicolor로 예측했다.

다른 장점인 통일된 인터페이스는 다른 기계 학습 알고리즘을 큰 변경없이 바로 적용할 수 있게 한다. 위의 소스에서 

estimators = [('pca', PCA()), ('knn', KNeighborsClassifier())]
parameters = {'pca__n_components':(2, 3), 'knn__n_neighbors':(1, 3, 10), 'knn__weights':('uniform', 'distance')}

을 아래와 같이 알고리즘과 매개 변수만 변경해도 바로 적용할 수 있다.

## SVM
estimators = [('pca', PCA()), ('svm', SVC())]
parameters = {'pca__n_components':(2, 3), 'svm__C':(0.4, 1, 1.5, 2, 3)}

## 결론
지금까지 기계학습과 파이썬으로 구현된 scikit-learn을 간략하게 살펴보고 붓꽃 예제에 적용해 보았다. 기계 학습에는 다양한 기법이 있고 여기에서 다루지 않은 적용된 기법에 대한 평가에 문제가 있다. scikit-learn은 문서화가 매우 잘되어 있어 기계 학습 기법을 학습한 후 scikit-learn 문서를 찾아 보면 그리 어렵지 않게 실제 문제에 적용할 수 있다.


https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors/classification.py
http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
http://www.astro.washington.edu/users/vanderplas/Astr599/notebooks/17_SklearnIntro
http://ko.wikipedia.org/wiki/%EC%A3%BC%EC%84%B1%EB%B6%84_%EB%B6%84%EC%84%9D

