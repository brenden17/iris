import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def loaddata(filename='pearson.dat'):
    data = np.loadtxt(filename)
    #print(data.shape)
    data = data * 2.54
    father, son = data[:, 0], data[:, 1]
    print('아버지의 평균키 {0}, 아들의 평균키 {1}'.format(father.mean(), son.mean()))
    print('아버지의 키의 분산 {0}, 아들의 키의 분산 {1}'.format(father.std(), son.std()))
    #draw_pearson(father, son)
    
def draw_pearson(x, y):
    plt.scatter(x, y)
    plt.title('The heihgt of father and son')
    plt.xlabel('The heihgt of father')
    plt.ylabel('The heihgt of son')
    plt.autoscale(tight=True)
    plt.grid()
    plt.show()

loaddata()


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.datasets import make_regression, make_s_curve

def create_save_reg(filename='reg.out'):
    #X, y = make_regression(n_samples=100, n_features=2, n_informative=1,random_state=0, noise=50)
    X, y = make_regression(n_samples=300, n_features=1, n_informative=1, bias=100, random_state=0, noise=40)
    reg = np.column_stack((X.ravel(), y))
    np.savetxt(filename, reg) 
    """
    X0, y0 = make_regression(n_samples=110, n_features=1, n_informative=1, bias=100, random_state=0, noise=20)
    X1, y1 = make_regression(n_samples=110, n_features=1, n_informative=1, bias=100, random_state=0, noise   =20)
    X2, y2 = make_regression(n_samples=110, n_features=1, n_informative=1, bias=100, random_state=0, noise=20)
    
    X0 -= 1
    y0 -= 50
    X2 += 1
    y2 += 50
    
    X = np.vstack((X0, X1, X2))
    y = np.vstack((y0, y1, y2))
    """
    
def loaddata(filename='reg.out'):
    data = np.loadtxt(filename)
    x, y = data[:,0], data[:,1]
    #print('x의 구간[{}, {}], y의 구간[{}, {}]'.format(x.max(), x.min(), y.max(), y.min()))
    return data[:,0], data[:,1]

def analysis():
    x, y = loaddata()
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.autoscale(tight=True)
    plt.grid()
    plt.show()

def error(f, x, y):
    return sp.sum((f(x)-y)**2)
    
def modeling1():
    x, y = loaddata()
    
    #polyorder = sp.polyfit(x, y, 1)
    polyorder, residuals, _, _, _ = sp.polyfit(x, y, 1, full=True)
    #print('반환된 차수 {}'.format(polyorder))
    print('잔차 {}'.format(residuals))
    model1 = sp.poly1d(polyorder)
    #print(model1)
    
    l = np.linspace(x.min(), x.max(),1000)
    plt.plot(l, model1(l))
    
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.autoscale(tight=True)
    plt.grid()
    plt.show()
    
    print('1.542->{}, 3.5->{}'.format(model1(1.542), model1(3.5)))
    print('오차 {:,}'.format(error(model1, x, y)))
    x = np.array([[n] for n in x])
    #print('오차 {:,}(np.linalg.lstsq)'.format(np.linalg.lstsq(x, y)))


def modeling2():
    x, y = loaddata()
    
    # change data
    y[x<-0.5] -= 200
    y[np.logical_and(x>1, x<1.5)] += 50
    y[x>1.5] += 550
    
    x = np.array([[n] for n in x])
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    
    clf = LinearRegression(fit_intercept=True, normalize=True)
    clf.fit(x, y)
    print'LinearRegression'
    print('linear regression coef : {}'.format(clf.coef_))
    print('linear regression score : {}'.format(clf.score(x, y)))
    print('linear regression error : {:,}'.format(error(clf.predict, x, y)))
    print clf.predict(1.542)
    print clf.predict(3.5)

    clf = Ridge(alpha=1.5, fit_intercept=True, normalize=False)
    clf.fit(x, y)
    print'ridge'
    print('ridge coef : {}'.format(clf.coef_))
    print('ridge coef : {}'.format(clf.score(x, y)))
    print('ridge coef : {:,}'.format(error(clf.predict, x, y)))
    print clf.predict(1.542)
    print clf.predict(3.5)
    
    clf = ElasticNet(fit_intercept=True)
    clf.fit(x, y)
    print'ElasticNet'
    print('{}'.format(clf.coef_))
    print('{}'.format(clf.score(x, y)))
    print('{:,}'.format(error(clf.predict, x, y)))
    print clf.predict(1.542)
    print clf.predict(3.5)
    
    from sklearn.svm import SVR
    clf = SVR()
    clf.fit(x, y) 
    print'SVR'
    print('{}'.format(clf.score(x, y)))
    #print clf.predict([1.542, 1])
    #print clf.predict(3.5)
    
def modeling3():
    x, y = loaddata()
    
    # change data
    y[x<-0.5] -= 200
    y[np.logical_and(x>1, x<1.5)] += 50
    y[x>1.5] += 550

    # change order
    order_number = 6
    l = np.linspace(x.min(), x.max(), 1000)
    fs = [sp.poly1d(sp.polyfit(x, y, n)) for n in range(1, order_number+1)]
    for n in range(0, order_number):
        plt.plot(l, fs[n](l), label=str(n+1))
        print('{} 차수의 오차 -> {:,}'.format(n+1, error(fs[n], x, y)))

    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.autoscale(tight=True)
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

def modeling4():
    x, y = loaddata()
    
    # change data
    y[x<-0.5] -= 200
    y[np.logical_and(x>1, x<1.5)] += 50
    y[x>1.5] += 550
    
    xy_range = [(x[x<-0.5], y[x<-0.5]), (x[np.logical_and(x>1, x<1.5)], y[np.logical_and(x>1, x<1.5)]), (x[x>1.5],  y[x>1.5])]
    fs = [sp.poly1d(sp.polyfit(xy[0], xy[1], 1)) for xy in xy_range]
    
    
    l_range = [np.linspace(x.min(), -0.5, 100), np.linspace(1, 1.5, 100), np.linspace(1.5, x.max(), 100)]
    for n in range(0, len(l_range)):
        plt.plot(l_range[n], fs[n](l_range[n]), label=str(n+1))
        
    print('오차 -> {:,}'.format(sum([error(fs[n], xy[0], xy[1]) for n, xy in enumerate(xy_range)])))
    
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.autoscale(tight=True)
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
        
# create_save_reg()
# modeling1()
# modeling2()
# modeling3()
modeling4()
