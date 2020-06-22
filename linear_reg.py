import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def mean(values):
    return np.mean(values)

def coefficients(x,y,mean_x,mean_y):
    denominator=sum([(i-mean_x)**2 for i in x])
    numerator=sum([(i-mean_x)*(j-mean_y) for i,j in zip(x,y)])
    #y=m(x)+c
    m=numerator/denominator
    c=mean_y-m*(mean_x)
    return (m,c)

def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


def simple_linear_reg(x_train,y_train,x_test,y_test):
    mean_x=mean(x_train)
    mean_y=mean(y_train)
    m,c=coefficients(x_train,y_train,mean_x,mean_y)
    y_pred=[(i*m)+c for i in x_test]
    rmse=rmse_metric(y_test,y_pred)
    return (y_pred,rmse)

def plot_graph(x,y,m,c):
    mean_x=mean(x_train)
    mean_y=mean(y_train)
    m,c=coefficients(x_train,y_train,mean_x,mean_y)
    a=plt.gca()
    plt.xlabel('INDEPENDENT VARIABLE')
    plt.ylabel('DEPENDENT VARIABLE')
    
    for i,j in zip(x,y):
        a.scatter(i,j,c='#000000')
    x = np.linspace(0,6,100)
    plt.plot(x,m*x+c,'-b')
    plt.show()

#Enter train data:
train_data=[[1,1],[2,3],[4,3]]
#Enter test data
test_data=[[3,2],[5,5]]

xy_list=train_data+test_data
x=[i[0] for i in xy_list]
y=[i[1] for i in xy_list]
x_train = [row[0] for row in train_data]
y_train = [row[1] for row in train_data]
x_test=[row[0] for row in test_data]
y_test=[row[1] for row in test_data]

print('RMSE: %.3f' %simple_linear_reg(x_train,y_train,x_test,y_test)[1])
plot_graph(x,y,x_train,y_train)
