import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def dm01_模型欠拟合():
 np.random.seed(666)
 x=np.random.uniform(-3,3,size=100)
 y=0.5*x**2+ x +2 +np.random.normal(0,1,size=100)

 estimator=LinearRegression()

 X=x.reshape(-1,1)
 estimator.fit(X,y)

 y_predict=estimator.predict(X)

 myret=mean_squared_error(y,y_predict)
 print('myret-->',myret)

 plt.scatter(x,y)
 plt.plot(x,y_predict,color='r')
 plt.show(block=True)  # 确保图形窗口保持打开

# 调用函数
dm01_模型欠拟合()