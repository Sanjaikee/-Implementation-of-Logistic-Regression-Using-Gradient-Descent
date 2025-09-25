# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. D3efine X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.
 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: sanjai s
RegisterNumber: 212223230186
*/

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1 (1).txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

print("Array of X") 
X[:5]

print("Array of y") 
y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
print("Exam 1- score Graph")
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

 plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
print("Sigmoid function graph")
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad
    
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print("X_train_grad value")
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print("Y_train_grad value")
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad 
   
 X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(" Print res.x")
print(res.fun)
print(res.x)   
    
 def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()
  
print("Decision boundary - graph for exam score")
plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1, 45, 85]),res.x))
print("Proability value ")
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
   
 print("Prediction value of mean")
np.mean(predict(res.x,X) == y)

```

## Output:

![Screenshot 2023-09-19 155328](https://github.com/22008496/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119476113/9310bee7-7783-448b-af0b-ff721807d114)
![Screenshot 2023-09-19 155335](https://github.com/22008496/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119476113/981a05a4-a0e1-4b45-a200-6b620e4861bf)
![Screenshot 2023-09-19 155346](https://github.com/22008496/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119476113/c0cb5695-90b6-4653-b421-ec205ae302d4)
![Screenshot 2023-09-19 155359](https://github.com/22008496/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119476113/8db18984-c894-42ac-aedd-8cad0f14210c)
![Screenshot 2023-09-19 155426](https://github.com/22008496/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119476113/5f4a976d-dd91-4c41-8940-fbf85fee722f)
![Screenshot 2023-09-19 155434](https://github.com/22008496/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119476113/beb065d0-5d83-491d-af5c-b7fbced3bcce)
![Screenshot 2023-09-19 155453](https://github.com/22008496/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119476113/46fb2db1-900b-40f1-91da-d8d6354f719a)
![Screenshot 2023-09-19 155513](https://github.com/22008496/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119476113/4f963992-8aa1-4a70-b88d-761e405a458f)
![Screenshot 2023-09-19 155522](https://github.com/22008496/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119476113/34173f4f-06d9-41d7-989f-b6a1c175e054)
![Screenshot 2023-09-19 155531](https://github.com/22008496/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119476113/43c0c2f6-5cc9-459f-98d5-6f12ed645093)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

