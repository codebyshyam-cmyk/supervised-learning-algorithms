import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



training_set = pd.read_csv("Salary Data.csv")
x = training_set["YearsExperience"].values
y = training_set["Salary"].values
plt.scatter(x,y)
plt.xlabel("years of experience")
plt.ylabel("salary ")

def cost_fuction(x,y,w,b):
    n = len(x)
    cost_sum = 0
    for i in range(n):
        f = w * x[i] + b #f here is the predicted value and y is the actual value
        cost = (f - y[i] )**2
        cost_sum += cost
    mse = (1/2*n)*cost_sum
    return mse

def gradient_function(x,y,w,b):
    n = len(x)
    dc_dw = 0
    dc_db = 0
    for i in range(n):
        f = w * x[i] + b

        dc_dw += (f-y[i]) * x[i]
        dc_db += (f-y[i])
    dc_dw = (1/n)* dc_dw
    dc_db = (1/n)* dc_db
    return dc_dw,dc_db

def gradient_descent(x,y,alpha,iterations):
    w = 0
    b = 0
    for i in range(iterations):
        dc_dw,dc_db = gradient_function(x,y,w,b)

        w =  w - alpha * dc_dw
        b =  b - alpha * dc_db

        print(f" Iterations {i} : cost {cost_fuction(x,y,w,b)}")

    return w,b

learning_rate =  0.01
iterations =  100000

final_w,final_b = gradient_descent(x,y,learning_rate,iterations)

x_vals = np.linspace(min(x),max(x),100)
y_vals = final_w* x_vals + final_b
plt.plot(x_vals,y_vals,color='red',label = 'regression line')
plt.show()