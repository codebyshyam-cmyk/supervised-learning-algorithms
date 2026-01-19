import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("C:\\Users\\saini\\Desktop\\supervised algorithm\\Salary Data.csv")
x_values =  np.array(data['YearsExperience'].values)
y_values =  np.array(data['Salary'].values)
w = 0
b = 0
n = len(x_values)
for i in range(10000):
    for i in range(n):
        f = w * x_values[i] + b
        dw = (f-y_values[i]) * x_values[i]
        db = (f-y_values[i])
        dw =(1/n)* dw
        db = (1/n)* db
        w = w - 0.01 * dw
        b = b - 0.01 * db
print(w,b)
new_y = x_values * w + b
plt.scatter(x_values,y_values)
plt.plot(x_values,new_y)
plt.show()