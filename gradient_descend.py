import numpy as np
import matplotlib.pyplot as plt

"""def y_function(x):
    return x**2

def y_derivative(x):
    return 2*x

x = np.arange(-100,100,0.1)"""

def y_function(x):
    return np.sin(x)

def y_derivative(x):
    return np.cos(x)
x = np.arange(-5,5,0.1)
y = y_function(x)

current_pos = (1,y_function(1))

learning_rate = 0.01

cur_x,cur_y = current_pos[0],current_pos[1]

for _ in range(1000):
    new_x = cur_x - (learning_rate * y_derivative(cur_x))
    plt.plot(x,y)
    plt.scatter(new_x,y_function(new_x),color='red')
    plt.pause(0.001)
    plt.clf()
    cur_x = new_x

plt.show()
