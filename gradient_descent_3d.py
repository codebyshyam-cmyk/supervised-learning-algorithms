import numpy as np
import matplotlib.pyplot as plt

def z_function(x,y):
    return np.sin(5 * x) * np.cos(5 * y) / 5

def gradient(x,y):
    return np.cos(5 * x) * np.cos(5 * y) , -np.sin(5 * x) * np.sin(5 * y)




x = np.arange(-1,1,0.05)
y = np.arange(-1,1,0.05)

X,Y = np.meshgrid(x,y)
Z= z_function(X,Y)
cur_x , cur_y  = 0.7,0.4
cur_z = z_function(cur_x,cur_y)

cur_x2 , cur_y2  = 0.2,-0.4
cur_z3 = z_function(cur_x2,cur_y2)

cur_x3 , cur_y3  = 0.1,0.9
cur_z3 = z_function(cur_x3,cur_y3)



learning_rate = 0.2
ax = plt.subplot(projection= '3d',computed_zorder= False)
for _ in range(1000):
    dx , dy = gradient(cur_x,cur_y)
    dx2 , dy2 = gradient(cur_x2,cur_y2)
    dx3 , dy3 = gradient(cur_x3,cur_y3)

    new_x = cur_x - learning_rate * dx
    new_y = cur_y - learning_rate * dy
    new_z = z_function(new_x,new_y)

    new_x2 = cur_x2 - learning_rate * dx2
    new_y2 = cur_y2 - learning_rate * dy2
    new_z2 = z_function(new_x2,new_y2)

    new_x3 = cur_x3 - learning_rate * dx3
    new_y3 = cur_y3 - learning_rate * dy3
    new_z3 = z_function(new_x3,new_y3)

    ax.plot_surface(X,Y,Z,cmap = 'viridis',zorder = 0)
    ax.scatter(new_x,new_y,new_z, color = 'magenta',zorder=1)
    ax.scatter(new_x2,new_y2,new_z2, color = 'green',zorder=1)
    ax.scatter(new_x3,new_y3,new_z3, color = 'yellow',zorder=1)

    cur_x = new_x
    cur_y = new_y
    cur_z = new_z

    cur_x2 = new_x2
    cur_y2 = new_y2
    cur_z2 = new_z2

    cur_x3 = new_x3
    cur_y3 = new_y3
    cur_z3 = new_z3
    plt.pause(0.01)
    ax.clear()
plt.show()