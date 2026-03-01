import numpy as np

x = np.array([
    [1500, 3, 10, 5],
    [1800, 4, 8, 8],
    [2400, 4, 5, 6],
    [3000, 5, 2, 3],
    [2000, 3, 12, 10],
    [1200, 2, 20, 15],
    [2200, 4, 7, 7],
    [1700, 3, 15, 12],
    [2800, 5, 4, 4],
    [1400, 2, 18, 14]
])

y = np.array([330, 370, 450, 540, 390, 250, 430, 350, 500, 280])

x = (x - x.mean(axis=0)) / x.std(axis=0)
def cost(x,y,w,b):
    cost = 0
    m,n = x.shape
    for i in range(m):
        y_pred = np.dot(x[i],w)+b
        res =( y_pred - y[i])**2
        cost += res
    return cost/(2 * m)

def compute__gradient(x,y,w,b):
    m,n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        y_pred = np.dot(x[i],w)+b
        dj_dw += (y_pred-y[i]) * x[i]
        dj_db += y_pred-y[i]
    return dj_dw/m , dj_db/m

def gradient(x,y,w,b,alpha,num_iters):
    
    for i in range(num_iters):
        dj_dw , dj_db = compute__gradient(x,y,w,b)
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db    
        if i % 100 == 0:
            print("Iteration", i, "Cost:", cost(x, y, w, b))
    return w,b

w,b = gradient(x,y,[0,0,0,0],0,0.01,1000)
print(w,b)