import numpy as np

def loss_function(x, y):
    return (x - 2)**4 + (x - 2*y)**2

def gradient(f, x, eps=1e-4):
    gradient = np.zeros(len(x))
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        gradient[i] = (f(*x_plus) - f(*x_minus)) / (2 * eps)
    return gradient

def nesterov_momentum(initial_point, max_iter, tol,vel=1,a=0.5):
    theta = np.copy(initial_point)
    path = [initial_point]  
    learning_rate = 0.1
    for k in range(max_iter):
        theta_prima = theta + a * vel
        grad= gradient(loss_function,theta_prima) 
        vel = vel*a - learning_rate*grad
        theta -= learning_rate * grad
        if np.linalg.norm(grad) < tol:
            return path, k
        path.append(theta)
    return path, k

initial_point = np.array([0.0, 1.0])  
max_iter = 100
tol = 1e-3  
path, k = nesterov_momentum(initial_point, max_iter, tol)
print("x =", path[-1], "-->","f(x) =", loss_function(*path[-1]))
print(k, "Iterations")