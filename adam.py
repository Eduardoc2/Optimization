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

def adam(initial_point, max_iter, tol):
    theta = np.copy(initial_point)
    path = [initial_point]  
    e = 0.01
    p1 = 0.9
    p2 = 0.999
    sigma = 1e-8
    r = np.zeros_like(theta)
    s = np.zeros_like(theta)
    for k in range(max_iter):
        grad = gradient(loss_function, theta)
        s = p1 * s + (1 - p1) * grad
        r = p2 * r + (1 - p2) * (grad **2)
        s_hat = s / (1 - p1**(k+1))
        r_hat = r / (1 - p2**(k+1))
        delta_theta = -e * s_hat / (np.sqrt(r_hat) + sigma)
        theta += delta_theta
        if np.linalg.norm(grad) < tol:
            return path, k
        path.append(theta)
    return path, k

initial_point = np.array([0.0, 1.0])  
max_iter = 500
tol = 1e-3  
path, k = adam(initial_point, max_iter, tol)
print("x =", path[-1], "-->","f(x) =", loss_function(*path[-1]))
print(k, "Iterations")