import numpy as np
from scipy.optimize import approx_fprime

def f(x,y):
    return (x - 2)**4 + (x - 2*y)**2

def gradiente(f,x,eps=1e-4):
    fx = (f(x[0] + eps, x[1]) - f(x[0] - eps, x[1])) / (2 * eps)
    fy = (f(x[0], x[1] + eps) - f(x[0], x[1] - eps)) / (2 * eps)
    return np.array([fx, fy])

def hessian(f, x, eps=1e-4):
    grad = gradiente(f, x, eps=1e-4)
  
    fxx = (f(x[0] + eps, x[1], grad[0] + eps, grad[1]) - 
           f(x[0] - eps, x[1], grad[0] - eps, grad[1])) / (4 * eps**2)
    fyx = (f(x[0], x[1] + eps, grad[0], grad[1] + eps) - 
           f(x[0], x[1] - eps, grad[0], grad[1] - eps)) / (4 * eps**2)
    fyy = (f(x[0], x[1] + eps, grad[0], grad[1] + eps) - 
           f(x[0], x[1] - eps, grad[0], grad[1] - eps)) / (4 * eps**2)
    fxy = (f(x[0] + eps, x[1], grad[0] + eps, grad[1]) - 
           f(x[0] - eps, x[1], grad[0] - eps, grad[1])) / (4 * eps**2)
    
    return np.array([[fxx, fxy], [fyx, fyy]])

def newton_ndim(f, hess, x0, tol, max_iter):
    x = []
    x.append(x0)
    for k in range(max_iter):
        hess = hessian(f,x[k])
        print(hess)
        grad = gradiente(f,x[k])

        x.append(x[k] - np.dot(np.linalg.inv(hess),grad))
        if np.linalg.norm(x[k+1] - x[k])  < tol:
            break
    return x[-1], k

x0 = np.array([0.00,3.00])
x_opt, k= newton_ndim(f,hessian,x0,tol=0.05,max_iter=100)
print("x =", x_opt, "-->","f(x) =", f(x_opt))
print(k, "Iterations")