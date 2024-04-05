import numpy as np

def steepest_descent(f,x,tol,grad_f,f_opt):
    for k in range(100):
        grad = grad_f(f,x) 
        if np.linalg.norm(grad) < tol:
            break
        d = -grad
        l = f_opt()
        x += l*d
        
def f(x):
    return (x[0] - 2)**4 + (x[0] - 2*x[1])**2

def grad_f(x):
    return np.array([4*(x[0]-2)**3+2*(x[0]-2*x[1]),2*(x[0]-2*x[1])*(-2)])
