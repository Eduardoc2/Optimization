import numpy as np
def golden_section(f, y, d_j, a, b, tol):
    l = a + (1-0.618)*(b-a)
    u = a + 0.618*(b-a)
    for _ in range(100):
        if (b-a) < tol:
            break
        elif f(y + l*d_j) > f(y + u*d_j):
            a = l
            l = u
            u = a + 0.618*(b-a)
        elif f(y + l*d_j) < f(y + u*d_j):
            b = u
            u = l
            l = a + (1-0.618)*(b-a)
    return (a+b)/2

def cyclic_coordinate(f,x0,tol,f_opt):
    n=len(x0)
    d=np.eye(n)
    y=np.copy(x0)
    for _ in range(100):
        for j in range(n):
            l=f_opt(f,y,d[j],-5,5,0.1)
            y += l*d[j]
        if np.linalg.norm(y - x0) < tol:
            break
        else:
            x0=np.copy(y)
    return y, _

def f(x):
    return (x[0]-2)**4 + (x[0]-2*x[1])**2

x0 = np.array([0.0,3.0])
x_opt, k= cyclic_coordinate(f, x0, 0.05, golden_section)
print("x =", x_opt, "-->","f(x) =", f(x_opt))
print(k, "Iterations")