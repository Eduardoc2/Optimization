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

def new_direction(l, d, n):
    a = np.zeros_like(d)
    b = np.zeros_like(d)
    for j in range(n):
        if l[j] == 0:
            a[j] = d[j]
        else:
            a[j] = sum(l[i] * d[i] for i in range(j,n))
        
        if j == 1:
            b[j] = a[j]
        else:
            b[j] = a[j] - sum(np.dot(a[j], d[i]) * d[i] for i in range(j-2))
    
        d[j] = b[j] / np.linalg.norm(b[j])
    
    return d

def rosenbrock(f,x,tol,f_opt, new_dir):
    n=len(x)
    d=np.eye(n)
    y=np.copy(x)
    l=np.zeros(n)
    for _ in range(100):
        for j in range(n):
            l[j]=f_opt(f,y,d[j],-5,5,0.01)
            #print(_,x,y,d,l[j],"---")
            y += l[j]*d[j]
        if np.linalg.norm(y - x) < tol:
            break
        else:
            x=np.copy(y)
        d=new_dir(l,d,n)
    return y, _

def f(x):
    return (x[0]-2)**4 + (x[0]-2*x[1])**2

x0 = np.array([0.00,3.00])
x_opt, k= rosenbrock(f, x0, 0.005, golden_section,new_direction)
print("x =", x_opt, "-->","f(x) =", f(x_opt))
print(k, "Iterations")