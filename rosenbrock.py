import numpy as np

def new_directions(l, d, n):
    a = np.zeros_like(d)
    b = np.zeros_like(d)
    for j in range(n):
        if l[j] == 0:
            a[j] = d[j]
        else:
            a[j] = sum(l[i] * d[i] for i in range(n))
        
        if j == 1:
            b[j] = a[j]
        else:
            b[j] = a[j] - sum(np.dot(a[j], d[i]) * d[i] for i in range(1, j-1))
        
        d[j] = b[j] / np.linalg.norm(b[j])
    
    return d

def rosenbrock(f,x,tol,f_opt, new_dir):
    n=len(x)
    d=np.eye(n)
    y=np.copy(x)
    for _ in range(100):
        for j in range(n):
            l=f_opt(f,y,d[j],-5,5,0.1)
            y += l*d[j]
        if np.linalg.norm(y - x) < tol:
            break
        else:
            x=np.copy(y)
        d=new_dir(l,d,n)
    return y, _
