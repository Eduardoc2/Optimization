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
    return a, b

def cyclic_coordinate(f,x0,tol):
    n=len(x0)
    d=np.eye(n)
    y=np.copy(x0)
    for k in range(100):
        for j in range(n):
            l=1
            y += l*d[j]
        if np.linalg.norm(y - x0) < tol:
            break
        else:
            x0=y
    return y

            
        
