import numpy as np

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

            
        
    