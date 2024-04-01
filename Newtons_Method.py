def f(x): #Objective function
    if x<0:
        return 4*x**3 + 3*x**4
    else:
        return 4*x**3 - 3*x**4
def df(x):
    if x<0:
        return 12*x**2 + 12*x**3
    else:
        return 12*x**2 - 12*x**3
def d2f(x):
    if x<0:
        return 24*x + 36*x**2
    else:
        return 24*x - 36*x**2

def newton_optimization(f,df,d2f,x0,tol=1e-6,max_iter=100):
    x=[x0]
    for k in range(max_iter):
        x.append(x[k]-df(x[k])/d2f(x[k])) 
        if abs(x[k+1] - x[k])  < tol:
            break
    return k, x[-1]

k, x_min= newton_optimization(f, df, d2f,x0=-0.8)

print("x =", x_min, "-->","f(x) =", f(x_min))
print(k, "Iterations")