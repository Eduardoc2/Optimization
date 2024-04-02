from sympy import symbols, diff
import numpy as np 

def f(x):
    return x**2+2*x
def df(x):
    return 2*x+2

def bisection(f,df,a, b, l):
    n = np.log(l/(b-a))/np.log(0.5)
    for k in range(int(np.ceil(n))):
        lambda_k=0.5*(a+b)
        if df(lambda_k) == 0:
            return lambda_k, a, b, k
        elif df(lambda_k) > 0:
            b = lambda_k
        else:
            a = lambda_k
    lambda_k=0.5*(a+b)
    return lambda_k ,a ,b, k

x_opt, a, b, k = bisection(f,df,-3,6,0.2)

print("Final interval: [", a,",",b,"]")
print("x =",x_opt, "-->","f(x) =",f(x_opt))
print(k, "Iterations")