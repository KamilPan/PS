import numpy as np
from scipy import optimize
from scipy.optimize import minimize


def u(z,vartheta=-2):
    return (z)**(1+vartheta)/(1+vartheta)

def prem(p,q):
    return p*q

def V(x,q,p=0.2,y=1):
    return p*u(y-x+q-prem(p,q),vartheta) + (1-p)*u(y-prem(p,q),vartheta)
