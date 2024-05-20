import numpy as np
from scipy.linalg import *
from matplotlib import pyplot as plt
from matplotlib import cm, ticker
from math import *
import random
from sympy.solvers import solve
from sympy import Symbol
import sympy as sym
import pandas as pd
from scipy.io import loadmat
import scipy as scipy
import plotly.express as px
import plotly.graph_objects as go

I1 = 0.025
I2 = 0.045
m2 = 1
l1 = 0.3
l2 = 0.33
s2 = 0.16
K = 1/0.06
tau = 0.06

#SHOULDER PUIS ELBOW

a1 = I1 + I2 + m2*l2*l2
a2 = m2*l1*s2
a3 = I2
B = np.array([[0.5,0.025],[0.025,0.5]])

Bdyn = np.array([[0.5,0.025],[0.025,0.5]])
np.random.seed(0)

Bruit = True


def Bruitage(Bruit,NbreVar,Var):

    if Bruit :
        Omega_sens = np.diag(np.concatenate((np.ones(int(NbreVar/2)),np.zeros(int(NbreVar/2)))))
        motor_noise = np.concatenate((np.random.normal(0,np.sqrt(Var),int(NbreVar/2)),np.zeros(int(NbreVar/2)))).T
        Omega_measure = np.diag(np.ones(NbreVar)*Var)
        measure_noise = np.concatenate((np.random.normal(0,np.sqrt(Var),int(NbreVar/2)),np.zeros(int(NbreVar/2)))).T

    else:
        Omega_sens = np.zeros(NbreVar).T
        motor_noise = np.zeros(NbreVar).T
        Omega_measure = np.zeros(NbreVar).T
        measure_noise = np.zeros(NbreVar).T
    return Omega_sens,motor_noise,Omega_measure,measure_noise

def f1(a,Nf):
    Td = 0.066
    return np.exp(-(a/(0.56*Nf))**Nf)*Nf/Td*(1/(0.56*Nf))**Nf*a**(Nf-1)

def f2(a,Nf,u):
    Td = 0.066 + u*(0.05-0.066)
    return np.exp(-(a/(0.56*Nf))**Nf)*Nf/Td*(1/(0.56*Nf))**Nf*a**(Nf-1)

def g(a,Nf):
    return -f1(a,Nf)*a

def g2(a,Nf,u):
    return -f2(a,Nf,u)*a

def Compute_Command_NL(a,dottau):
    MmInv = np.array([[10,0],[0,5]])
    Mm = np.array([[.1,0],[0,.2]])
    l = 0.3
    Nf = 2.11+4.16*(1/l-1)

    FL = np.exp(-np.abs((l**1.93-1)/1.03)**1.87)
    FP = -0.02*np.exp(13.8-18.7*l)
    F = FL+FP 

    FinvMatrix = np.array([[1/f1(a[0],Nf),0],[0,1/f1(a[1],Nf)]])
    FMatrix =  np.array([[f1(a[0],Nf),0],[0,f1(a[1],Nf)]])
    GVector = np.array([[g(a[0],Nf)],[g(a[1],Nf)]])
    u = FinvMatrix @ (1/F *MmInv @ dottau.reshape((2,1)) - GVector)
    newtau = Mm@(FMatrix@u+GVector)*F
    if u[0] > a[0]: 
        FinvMatrix[0,0] = 1/f2(a[0],Nf,u[0]) 
        FMatrix[0,0] = f2(a[0],Nf,u[0]) 
        GVector[0] = g2(a[0],Nf,u[0])
    if u[1]>a[1]:
        FinvMatrix[1,1] = 1/f2(a[1],Nf,u[1]) 
        FMatrix[1,1] = f2(a[1],Nf,u[1]) 
        GVector[1] = g2(a[1],Nf,u[1])
    u = FinvMatrix @ (1/F *MmInv @ dottau.reshape((2,1)) - GVector)
    return u.reshape(2),newtau.reshape(2)

def Compute_Command(dottau,M,x,C,B):
    u = dottau/K + M@np.array([x[2],x[5]]) + C + B@np.array([x[1],x[4]])
    return u