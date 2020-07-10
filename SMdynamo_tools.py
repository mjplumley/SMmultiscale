#!/usr/bin/env python
# coding: utf-8

# Tools for running the single mode dynamo codes

import numpy as np
import scipy.sparse

def cheb_grid(N):
    ''' this function returns a chebyshev grid on the interval [-1,1] with entry x(1)=1 and x(end)=-1 '''

    return np.transpose( np.cos(np.pi*np.linspace(0,1,N)) )

def phys2cheb(vec):
    N = len(vec) - 1
    vec_even = np.concatenate( (vec, np.flipud(vec[1:N]) ))
    F = np.real( np.fft.fft(vec_even) )
    if N > 1:
        y = (1/N)*F[0:N+1]
        y[0] = 0.5*y[0]
        y[-1] = 0.5*y[-1]
    else:
        y = vec
    return y

def cheb2phys(vec):
    N = len(vec)
    if N==1:
        y = vec
    else:
        vec_even = (N-1)*np.concatenate( (vec, np.flipud(vec[1:N-1]) ))
        vec_even[0] = 2*vec_even[0]
        vec_even[N-1] = 2*vec_even[N-1]
        
        f = np.real( np.fft.ifft(vec_even) ) 
        y = f[0:N]
    return y

def myC(p):
    if p == 0:
        return 2
    elif p < 0:
        return 0
    else:
        return 1


def intg( N ):
#INTG Spectral matrix for single integral -- Integration matrix

    ds = np.zeros( (N,N) )

    # lower diagonal n-1
    for n in range(0,N-1):
        ds[n+1,n] = myC(n)/(2*(n+1))

    # upper diagonal n+1
    for n in range(0,N-2):
        ds[n+1,n+2] = -1/(2*(n+1))
    return ds

def intg2( N ):
#INTG2 Spectral matrix for double integral

    ds = np.zeros( (N,N) )

    # 2nd lower diagonal n-2
    for n in range(1,N-1):
        ds[n+1,n-1] = myC(n-1)/(4*(n+1)*n)

    # diagonal n
    for n in range(1,N-1):
        ds[n+1,n+1] = -myC(n)/(2*(n)*(n+2))
    
    # 2nd upper diagonal n+2
    for n in range(1,N-3):
        ds[n+1,n+3] = 1/(4*(n+1)*(n+2))

    return ds


def identity(N,zRows):
    Iz  = np.eye(N)
    for i in range(zRows):
        Iz[i,i] = 0
    return Iz

# TAU LINES FOR THE BOUNDARY CONDITIONS
def boundaries_w_top(N):
    T = np.zeros( (N,N) )
    for i in range(N):
        T[0,i] = (-1)**i
    return T

def boundaries_w_bottom(N):
    T = np.zeros( (N,N) )
    for i in range(N):
        T[0,i] = 1
    return T

def boundaries_T(N):
    T = np.zeros( (N,N) )
    for i in range(N):
        T[0,i] = 1
        T[1,i] = (-1)**i
    return T

def boundaries_B(N,bc):
    T = np.zeros( (N,N) )
    if bc == 0:
        for i in range(N):
            T[0,i] = i**2
            T[1,i] = (-1)**i*i**2
    elif bc == 1:
        for i in range(N):
            T[0,i] = 1
            T[1,i] = (-1)**i
    return T

def DiffCheb(M,T):

    # differentiates using Chebshev recurrence
    # assumes grid is [-1,1] - multiply by appropriate scaling afterwards

    #CHEBYSHEV CONSTANTS
    c = np.ones(M)
    c[0]=2
    
    DT = np.zeros(M)
    DT[M-2] = 2*M*T[M-1]

    for j in range(M-3,-1,-1):
        DT[j] = 1/c[j]*(DT[j+2] + 2*(j+1)*T[j+1])
    
    return DT
