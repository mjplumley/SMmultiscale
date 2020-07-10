#!/usr/bin/env python
# coding: utf-8

# python to run the single mode dynamo code 
# multiscale version

import numpy as np
import scipy 
import json
from SMdynamo_tools import *

#np.fft.fft()  <- usage for the fft



# RUN PARAMETERS
M = 5            # Chebyshev points
dt = 1e-2         # fast time step size
imax = 5e1          # number of fast time steps
slowfac = 10       # number of fast steps per slow step (only needed for the multiscale )

# FLOW PARAMETERS
Ra = 20        # Rayleigh number
Pr = 1         # Prandtl number
Ek = 1     # Ekman number
k = 1.3048     # wavenumber
Pm = 1         # Magnetic Prandtl number

# MAGNETIC BOUNDARY CONDITIONS
# BC = 0 -> PERFECTLY CONDUCTING
# BC = 1 -> PERFECTLY INSULATING
BC = 1

# TIME-STEPPING CONSTANTS
# CRANK-NICOLSON: alpha=2*beta
# alpha > 0, alpha >= beta
alpha=1
beta=1/2

# PRECISION FOR ZERO (VALUES LESS THAN THIS WILL BE SET TO ZERO)
# USEFUL FOR CREATING SPARSE MATRICES
eps = 1e-15;

# DEGREEE OF ACCURACY ON COEFFICIENT FUNCTION
accuracy = 1e-15;

# Using the squares horizontal planform
# SET PLANFORM: squares=0,hexagons=1,triangles=2,patchwork=3
# CALCULATES AVERAGES BASED ON PLANFORM
planform = 0
if planform == 0:
    c1 = 0.5
    c2 = 0.5
elif planform == 1:
    c1 = 0.75
    c2 = 0.75
elif planform == 2:
    c1 = 0.75
    c2 = 0.75
elif planform == 3:
    c1 = 3/4
    c2 = 1/4

# COEFFICIENTS FOR ALPHA TERM
alpha11=2*Pm*c1
alpha22=2*Pm*c2
# COEFFICIENTS FOR LORENTZ TERMS
LorX = -Pm*c1
LorY = -Pm*c2
# COEFFICIENTS FOR TEMPERATURE TERMS
cT1 = 1/(1 + (alpha-beta)*dt*k**2/Pr)
cT2 = (1 - beta*dt*k**2/Pr)
cT3 = 1/(1 + 0.5*dt*k**2/Pr)
cT4 = (1 - 0.5*dt*k**2/Pr)

# DEFINE EPSILON EXPONENTS
ep=Ek**(1/3)
ep2=ep**2
ep32=ep**(3/2)

# DEFINE CHEBYSHEV GRID
Z = cheb_grid(M)
Z = 0.5*(1-Z)


# MAKE THE QUASI_INVERSE MATRICES
QI = intg(M)
QI2 = intg2(M)
Iz  = identity(M,0)
Iz1 = identity(M,1)
Iz2 = identity(M,2)

# MATRIX FOR MEAN TEMPERATURE EQUATION
AT = 4*Iz2 + boundaries_T(M) # no timestepping on MT

# MATRIX FOR VORTICITY/MOMENTUM EQUATIONS
# COUPLED SOLVE
A11 = ((1 + (alpha-beta)*dt*k**2)*QI)
A12 = -2*(alpha-beta)*dt/k**2*Iz1 + boundaries_w_top(M)
A21 = (-2*(alpha-beta)*dt*Iz1)
A22 = (1 + (alpha-beta)*dt*k**2)*QI + boundaries_w_bottom(M)

AZ1 = np.concatenate((A11, A12), axis = 1)
AZ2 = np.concatenate((A21, A22), axis = 1)
AZ1 = np.concatenate((AZ1, AZ2))

A11 = ((1 + 0.5*dt*k**2)*QI)
A12 = -dt/k**2*Iz1 + boundaries_w_top(M)
A21 = (-dt*Iz1)
A22 = (1 + 0.5*dt*k**2)*QI + boundaries_w_bottom(M)

AZ22 = np.concatenate((A11, A12), axis = 1)
AZ2 = np.concatenate((A21, A22), axis = 1)
AZ2 = np.concatenate((AZ22, AZ2))

BZ1 = ((1-beta*dt*k**2)*QI)
BZ2 = ((1-0.5*dt*k**2)*QI)
BZ = np.concatenate((BZ1, BZ2))

# Matrices for Induction equations
AB1 = QI2 - 4*(alpha-beta)*(ep32*dt/Pm)*Iz2 + boundaries_B(M,BC)
AB2 = QI2 - 4*(0.5*ep32*dt/Pm)*Iz2 + boundaries_B(M,BC)
BB1 = QI2 + 4*beta*(ep32*dt/Pm)*Iz2
BB2 = QI2 + 4*(0.5*ep32*dt/Pm)*Iz2


# INITIALIZE 
Nu = np.zeros( (int(imax)+1,1) ) 
Wrms = np.zeros( (int(imax)+1,1 ))
Bxrms = np.zeros( (int(imax)+1,1 ))
time = np.zeros( (int(imax)+1,1 ))


# USE LINEAR EIGENFUNCTIONS AS INITIAL CONDITIONS WITH NOISE
amp=0;
pi = np.pi
W_old = np.sin(pi*Z) + amp*np.random.randn(M)
W_old = phys2cheb(W_old)
DTb = -1*np.ones(M) + amp*np.random.randn(M)
T_old = (Pr/k**2)*np.sin(pi*Z) + amp*np.random.randn(M)
T_old = phys2cheb(T_old)
Psi_old = -pi/k**4*np.cos(pi*Z) + amp*np.random.randn(M)
Psi_old = phys2cheb(Psi_old)
Tb_old = 1-Z + amp*amp*np.random.randn(M)
Tb_old = phys2cheb(Tb_old)

if BC == 0:
    Bx_old = Psi_old
    By_old = Psi_old
elif BC == 1:
    Bx_old = W_old
    By_old = W_old

    
PWx_old = np.zeros(M)
PWy_old = np.zeros(M)
PWx = np.zeros(M)
PWy = np.zeros(M)

ilast = 0
ifac = 1

# THE TIME STEPPING LOOP

for i in range(0,int(imax)+1):
    t = i*dt + ilast*dt
    
    # COMPUTE NONLINEAR TERMS
    C_old = phys2cheb(cheb2phys(W_old)*DTb )
    F_old =  phys2cheb(cheb2phys(W_old)*cheb2phys(T_old)) 
   
    # NONLINEAR TERMS FOR Bx, By EQUATIONS 
    PWx_temp = phys2cheb(cheb2phys(Psi_old)*cheb2phys(W_old)*cheb2phys(Bx_old))
    PWy_temp = phys2cheb(cheb2phys(Psi_old)*cheb2phys(W_old)*cheb2phys(By_old))
    PWx_old = PWx_old + PWx_temp
    PWy_old = PWy_old + PWy_temp
    
    Psi_LOR = LorX*phys2cheb(((cheb2phys(Bx_old))**2)*cheb2phys(Psi_old)) + LorY*phys2cheb(((cheb2phys(By_old))**2)*cheb2phys(Psi_old))
    W_LOR = LorX*phys2cheb(((cheb2phys(Bx_old))**2)*cheb2phys(W_old)) + LorY*phys2cheb(((cheb2phys(By_old))**2)*cheb2phys(W_old))
    
    
    ######################
    # FIRST SOLVING STAGE
    ######################
    
    # FLUCTUATING HEAT
    T = cT1*(cT2*T_old - alpha*dt*C_old)   
    
    # VORTICITY
    b1 = np.dot(BZ1 ,Psi_old) + 2*beta*dt/k**2*np.dot( Iz1, W_old.T) + alpha*dt*np.dot( QI, Psi_LOR)
    b2 = np.dot(BZ1,W_old) + 2*beta*dt*np.dot(Iz1,Psi_old.T) + dt*Ra/Pr*(beta*np.dot(QI,T_old.T) + (alpha-beta)*np.dot(QI,T.T)) + alpha*dt*np.dot(QI,W_LOR)
    bZ = np.concatenate( (b1, b2) )  
    Psi_W = np.linalg.solve(AZ1, bZ.T)
    Psi = Psi_W[0:M]
    W = Psi_W[M:2*M]
   
  
    # MEAN HEAT
    bT = -2*Pr*Iz2@QI@F_old
    bT[0] = 1
    bT[1] = 0
    Tb = np.linalg.solve(AT,bT)
    
    DTb = DiffCheb(M,Tb)
    DTb = -2*cheb2phys(DTb)
    
        
    # MAGNETIC STEPS
    if i%slowfac ==0:
        PWx_old = PWx_old/slowfac
        PWy_old = PWy_old/slowfac
        
        b1 = np.dot(BB1,Bx_old) + -2*Pm*c2*2*Iz2@QI@PWy_old*alpha*dt*ep32
        Bx = np.linalg.solve(AB1,b1)
    
        b1 = np.dot(BB1,By_old) + 2*Pm*c1*2*Iz2@QI@PWx_old*alpha*dt*ep32
        By = np.linalg.solve(AB1,b1)
    
        
    ######################
    # SECOND SOLVING STAGE
    ######################
    
    # FLUCTUATING HEAT
    C = phys2cheb(cheb2phys(W)*DTb)
    T = cT3*(cT4*T_old - (0.5*(2*alpha-1)/alpha)*dt*C_old - 0.5*dt/alpha*C)
    
    # VORTICITY
    Psi_LOR_1 = LorX*phys2cheb(((cheb2phys(Bx))**2)*cheb2phys(Psi)) + LorY*phys2cheb(((cheb2phys(By))**2)*cheb2phys(Psi))
    W_LOR_1 = LorX*phys2cheb(((cheb2phys(Bx))**2)*cheb2phys(W)) + LorY*phys2cheb(((cheb2phys(By))**2)*cheb2phys(W))
    bxx = BZ2@Psi_old + dt/k**2*Iz1@W_old + 0.5*dt*(2*alpha-1)/alpha*QI@Psi_LOR + 0.5*dt/alpha*QI@Psi_LOR_1
    byy = BZ2@W_old + dt*Iz1@Psi_old + 0.5*dt*Ra/Pr*QI@(T_old + T) +0.5*dt*(2*alpha-1)/alpha*QI@W_LOR + 0.5*dt/alpha*QI@W_LOR_1  
    bZ = np.concatenate( (bxx,byy) )
    Psi_W = np.linalg.solve(AZ2, bZ)
    Psi = Psi_W[0:M]
    W = Psi_W[M:2*M]
    
    # MEAN HEAT
    F  = phys2cheb(cheb2phys(W)*cheb2phys(T));
    bT = -2*Pr*Iz2@QI@F
    bT[0] = 1;
    bT[1] = 0; 
    Tb = np.linalg.solve(AT,bT)
    
    DTb = DiffCheb(M,Tb)
    DTb = -2*cheb2phys(DTb)
    
    # MAGNETIC 
    PWy_temp = phys2cheb(cheb2phys(Psi)*cheb2phys(W)*cheb2phys(By))
    PWx_temp = phys2cheb(cheb2phys(Psi)*cheb2phys(W)*cheb2phys(Bx))
    PWy = PWy + PWy_temp
    PWx = PWx + PWx_temp
    
    if i%slowfac == 0:
        PWy = PWy/slowfac
        PWx = PWx/slowfac
        
        b1 = BB2@Bx_old + -2*Pm*c2*2*Iz2@QI@PWy_old*(2*alpha-1)*dt*ep32/(2*alpha)+ -2*Pm*c2*2*Iz2@QI@PWy*dt*ep32/(2*alpha)
        Bx = np.linalg.solve(AB2,b1)

        b1 = BB2@By_old + 2*Pm*c1*2*Iz2@QI@PWx_old*(2*alpha-1)*dt*ep32/(2*alpha) + 2*Pm*c1*2*Iz2@QI@PWx*dt*ep32/(2*alpha)
        By = np.linalg.solve(AB2,b1)
        
        # Reset the values of the averages
        PWx_old = np.zeros(M)
        PWy_old = np.zeros(M)
        PWx = np.zeros(M)
        PWy = np.zeros(M)

    ######################
    # DIAGNOSTICS
    ######################
    Nu[i] = -DTb[0]
    if abs(DTb[0]) > 1000:
        print('Nu is too large. Stopped at ' , t)
        break
    Wrms[i] = np.linalg.norm(cheb2phys(W))
    Bxrms[i] = np.linalg.norm(cheb2phys(Bx))
    time[i] = t
    
    
    # RESET VECTORS
    W_old = W
    Psi_old = Psi
    T_old = T
    Tb_old = Tb       
    Bx_old = Bx
    By_old = By
    ifac = ifac + 1

# PLOT RESULTS
#plt.plot(time,Nu)
#plt.title('Nu');

#plt.plot(time,Wrms,label='RMS(W)')
#plt.plot(time,Bxrms, label='RMS(Bx)')
#plt.title('RMS(W), RMS(Bx)');
#plt.legend(loc="upper right");
    
# WRITE RESULTS TO TEXT FILES
time_Nu = np.hstack((time,Nu))
np.savetxt("Nusselt.txt", time_Nu)

time_W = np.hstack((time,Wrms))
np.savetxt("W_rms.txt", time_W)

time_B = np.hstack((time,Bxrms))
np.savetxt("Bx_rms.txt", time_B)

