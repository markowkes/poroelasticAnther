#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:18:20 2024

@author: mowkes
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


def main():

    # Parameters
    L=1
    η=1
    λ=1
    Load = 1
    period = 1
    tFinal = 10

    # Discretization inputs
    Nx=60
    dt=1e-3
    
    # Create grid
    x = np.linspace(0, L, num=Nx)  
    dx = x[1] - x[0]                
    
    # Create A operator
    A = createAop(x,dt,η,λ)
 
    # for i in range(2*Nx):
    #     for j in range(2*Nx):
    #         print("{:8.2f}".format(A[i,j]), end =" ")
    #     print()
    
    # Initial condition
    t = 0
    M = np.zeros(Nx)
    w = np.zeros(Nx) 

    # Preallocate
    R = np.zeros(2*Nx)
    
    # ## Test solution of 2.46b
    # w = computeW(A, x, Mp, eta, qn)
    # plt.plot(x,w)
    # # Pressureize
    # #Mp -= np.sin(np.pi*x/L)
    # Mp[int(Nx/2):] = -1
    # w = computeW(A, x, Mp, eta, qn)
    # plt.plot(x,w,'--')
    # plt.show()
    # return
    # ##
    
    # Loop over time
    Nt = int(tFinal/dt)
    for n in range(Nt):
        
        # Update time 
        t += dt
        
        # Compute loading - Ditribute L over center grid cell(s)
        qn = np.zeros(Nx)
        if np.remainder(Nx,2)==0: # even Nx => split force over 2 center cells 
            qn[int(Nx/2)-1] = Load*np.sin(2*np.pi*t/period)/(2*dx)
            qn[int(Nx/2)  ] = Load*np.sin(2*np.pi*t/period)/(2*dx)
        else:
            qn[int(np.floor(Nx/2))] = Load*np.sin(2*np.pi*t/period)/dx 
        
        # Compute RHS using tⁿ quantities
        for i in range(2,Nx-2): # Interior points for w
            R[i] = -qn[i]
        for i in range(1,Nx-1): # Interior points for M
            R[i+Nx] = (                                \
                M[i] +                                 \
                η   *(w[i-1] - 2*w[i] + w[i+1])/dx**2 + \
                dt/2*(M[i-1] - 2*M[i] + M[i+1])/dx**2
            )

        # Solve system of equations for solution at tⁿ⁺¹
        sol = np.linalg.solve(A,R) 
        w = sol[0:Nx]
        M = sol[Nx:2*Nx]
            
        if np.remainder(n,100)==1:
            outputs(t,x,w,M)
        
def createAop(x,dt,η,λ):
    dx = x[1] - x[0]
    Nx = np.size(x)
    A = np.zeros((2*Nx,2*Nx))
    
    ## w equations 
    ## --------------------------------
    # Loop over interior points for w
    for i in range(2,Nx-2):
        A[i,i     ] =  6/dx**4 # d⁴w/dx⁴
        A[i,i-1   ] = -4/dx**4
        A[i,i+1   ] = -4/dx**4
        A[i,i-2   ] =  1/dx**4
        A[i,i+2   ] =  1/dx**4
        A[i,i  +Nx] =  2*η/dx**2 # η⋅d²M/dx²
        A[i,i-1+Nx] = -1*η/dx**2 
        A[i,i+1+Nx] = -1*η/dx**2 
    # Boundary conditions @ x=0
    A[0,0] =  1      # w = 0
    A[1,0] =  2/dx**2 # d²w/dx² = 0
    A[1,1] = -5/dx**2 
    A[1,2] =  4/dx**2 
    A[1,3] = -1/dx**2 
    # Boundary conditions @ x=L
    A[Nx-1,Nx-1] =  1      # w = 0
    A[Nx-2,Nx-1] =  2/dx**2 # d²w/dx² = 0
    A[Nx-2,Nx-2] = -5/dx**2 
    A[Nx-2,Nx-3] =  4/dx**2 
    A[Nx-2,Nx-4] = -1/dx**2 

    ## M equations 
    ## --------------------------------
    # Loop over interior points for M
    for i in range(1,Nx-1):
        A[Nx+i,i  +Nx] = 1 + dt/dx**2 # dM/dt & d²M/dx²
        A[Nx+i,i-1+Nx] = -dt/(2*dx**2)
        A[Nx+i,i+1+Nx] = -dt/(2*dx**2)
        A[Nx+i,i     ] = -2*λ/dx**2  # d(d²w/dx²)/dt = 0
        A[Nx+i,i-1   ] =    λ/dx**2
        A[Nx+i,i+1   ] =    λ/dx**2
    # Boundary conditions @ x=0
    A[Nx,Nx  ] =  4/(2*dx) # dM/dx = 0
    A[Nx,Nx-1] = -3/(2*dx)
    A[Nx,Nx+1] = -1/(2*dx)
    # Boundary condition @ x=L 
    A[2*Nx-1,2*Nx-1] = 1 # M = 0

    return A

def outputs(t,x,w,Mp):
    
    # Analytic solution
    #(w_ana,Mp_ana) = analytic(t)
    
    # Create a figure containing a two rows and 1 column of axes
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(5, 2.7),dpi=300)

    # Add to avoid overlapping
    fig.tight_layout()

    # Plot on first subplot
    ax1.plot(x,w)   # Plot y1(x)
    ax1.set_xlabel("x")
    ax1.set_ylabel("w(x)")
    ax1.set_title('Time = {0:2f}'.format(t))
    #ax1.plot(x,w_ana,'--')

    # Plot on second subplot
    ax2.plot(x,Mp)   # Plot y2(x)
    ax2.set_xlabel("x")
    ax2.set_ylabel("Mp(x)")
    #ax2.plot(x,Mp_ana,'--')

    plt.savefig("subplots.pdf", bbox_inches="tight")
    plt.show()


def analytic(t):
    Nx = 50
    Lx = 1
    eta = 1
    lam = 1
    x = np.linspace(0, Lx, num=Nx) 
    
    # Section 3.1 Analytic solution
    
    # Loading 
    # qn = q(x,t) = sin(pi*x)
    # Eq. 3.4a,b 
    b1 = 1 # all other bn's = 0
    
    # Let w (t=0) = 0  Eq. 3.6a
    # Let Mp(t=0) = 0  Eq. 3.6b
    wn0 = 0 # for all n
    mn0 = 0 # for all n

    # Solve Eq. 3.7a for mn(t)
    def m(n,t,mn0,wn0,bn,eta,lam):
        tp = sp.symbols('tp')
        mn =  mn0 - lam/((1+lam*eta)*n**2*np.pi**2)* \
            sp.integrate(bn*np.exp(n**2*np.pi**2/(1+lam*eta))*tp,(tp,0,t))
        return mn
    
    # Compute m1 (all others are zero with loading) Eq. 3.7a
    m1 = m(1,t,mn0,wn0,b1,eta,lam)
    
    # Compute w1 (all others are zero with loading) Eq. 3.7b
    w1 = -b1/(1**4*np.pi**4) - eta/(1**2*np.pi**2)*m1
    
    # Compute w with Eq. 3.3a
    w  = w1*np.sin(1*np.pi*x)
    Mp = m1*np.sin(1*np.pi*x)
    
    return (w,Mp)

if __name__ == '__main__':
    main()
    


    