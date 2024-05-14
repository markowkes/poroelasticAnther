# Poroelastic beam
using Printf
using Plots

function solver()
    # Parameters
    L=1
    η=1
    λ=1
    tFinal = 10

    # Discretization inputs
    Nx=60
    dt=1e-3

    # Create grid
    x = LinRange(0, L, Nx)  
    dx = x[2] - x[1]                

    # Create A operator
    A = createAop(x,dt,η,λ)

    # Initial condition
    t = 0
    M = zeros(Nx)
    w = zeros(Nx) 

    # Preallocate
    R = zeros(2Nx)

    # Loop over time 
    Nt = floor(tFinal/dt)
    for n = 1:Nt
        
        # Update time 
        t += dt

        # Compute loading - Ditribute L over center grid cell(s)
        qn = zeros(Nx)
        # if Nx % 2==0 # even Nx => split force over 2 center cells 
        #     qn[Int(Nx/2)  ] = L/(2*dx)
        #     qn[Int(Nx/2)+1] = L/(2*dx)
        # else
        #     qn[Int(floor(Nx/2))+1] = L/dx 
        # end
        f=20
        qn = sin(t*f)*sin.(2*pi*x/1)

        # Compute RHS using tⁿ quantities
        for i = 3:Nx-2 # Interior points for w
            R[i] = -qn[i]
        end
        for i = 2:Nx-1 # Interior points for M
            R[i+Nx] = (
                M[i] +
                η   *(w[i-1] - 2w[i] + w[i+1])/dx^2 +
                dt/2*(M[i-1] - 2M[i] + M[i+1])/dx^2
            )
        end

        # # Override R to enforce M=0
        # R[Nx+1:2Nx] .= 0

        # Solve system of equations for solution at tⁿ⁺¹
        sol = A\R 
        w = sol[1:Nx]
        M = sol[Nx+1:2Nx]

        # Plot solution
        if n % 100 == 1
            p1 = plot(x,w, label="w(x)")
            p2 = plot(x,M, label="M(x)")
            plt = plot(p1,p2)
            display(plt)
        end

        
    end

    return nothing
end

function createAop(x,dt,η,λ)
    dx = x[2] - x[1]
    Nx = length(x)
    A = zeros(2Nx,2Nx)
    
    ## w equations 
    ## --------------------------------
    # Loop over interior points for w
    for i = 3:Nx-2
        A[i,i     ] =  6/dx^4 # d⁴w/dx⁴
        A[i,i-1   ] = -4/dx^4
        A[i,i+1   ] = -4/dx^4
        A[i,i-2   ] =  1/dx^4
        A[i,i+2   ] =  1/dx^4
        A[i,i  +Nx] =  2η/dx^2 # η⋅d²M/dx²
        A[i,i-1+Nx] = -1η/dx^2 
        A[i,i+1+Nx] = -1η/dx^2 
    end
    # Boundary conditions @ x=0
    A[1,1] =  1      # w = 0
    A[2,1] =  2/dx^2 # d²w/dx² = 0
    A[2,2] = -5/dx^2 
    A[2,3] =  4/dx^2 
    A[2,4] = -1/dx^2 
    # Boundary conditions @ x=L
    A[Nx  ,Nx  ] =  1      # w = 0
    A[Nx-1,Nx  ] =  2/dx^2 # d²w/dx² = 0
    A[Nx-1,Nx-1] = -5/dx^2 
    A[Nx-1,Nx-2] =  4/dx^2 
    A[Nx-1,Nx-3] = -1/dx^2 

    ## M equations 
    ## --------------------------------
    # Loop over interior points for M
    for i = 2:Nx-1
        A[Nx+i,i  +Nx] = 1 + dt/dx^2 # dM/dt & d²M/dx²
        A[Nx+i,i-1+Nx] = -dt/(2*dx^2)
        A[Nx+i,i+1+Nx] = -dt/(2*dx^2)
        A[Nx+i,i     ] = -2λ/dx^2  # d(d²w/dx²)/dt = 0
        A[Nx+i,i-1   ] =   λ/dx^2
        A[Nx+i,i+1   ] =   λ/dx^2
    end
    # Boundary conditions @ x=0
    # A[Nx+1,Nx+1] =  4/(2dx) # dM/dx(x=0) = 0
    # A[Nx+1,Nx+2] = -3/(2dx)
    # A[Nx+1,Nx+3] = -1/(2dx)
    A[Nx+1,Nx+1] = 1  # M(x=0) = 0
    # Boundary condition @ x=L 
    A[2Nx,2Nx] = 1 # M = 0

    # # Make M zero 
    # for i=Nx+1:2Nx 
    #     A[i,:] .= 0
    #     A[i,i] = 1
    # end

    return A
end

# Test creating A operator
# A = createAop(LinRange(0,1,6),1e-3,1,1)
# display(A)

solver()