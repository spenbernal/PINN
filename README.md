# Physics Informed Neural Network for the 1D diffusion equation
This project explores how to implement a PINN using pytorch, and using it on the 1D diffusion equation. 

$$
u_t = u_{xx}, \qquad 0 \leq x \leq 1, \quad 0 \leq t \leq 1
$$ 

subject to initial condition

$$
u(x,0) = \sin(\pi x) 
$$

with homogeneous dirichlet boundary conditions

$$
u(0,t) = u(1,t) = 0
$$

The PINN is tested against the **finite difference method** from one of my previous projects, and the numerical solutions are tested against the analytic solution.