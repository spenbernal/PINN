import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pinn as p
import fd
from scipy.interpolate import RegularGridInterpolator

device = torch.device('mps')

# Solving the Heat equation with homogeneous dirichlet boundary conditions
# PDE: u_t = u_xx, defined on [0,1] x [0,1]
# Initial Condition: u(x,0) = sin(pi*x), for 0 <= x <= 1
# Boundary Conditions: u(0, t) = u(1, t) = 0, for 0 <= t <= 1


if __name__ == '__main__':
    # results to save plots
    pd = Path().parent
    plots = pd / 'plots'
    plots.mkdir(exist_ok= True)
    
    plot_file = plots / 'contours.png'
    
    
    # Time-Space Domain: [0,1) x [0,1)
    T = 1
    X = 1
    # Start PINN section
    N_b = 5000
    N_i = 5000
    N_colloc = 10000
    # Random fixed collocation points (interior points)
    X_colloc = X * torch.rand(size=(N_colloc, 1))
    T_colloc = T * torch.rand(size=(N_colloc, 1))
    D_colloc = torch.concatenate([T_colloc, X_colloc], dim= 1)
    # Boundary points
    X_b_left = torch.zeros((N_b, 1))
    X_b_right = torch.ones((N_b, 1))
    X_b = torch.cat([X_b_left, X_b_right], dim= 0)
    t_b = torch.linspace(0, T, N_b)
    T_b = t_b.repeat(2, 1).reshape(-1, 1)
    D_b = torch.concatenate([T_b, X_b], dim= 1)
    # Initial Condition points
    X_i = torch.linspace(0, 1, N_i).unsqueeze(dim= 1)
    T_i = torch.zeros_like(X_i)
    D_i = torch.concatenate([T_i, X_i], dim= 1) # N_ic, 2
    # boundary values, initial condition values, and rhs of pde
    u_ic = torch.sin(torch.pi * X_i) # N_ic, 1
    u_bc = torch.zeros((2*N_b, 1)) # 2N_b,1 
    u_colloc = torch.zeros((N_colloc, 1)) # N_colloc,1
    
    # move model and data to device for compute
    D_colloc = D_colloc.to(device)
    D_b = D_b.to(device)
    D_i = D_i.to(device)
    D_colloc.requires_grad_(True)
    
    u_ic = u_ic.to(device)
    u_bc = u_bc.to(device)
    u_colloc = u_colloc.to(device)
    
    model = p.PINN().to(device)
    
    epochs = 1000
    eta = 0.001
    optimizer = torch.optim.Adam(model.parameters(), eta)
    
    pinn_losses = p.train(model, optimizer, D_b, D_i, D_colloc, u_bc, u_ic, u_colloc, epochs)
    # End of PINN section
    
    # Start Finite Difference Section
    finitediff = fd.FiniteDifferenceHeatEq()
    
    dx = .01 # mesh size for spatial variable
    s = .5 # mesh factor, needs to be <= .5 to ensure stability of solution
    dt = s*dx**2 # mesh size for time variable

    x0 = 0 # spatial starting point
    xL = 1 # spatial ending point

    t0 = 0 # time starting point
    tL = 1 # time starting point

    x_fd = np.arange(x0, xL+dx, dx) # space array
    t_fd = np.arange(t0, tL+dt, dt) #time array

    init_cond = np.sin(np.pi * x_fd) #initial condition
    
    # dirichlet condition enforced inside function
    u_fd = finitediff.solve_dirichlet(x_fd, t_fd, s, init_cond)
    
    # -------------------------------------------
    # -----------------Inference-----------------
    # -------------------------------------------
    
    # evaluate PINN 
    model.eval()
    
    N_t = 500
    N_x = 500
    
    x_test = torch.linspace(0, X, N_x) 
    t_test = torch.linspace(0, T, N_t)
    
    T_test, X_test = torch.meshgrid(t_test, x_test, indexing='ij') #100,100
    D_test = torch.stack([T_test.reshape(-1), X_test.reshape(-1)], dim= 1).to(device)
    with torch.no_grad():
        u_theta = model(D_test) #10000,1
        
    u_theta = u_theta.reshape(N_t, N_x) #100,100
    u_theta = u_theta.cpu()
    u_analytic = torch.exp(-torch.pow(torch.pi, torch.tensor(2)) * T_test) * torch.sin(torch.pi * X_test)
    
    fd_interpolator = RegularGridInterpolator((t_fd, x_fd), u_fd)
    u_fd_test = fd_interpolator(D_test.detach().cpu().numpy())
    u_fd_test = u_fd_test.reshape(N_t, N_x)
    
    print('---------- Test MSE ----------')
    fd_mse = np.mean(np.square(u_fd_test - u_analytic.detach().numpy()))
    pinn_mse = torch.mean(torch.square(u_theta - u_analytic))
    print(f'Finite Difference MSE: {fd_mse} | PINN MSE: {pinn_mse}')
    
    
    fig, ax = plt.subplots(1, 3, figsize=(10,4))
    fig.suptitle('Heat Equation $u_t = u_{xx}$')
    fig.supxlabel('$X$ (m)')
    fig.supylabel('T (s)')
    ax[0].imshow(u_fd, cmap= 'magma', origin='lower', extent=[0, 1, 0, 1])
    ax[0].set_title('Finite Difference Solution')

    ax[1].imshow(u_theta, cmap= 'magma', origin='lower', extent=[0, 1, 0, 1])
    ax[1].set_title('PINN Solution')
    ax[1].get_yaxis().set_visible(False)
    
    im = ax[2].imshow(u_analytic, cmap= 'magma', origin='lower', extent=[0, 1, 0, 1])
    ax[2].get_yaxis().set_visible(False)
    ax[2].set_title('Analytic Solution')
    
    cbar = fig.colorbar(im, ax=ax, shrink= 0.6)
    cbar.set_label('Temperature $(u(t,x))$')
    plt.savefig(plot_file)
    
    loss_plot = plots / 'pinn_mse.png'
    plt.figure(figsize=(8,6))
    plot_loss = np.array(pinn_losses)
    plt.plot(plot_loss, label='mse')
    plt.legend()
    plt.title('MSE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.savefig(loss_plot)
    