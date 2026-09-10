import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# PINN for the 1 dimensional heat equation
# u_t = u_xx, x in [0,1], t in [0, T]
# IC: u(x,0) = sin(pi x) 
# Homogeneous dirichlet BC: u(0,t) = u(1,t) = 0

# (t,x) = (N,2)

device = torch.device('mps')

class PINN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 64), 
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, D):
        # D: (N,2)
        u = self.fc(D)
        return u

# domain: [0,T] x [0,1]
def train(model: PINN, optimizer: torch.optim.Optimizer, D_bc: torch.Tensor, D_ic: torch.Tensor, D_colloc: torch.Tensor, 
          u_bc: torch.Tensor, u_ic: torch.Tensor, u_colloc: torch.Tensor, epochs: int):    
    model.train()
    losses = []
    for e in range(epochs):
        #u(t,x)
        u_hat_bc = model(D_bc)
        u_hat_ic = model(D_ic)
        u_hat_colloc = model(D_colloc)
        du = torch.autograd.grad(u_hat_colloc, D_colloc, 
                                 grad_outputs=torch.ones_like(u_hat_colloc), create_graph=True)[0]
        du_dt = du[:, 0:1]
        du_dx = du[:, 1:2]
        d2u = torch.autograd.grad(du_dx, D_colloc, 
                                  grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
        du_dxx = d2u[:, 1:2] 
        
        residual = du_dt - du_dxx
        boundary_loss = F.mse_loss(u_hat_bc, u_bc)
        init_cond_loss = F.mse_loss(u_hat_ic, u_ic)
        colloc_loss = F.mse_loss(residual, u_colloc)
        loss = boundary_loss + init_cond_loss + colloc_loss
        losses.append(loss.item())
        print(f'Epoch: {e+1} | Total Loss: {loss.item():.5f} | \
              Boundary Loss: {boundary_loss.item():.5f} | \
              Initial Condition Loss: {init_cond_loss.item():.5f} \
              Collocation Loss: {colloc_loss.item():.5f}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return losses

    
    
    
    
    
    
    
    
    
    
    