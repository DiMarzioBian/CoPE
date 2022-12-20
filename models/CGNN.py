import math
import numpy as np
from scipy import special

import torch
from torch import nn


class InvNet(nn.Module):
    def __init__(self, order):
        super().__init__()
        self.order = order

    def forward(self, adj_obs, x, alpha=1.):
        z_stack = [x]
        z = x
        for _ in range(self.order):
            z = alpha * (adj_obs @ z)
            z_stack.append(z)
        return torch.stack(z_stack, 0).sum(0)


class ExpNet(nn.Module):
    def __init__(self, order):
        super().__init__()
        self.order = order

        # compute bessel coefficients
        c_bessel = 2 * (special.jv(np.arange(order + 1), 0 - 1j) * (0 + 1j) ** np.arange(order + 1)).real
        c_bessel[0] /= 2
        self.register_buffer('c_bessel', torch.tensor(c_bessel, dtype=torch.float32).reshape(-1, 1, 1))

    def forward(self, adj_obs, x, alpha):
        # Recursion of 1st kind Chebyshev polynomials
        pp_state = x
        p_state = alpha * (adj_obs @ x)
        zs = [pp_state, p_state]
        for _ in range(self.order - 1):
            n_state = 2 * alpha * (adj_obs @ p_state) - pp_state
            zs.append(n_state)
            pp_state, p_state = p_state, n_state
        return (torch.stack(zs, 0) * self.c_bessel).sum(0)


class CGNN(nn.Module):
    """ CGNN """
    def __init__(self, args):
        super().__init__()
        self.ts_max = args.ts_max
        self.d_latent = args.d_latent
        self.n_node = args.n_user + args.n_item

        self.exp_net = ExpNet(args.exp_order)
        self.inv_net = InvNet(args.inv_order)

        mat_I = torch.sparse_coo_tensor(indices=torch.arange(self.n_node).unsqueeze(0).repeat(2, 1),
                                        values=torch.ones(self.n_node), size=[self.n_node, self.n_node])
        self.register_buffer('mat_I', mat_I)

        self.alpha = nn.Parameter(torch.ones(self.n_node) * 3)

    def forward(self, x_embed, x_t, t_diff, adj_obs):
        alpha = torch.sigmoid(self.alpha).unsqueeze(1)

        # time shift
        z = torch.cat([x_embed, x_t], dim=1) * t_diff.neg().exp()

        # approximate matrix exponential
        if t_diff > 0:
            z = self.exp_net(adj_obs, z, alpha)

        # approximate matrix inverse
        x_embed_exp, x_t_exp = torch.split(z, self.d_latent, dim=1)
        x_embed_inv = self.inv_net(adj_obs, x_embed - x_embed_exp, alpha)

        return x_embed_inv + x_t_exp
