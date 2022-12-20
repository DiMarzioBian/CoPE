import torch
from torch import nn
from torch.nn import functional as F

from models.CGNN import CGNN


class CoPE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.n_user = args.n_user
        self.n_item = args.n_item
        self.d_latent = args.d_latent
        self.n_neg_sample = args.n_neg_sample
        self.idx_pad = args.idx_pad

        # model architecture
        self.embeds_u = nn.Embedding(self.n_user, self.d_latent)
        self.embeds_i = nn.Embedding(self.n_item, self.d_latent)
        self.embeds_u.weight.data.normal_().fmod_(2).mul_(0.01).add_(0)  # truncated normalization
        self.embeds_i.weight.data.normal_().fmod_(2).mul_(0.01).add_(0)

        self.propagator = PropagateUnit(args)
        self.updater = DiscUpdateUnit(args)

        self.predictor = PredictUnit(args)

    def get_init_states(self):
        return self.embeds_u.weight.clone().detach(), self.embeds_i.weight.clone().detach()

    def forward(self, batch, xu_in, xi_in):
        ts_diff, adj_obs, adj_ins_i2u, adj_ins_u2i, tgt_u, tgt_i, tgt_u_neg, \
            tgt_i_neg = batch

        # propagate during non-event
        xu_t_minus, xi_t_minus = self.propagator(adj_obs, ts_diff, xu_in, xi_in,
                                                 self.embeds_u.weight, self.embeds_i.weight)

        # user states
        states_u_pos = F.embedding(tgt_u, xu_t_minus)
        states_u_neg = F.embedding(tgt_u_neg, xu_t_minus)

        # item states
        states_i_pos = F.embedding(tgt_i, xi_t_minus)
        states_i_neg = F.embedding(tgt_i_neg, xi_t_minus)

        xu_pos = torch.cat([states_u_pos, self.embeds_u(tgt_u)], dim=-1)
        xu_neg = torch.cat([states_u_neg, self.embeds_u(tgt_u_neg)], dim=-1)

        xi_pos = torch.cat([states_i_pos, self.embeds_i(tgt_i)], dim=-1)
        xi_neg = torch.cat([states_i_neg, self.embeds_i(tgt_i_neg)], dim=-1)

        loss_rec = self.cal_loss(xu_pos, xi_pos, xu_neg, xi_neg)

        # update by instant graph
        xi_enc = torch.cat([xi_t_minus, self.embeds_i.weight], dim=-1).unsqueeze(1)
        xu_t_plus, xi_t_plus, loss_jump = self.updater(xu_t_minus, xi_t_minus, adj_ins_i2u, adj_ins_u2i)

        return loss_rec, loss_jump, xu_t_plus, xi_t_plus, xu_pos, xi_enc

    def cal_loss(self, xu_pos, xi_pos, xu_neg, xi_neg):
        pos_scores = self.predictor(xu_pos, xi_pos)

        neg_scores_u = self.predictor(xu_pos, xi_neg)
        neg_scores_i = self.predictor(xu_neg, xi_pos)

        scores = torch.cat([pos_scores, neg_scores_u, neg_scores_i], dim=-1)
        loss = -F.log_softmax(scores, 1)[:, 0].mean()
        return loss


class PropagateUnit(nn.Module):
    """ propagate from t_plus to t_minus """
    def __init__(self, args):
        super().__init__()
        self.n_user = args.n_user
        self.n_item = args.n_item

        self.cgnn = CGNN(args)

    def forward(self, adj_obs, t_diff, x_u, x_i, xu_embed, xi_embed):
        x_t = torch.cat([x_u, x_i], 0)
        x_embed = torch.cat([xu_embed, xi_embed], 0)
        norm = torch.norm(x_t, dim=1).max()
        x_t = x_t / norm
        x_embed = x_embed / norm

        z = self.cgnn(x_embed, x_t, t_diff, adj_obs)
        return torch.split(z, [self.n_user, self.n_item], 0)


class DiscUpdateUnit(nn.Module):
    """ update from t_minus to t_plus """
    def __init__(self, args):
        super().__init__()
        self.d_latent = args.d_latent
        self.fc_uu = nn.Linear(self.d_latent, self.d_latent)
        self.fc_ii = nn.Linear(self.d_latent, self.d_latent)
        self.fc_ui = nn.Linear(self.d_latent, self.d_latent, bias=False)
        self.fc_iu = nn.Linear(self.d_latent, self.d_latent, bias=False)

    def forward(self, xu_t_minus, xi_t_minus, adj_ins_i2u, adj_ins_u2i):

        delta_u = F.relu(self.fc_uu(xu_t_minus) + adj_ins_i2u @ self.fc_iu(xi_t_minus))
        delta_i = F.relu(self.fc_ii(xi_t_minus) + adj_ins_u2i @ self.fc_ui(xu_t_minus))

        mask_u = (torch.sparse.sum(adj_ins_i2u, 1).to_dense() > 0).float()
        mask_i = (torch.sparse.sum(adj_ins_u2i, 1).to_dense() > 0).float()

        delta_u = delta_u * mask_u.unsqueeze(1)
        delta_i = delta_i * mask_i.unsqueeze(1)

        xu_t_plus = xu_t_minus + delta_u
        xi_t_plus = xi_t_minus + delta_i

        loss_jump = (delta_u ** 2).sum() / mask_u.sum() + (delta_i ** 2).sum() / mask_i.sum()

        return xu_t_plus, xi_t_plus, loss_jump


class PredictUnit(nn.Module):
    """ predict from t_minus """
    def __init__(self, args):
        super().__init__()
        self.d_latent = args.d_latent
        self.u_pred_mapping = nn.Linear(2 * self.d_latent, 2 * self.d_latent)
        self.i_pred_mapping = nn.Linear(2 * self.d_latent, 2 * self.d_latent)
        nn.init.eye_(self.u_pred_mapping.weight.data)
        nn.init.eye_(self.i_pred_mapping.weight.data)
        nn.init.zeros_(self.u_pred_mapping.bias.data)
        nn.init.zeros_(self.i_pred_mapping.bias.data)

    def forward(self, h_u, h_i):
        h_u = self.u_pred_mapping(h_u)
        h_i = self.i_pred_mapping(h_i)
        return (h_u * h_i).sum(dim=-1)
