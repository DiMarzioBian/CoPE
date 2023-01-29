import time
import numpy as np
import torch
from tqdm import tqdm

from dataloader import get_dataloader
from models.CoPE import CoPE
from utils.metrics import cal_mrr, cal_recall
from utils.constant import U_MARK, I_MARK, OCCUR_U_MARK


class Trainer(object):
    def __init__(self, args, noter):
        self.trainloader, self.valloader, self.testloader = get_dataloader(args, noter)
        self.model = CoPE(args).to(args.device)
        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()), lr=args.lr,
                                          weight_decay=args.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        self.noter = noter

        self.len_train_dl = len(self.trainloader)
        self.len_val_dl = len(self.valloader)
        self.len_test_dl = len(self.testloader)

        self.len_tbptt = args.len_tbptt
        self.alpha_jump = args.alpha_jump
        self.k_metric = args.k_metric

        self.case_study = args.case_study
        self.u_mark = U_MARK
        self.i_mark = I_MARK
        self.occur_u_mark = OCCUR_U_MARK
        self.counter_u_mark = [0] * len(self.occur_u_mark)
        self.rank_u_mark = np.ones((len(self.u_mark), len(self.i_mark), max(self.occur_u_mark))) * (-1)

        self.xu_t_plus = None
        self.xi_t_plus = None

        self.noter.log_brief()

    def run_one_epoch(self):
        self.xu_t_plus, self.xi_t_plus = self.model.get_init_states()
        self.run_train()
        return self.run_valid()

    def run_train(self):
        loss_rec_tbptt, loss_jump_tbptt, loss_rec_total, loss_jump_total = 0., 0., 0., 0.
        count_tbptt, count_total = 0, 0
        time_start = time.time()

        self.model.train()
        self.optimizer.zero_grad()

        xu_t_plus, xi_t_plus = self.xu_t_plus, self.xi_t_plus
        for batch in tqdm(self.trainloader, desc='  - training', leave=False):
            (tgt_u, tgt_i) = batch[4:6]
            loss_rec_batch, loss_jump_batch, xu_t_plus, xi_t_plus, xu_enc, xi_enc = self.model(batch, xu_t_plus, xi_t_plus)

            loss_rec_tbptt += loss_rec_batch
            loss_jump_tbptt += loss_jump_batch
            count_tbptt += 1
            count_total += 1

            loss_rec_total += loss_rec_batch.item()
            loss_jump_total += loss_jump_batch.item()

            if (count_tbptt % self.len_tbptt) == 0 or count_total == self.len_train_dl:
                loss = (loss_rec_tbptt + loss_jump_tbptt * self.alpha_jump) / count_tbptt
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                xu_t_plus = xu_t_plus.detach()
                xi_t_plus = xi_t_plus.detach()

                count_tbptt, loss_rec_tbptt, loss_jump_tbptt = 0, 0, 0

            scores_u = self.model.predictor(xu_enc, xi_enc[1:].squeeze(1).unsqueeze(0))
            self.get_rank_case(scores_u, tgt_u)

        loss_rec_total /= self.len_train_dl
        loss_jump_total /= self.len_train_dl
        loss_tr = loss_rec_total + loss_jump_total

        self.noter.log_train(loss_tr, loss_rec_total, loss_jump_total, time.time() - time_start)
        self.xu_t_plus = xu_t_plus  # detached at last batch
        self.xi_t_plus = xi_t_plus

    def run_valid(self):
        # validating phase
        rank_val, loss_rec_val, loss_jump_val = self.rollout('validating')
        loss_rec_val /= self.len_val_dl
        loss_jump_val /= self.len_val_dl
        loss_val = loss_rec_val + loss_jump_val * self.alpha_jump

        recall_val = cal_recall(rank_val, self.k_metric)
        mrr_val = cal_mrr(rank_val)

        self.noter.log_valid(loss_val, loss_rec_val, loss_jump_val, recall_val, mrr_val)
        return recall_val, mrr_val, loss_rec_val

    def run_test(self):
        # testing phase
        rank_test, *_ = self.rollout('testing')

        recall_test = cal_recall(rank_test, self.k_metric)
        mrr_test = cal_mrr(rank_test)

        self.noter.log_test(recall_test, mrr_test)
        return [recall_test, mrr_test]

    def rollout(self, mode: str):
        """ rollout evaluation """
        self.model.eval()
        loss_rec, loss_jump, loss_rec_total, loss_jump_total = 0., 0., 0., 0.

        if mode == 'validating':
            dl = self.valloader
            len_dl = self.len_val_dl
        else:
            assert mode == 'testing'
            dl = self.testloader
            len_dl = self.len_test_dl

        rank_u = []
        xu_t_plus, xi_t_plus = self.xu_t_plus, self.xi_t_plus
        with torch.no_grad():
            for batch in tqdm(dl, desc='  - ' + mode, leave=False):
                (tgt_u, tgt_i) = batch[4:6]
                loss_rec_batch, loss_jump_batch, xu_t_plus, xi_t_plus, xu_enc, xi_enc = self.model(batch,
                                                                                                   xu_t_plus,
                                                                                                   xi_t_plus)

                loss_rec_total += loss_rec_batch.item() / len_dl
                loss_jump_total += loss_jump_batch.item() / len_dl

                rank_u_batch, scores_u = self.compute_rank(xu_enc, xi_enc[1:], tgt_i - 1)
                rank_u.extend(rank_u_batch)

                self.get_rank_case(scores_u, tgt_u)

            self.xu_t_plus = xu_t_plus.detach()
            self.xi_t_plus = xi_t_plus.detach()

        return rank_u, loss_rec_total, loss_jump_total

    def compute_rank(self, xu_enc, xi_enc, tgt_i):
        scores = self.model.predictor(xu_enc, xi_enc.squeeze(1).unsqueeze(0))
        rank_u = []
        for line, i in zip(scores, tgt_i):
            r = (line >= line[i]).sum().item()
            rank_u.append(r)
        return rank_u, scores

    def cal_rank_case(self, scores, idx_mark, counter):
        for i, item in enumerate(self.i_mark):
            r = (scores > scores[item]).sum().item() + 1
            self.rank_u_mark[idx_mark, i, counter] = r

    def get_rank_case(self, scores_u, tgt_u):
        flag_u_mark = [(tgt_u == u).nonzero(as_tuple=True)[0] for u in self.u_mark]

        for i, flag in enumerate(flag_u_mark):
            if len(flag) != 0:
                if len(flag) > 1:
                    flag = flag[0]
                self.cal_rank_case(scores_u[flag].squeeze(0), i, self.counter_u_mark[i])
                self.counter_u_mark[i] += 1

    def reset_rank_case(self):
        self.counter_u_mark = [0] * len(self.occur_u_mark)
        self.rank_u_mark = np.ones((len(self.u_mark), len(self.i_mark), max(self.occur_u_mark))) * (-1)

