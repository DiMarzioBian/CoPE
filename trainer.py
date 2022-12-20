import time
from tqdm import tqdm
import torch

from dataloader import get_dataloader
from models.CoPE import CoPE
from utils.metrics import cal_mrr, cal_recall


class Trainer(object):
    def __init__(self, args, noter):
        self.trainloader, self.valloader, self.testloader = get_dataloader(args)
        self.model = CoPE(args).to(args.device)
        self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()), lr=args.lr,
                                          weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        self.noter = noter

        self.len_train_dl = len(self.trainloader)
        self.len_val_dl = len(self.valloader)
        self.len_test_dl = len(self.testloader)

        self.len_tbptt = args.len_tbptt
        self.alpha_jump = args.alpha_jump
        self.k_metric = args.k_metric

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
            loss_rec_batch, loss_jump_batch, xu_t_plus, xi_t_plus, *_ = self.model(batch, xu_t_plus, xi_t_plus)

            loss_rec_total += loss_rec_batch.item()
            loss_jump_total += loss_jump_batch.item()

            loss_rec_tbptt += loss_rec_batch
            loss_jump_tbptt += loss_jump_batch
            count_tbptt += 1
            count_total += 1

            if (count_tbptt % self.len_tbptt) == 0 or count_total == self.len_train_dl:
                loss = (loss_rec_tbptt + loss_jump_tbptt * self.alpha_jump) / count_tbptt
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                xu_t_plus = xu_t_plus.detach()
                xi_t_plus = xi_t_plus.detach()

                count_tbptt, loss_rec_tbptt, loss_jump_tbptt = 0, 0, 0

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
        else:
            assert mode == 'testing'
            dl = self.testloader

        rank_u = []
        xu_t_plus, xi_t_plus = self.xu_t_plus, self.xi_t_plus
        with torch.no_grad():
            for batch in tqdm(dl, desc='  - ' + mode, leave=False):
                (tgt_u, tgt_i) = batch[4:6]
                loss_rec_batch, loss_jump_batch, xu_t_plus, xi_t_plus, xu_enc, xi_enc = self.model(batch, xu_t_plus,
                                                                                                   xi_t_plus)

                loss_rec_total += loss_rec_batch.item()
                loss_jump_total += loss_jump_batch.item()

                rank_u_batch = self.compute_rank(xu_enc, xi_enc[1:], tgt_i - 1)
                rank_u.extend(rank_u_batch)

            self.xu_t_plus = xu_t_plus.detach()
            self.xi_t_plus = xi_t_plus.detach()

        return rank_u, loss_rec_total, loss_jump_total

    def compute_rank(self, xu_enc, xi_enc, tgt_i):
        scores = self.model.predictor(xu_enc, xi_enc.squeeze(1).unsqueeze(0))
        rank_u = []
        for line, i in zip(scores, tgt_i):
            r = (line >= line[i]).sum().item()
            rank_u.append(r)
        return rank_u
