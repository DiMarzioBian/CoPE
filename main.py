import os
import argparse
import numpy as np
import torch

from utils.noter import Noter
from utils.constant import MAPPING_DATASET, IDX_PAD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='video', help='garden, video, game, ml, mlm or yoo')
    parser.add_argument('--case_study', action='store_false', help='order of Chebyshev polynomial')

    # cgnn
    parser.add_argument('--inv_order', type=int, default=10, help='order of Neumann series')
    parser.add_argument('--exp_order', type=int, default=3, help='order of Chebyshev polynomial')

    # jump loss
    parser.add_argument('--alpha_jump', type=float, default=0, help='ratio of update loss, 1e-2, 1e-3, 1e-4, 1e-5')

    # hyperparameters
    parser.add_argument('--d_latent', type=int, default=128)
    parser.add_argument('--alpha_spectrum', type=float, default=0.98, help='limited spectrum of graph')
    parser.add_argument('--n_neg_sample', type=int, default=8)

    # optimizer
    parser.add_argument('--n_epoch', type=int, default=50)
    parser.add_argument('--len_tbptt', type=int, default=20, help='truncated back propagation through time')
    parser.add_argument('--len_meta', type=int, default=5, help='conduct meta learning')
    parser.add_argument('--lr', type=float, default=1e-3, help='5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 1e-4')
    parser.add_argument('--l2', type=float, default=5e-3, help='weight_decay, 1e-3, 1e-4, 1e-5, 0')
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='i.e. gamma value')
    parser.add_argument('--n_lr_decay', type=int, default=5)

    # training settings
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--k_metric', type=int, default=10)
    parser.add_argument('--proportion_train', type=float, default=0.8, help='proportion of training set')
    parser.add_argument('--n_batch_load', type=int, default=10, help='number of thread loading batches')
    parser.add_argument('--es_patience', type=int, default=10)

    args = parser.parse_args()

    args.idx_pad = IDX_PAD
    (args.dataset, args.path_raw) = MAPPING_DATASET[args.data]
    args.lr_min = args.lr ** (args.n_lr_decay + 1)
    if args.dataset != 'aivideo':
        args.case_study = False

    args.path_raw = f'data/{args.path_raw}'
    args.path_csv = f'data_processed/{args.dataset}_5.csv'
    args.path_log = f'log/'
    args.path_case = f'log/case_study/'
    for p in [args.path_log, args.path_case]:
        if not os.path.exists(p):
            os.makedirs(p)

    args.device = torch.device('cuda:' + args.cuda) if torch.cuda.is_available() else torch.device('cpu')

    # seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # initialize
    noter = Noter(args)
    if args.case_study:
        from trainer_case import Trainer
    else:
        from trainer import Trainer
    trainer = Trainer(args, noter)

    # modeling
    recall_best, mrr_best, loss_best = 0, 0, 1e5
    res_recall_final, res_mrr_final, res_loss_final = [0]*5, [0]*5, [0]*5
    res_case = {}
    epoch, es_counter = 0, 0
    lr_register = args.lr

    for epoch in range(1, args.n_epoch+1):
        noter.log_msg(f'\n[Epoch {epoch}]')
        recall_val, mrr_val, loss_rec_val = trainer.run_one_epoch()
        trainer.scheduler.step()

        # models selection
        msg_best_val = ''
        if loss_rec_val < loss_best:
            loss_best = loss_rec_val
            msg_best_val += f' loss |'
            if args.case_study:
                res_case['loss'] = trainer.rank_u_mark

        if recall_val > recall_best:
            recall_best = recall_val
            msg_best_val += f' recall |'
            if args.case_study:
                res_case['recall'] = trainer.rank_u_mark

        if mrr_val > mrr_best:
            mrr_best = mrr_val
            msg_best_val += f' mrr |'
            if args.case_study:
                res_case['mrr'] = trainer.rank_u_mark

        if len(msg_best_val) > 0:
            res_test = trainer.run_test()
            noter.log_msg('\t| new   |' + msg_best_val)

            if 'loss' in msg_best_val:
                res_loss_final = [epoch] + res_test
            if 'recall' in msg_best_val:
                res_recall_final = [epoch] + res_test
            if 'mrr' in msg_best_val:
                res_mrr_final = [epoch] + res_test

        # case study on amazon instant video dataset
        if args.case_study:
            noter.log_case(trainer.rank_u_mark)
            trainer.reset_rank_case()

        # lr changing notice
        lr_current = trainer.scheduler.get_last_lr()[0]
        if lr_register != lr_current:
            if trainer.optimizer.param_groups[0]['lr'] == args.lr_min:
                noter.log_msg(f'\t| lr    | reaches btm | {args.lr_min:.2e} |')
            else:
                noter.log_msg(f'\t| lr    | from {lr_register:.2e} | to {lr_current:.2e} |')
                lr_register = lr_current

        # early stop
        if loss_rec_val > loss_best:
            es_counter += 1
            noter.log_msg(f'\t| es    | {es_counter} / {args.es_patience} |')
        elif es_counter != 0:
            es_counter = 0
            noter.log_msg(f'\t| es    | 0 / {args.es_patience} |')

        if es_counter >= args.es_patience:
            break

    noter.log_final_result(epoch, {
        'loss  ': res_loss_final,
        'recall': res_recall_final,
        'mrr   ': res_mrr_final,
    })


if __name__ == '__main__':
    main()
