import os
from os.path import join
import time

from utils.constant import OCCUR_U_MARK


class Noter(object):

    def __init__(self, args):
        self.args = args

        self.cuda = args.cuda
        self.dataset = args.dataset

        self.lr = args.lr
        self.l2 = args.l2
        self.alpha_jump = args.alpha_jump

        self.occur_u_mark = OCCUR_U_MARK

        self.f_log = join(args.path_log, time.strftime('%m-%d-%H-%M-', time.localtime()) + args.data + '-gpu' +
                          args.cuda + '-' + str(args.alpha_jump) + '.txt')

        self.f_case = join(args.path_case, time.strftime('cs-%m-%d-%H-%M-', time.localtime()) + '-' +
                           str(args.alpha_jump) + '.txt')

        for f in [self.f_log, self.f_case]:
            if os.path.exists(f):
                os.remove(f)  # remove the existing file if duplicate

        self.welcome = ('-' * 20 + ' Experiment: CoPE (CIKM\'21) ' + '-' * 20)
        print('\n' + self.welcome)
        self.write(self.welcome + '\n')

    # write into log file
    def write(self, msg):
        with open(self.f_log, 'a') as out:
            print(msg, file=out)

    # log any message
    def log_msg(self, msg):
        print(msg)
        self.write(msg)

    # print and save experiment briefs
    def log_brief(self):
        msg = f'\n[Info] Experiment (dataset:{self.dataset}, cuda:{self.cuda}) ' \
              f'\n\t| lr {self.lr:.0e} | l2 {self.l2:.0e} | alpha_jump {self.alpha_jump:.0e} |\n'
        self.log_msg(msg)

    # save args into log file
    def save_args(self):
        info = '-' * 10 + ' Experiment settings ' + '-' * 10 + '\n'
        for k, v in vars(self.args).items():
            info += '\n\t{} : {}'.format(k, str(v))
        self.write(info + '\n')

    # print and save train phase result
    def log_train(self, loss, loss_rec, loss_jump, t_gap):
        msg = (f'\t| train | loss {loss:.4f} | loss_rec {loss_rec:.4f} | loss_jump {loss_jump:.4f} '
               f'| time {t_gap:.1f}s | ')
        self.log_msg(msg)

    # print and save valid phase result
    def log_valid(self, loss, loss_rec, loss_jump, mrr, recall):
        msg = (f'\t| valid | loss {loss:.4f} | loss_rec {loss_rec:.4f} | loss_jump {loss_jump:.4f} '
               f'| mrr {mrr:.4f} | recall {recall:.4f} |')
        self.log_msg(msg)

    # print and save test phase result
    def log_test(self, mrr, recall):
        msg = f'\t| test  | mrr {mrr:.4f} | recall {recall:.4f} |'
        self.log_msg(msg)

    # print and save final result
    def log_final_result(self, epoch: int, dict_res: dict):
        self.log_msg('\n' + '-' * 10 + f' TCGRec experiment ends at epoch {epoch} ' + '-' * 10)
        self.log_brief()

        msg = ''
        for type_mode, res in dict_res.items():
            msg += f'\t| {type_mode} | epoch {res[0]} | mrr {res[1]:.4f} | recall {res[2]:.4f}\n'
        self.log_msg(msg)

    # print and save case study
    def log_case(self, res_case):
        self.log_msg(f'\n\t| U-I 1-1 | {list2str(res_case[0][0][:self.occur_u_mark[0]])} |'
                     f'\n\t| U_I 1-2 | {list2str(res_case[0][1][:self.occur_u_mark[0]])} |'
                     f'\n\t| U_I 1-3 | {list2str(res_case[0][2][:self.occur_u_mark[0]])} |'
                     f'\n\t| U_I 2-1 | {list2str(res_case[1][0][:self.occur_u_mark[1]])} |'
                     f'\n\t| U_I 2-2 | {list2str(res_case[1][1][:self.occur_u_mark[1]])} |'
                     f'\n\t| U_I 2-3 | {list2str(res_case[1][2][:self.occur_u_mark[1]])} |'
                     )


def list2str(l):
    msg = ''
    for i in l:
        msg += f' {str(int(i))}'
    return msg[1:]
