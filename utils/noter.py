import os
from os.path import join
import time


class Noter(object):

    def __init__(self, args):
        self.args = args

        self.cuda = args.cuda
        self.dataset = args.dataset

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.alpha_jump = args.alpha_jump

        self.f_log = join(args.path_log, time.strftime('%m-%d-%H-%M-', time.localtime()) + args.data + '-gpu' +
                          args.cuda + str(args.alpha_jump) + '.txt')

        if os.path.exists(self.f_log):
            os.remove(self.f_log)  # remove the existing file if duplicate

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
              f'\n\t| lr {self.lr:.0e} | weight_decay {self.weight_decay:.0e} | alpha_jump {self.alpha_jump:.0e} |\n'
        print(msg)
        self.write(msg)

    # save args into log file
    def save_args(self):
        info = '-' * 10 + ' Experiment settings ' + '-' * 10 + '\n'
        for k, v in vars(self.args).items():
            info += '\n\t{} : {}'.format(k, str(v))
        self.write(info + '\n')

    # print and save train phase result
    def log_train(self, loss, loss_rec, loss_jump, t_gap):
        msg = (f'\t| train | loss {loss:.4f} | loss_rec {loss_rec:.4f} | loss_jump {loss_jump:.4f} '
               f'| time {t_gap:.1f}s |')
        print(msg)
        self.write(msg)

    # print and save valid phase result
    def log_valid(self, loss, loss_rec, loss_jump, recall, mrr):
        msg = (f'\t| valid | loss {loss:.4f} | loss_rec {loss_rec:.4f} | loss_jump {loss_jump:.4f} '
               f'| recall {recall:.4f} | mrr {mrr:.4f} |')
        print(msg)
        self.write(msg)

    # print and save test phase result
    def log_test(self, recall, mrr):
        msg = f'\t| test  | recall {recall:.4f} | mrr {mrr:.4f} |'
        print(msg)
        self.write(msg)

    # print and save final result
    def log_final_result(self, epoch: int, dict_res: dict):
        msg = '-' * 10 + f'Modeling ends at epoch {epoch}' + '-' * 10
        self.log_brief()

        for type_mode, res in dict_res.items():
            msg += f'\n\t| {type_mode} | epoch {res[0]} | recall_test {res[1]:.4f} | mrr_test {res[2]:.4f} |'
        msg += '\n'
        print(msg)
        self.write(msg)
