import argparse
import torch
import os
import torch.backends.cudnn as cudnn
import numpy as np

from utils.data import arg2str, str2bool


class BaseConfig(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser()

        #basic cfg
        self.parser.add_argument('--exp_name', type=str, help='Experiment name', default='test')
        self.parser.add_argument('--gpu_id', default='2', type=str, help='GPU ID Use to train/test module')
        self.parser.add_argument('--save_folder', default='/data2/wangjinhong/output/wjh/save_model/Cataract_OCT/checkpoint_catar/', help='Path to save checkpoint models')
        self.parser.add_argument('--save_log', default='/data2/wangjinhong/output/wjh/save_log/Cataract_OCT/logs_catar/',help='Path to save checkpoint models')
        self.parser.add_argument('--resume', default=None, type=str, help='Path to target resume checkpoint')
        self.parser.add_argument('--pretrained', default=None, type=str, help='Path to target pretrained checkpoint')
        self.parser.add_argument('--img_root', default='/data2/wangjinhong/data/白内障OCT/OCT图像(总1666)/', type=str, help='Image root')
        self.parser.add_argument('--data_list', default='/data2/wangjinhong/data/白内障OCT/黄斑OCT-AI数据汇总6.csv')
        self.parser.add_argument('--v_list', default='/data2/wangjinhong/data/白内障OCT/v6.txt')

        self.parser.add_argument('--train_csv', default='/data2/wangjinhong/data/白内障OCT/train.txt', type=str,
                                 help='Path of training label files')
        self.parser.add_argument('--val_csv', default='/data2/wangjinhong/data/白内障OCT/val.txt', type=str,
                                 help='Path of validation label files')
        self.parser.add_argument('--model_name', default='ResVAE', type=str, help='module')
        self.parser.add_argument('--eval_only', action='store_true', help='use eval mode')

        #train cfg

        self.parser.add_argument('--start_iter', default=0, type=int,help='Begin counting iterations starting from this value (should be used with resume)')
        self.parser.add_argument('--max_iter', default=4000, type=int, help='Number of training iterations')
        self.parser.add_argument('--warmup_steps', default=100, type=int, help='Number of warmup iterations')

        self.parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
        self.parser.add_argument('--lr_adjust', default='fix', choices=['fix', 'poly'], type=str, help='Learning Rate Adjust Strategy')
        self.parser.add_argument('--stepvalues', default=[1000,2000,3000], nargs='*', type=int, help='# of iter to change lr')

        self.parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, help='Weight decay for SGD')
        self.parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD lr')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
        self.parser.add_argument('--optim', default='SGD', type=str, choices=['SGD', 'Adam'], help='Optimizer')
        self.parser.add_argument('--save_freq', default=400, type=int, help='save weights every # iterations')
        self.parser.add_argument('--display_freq', default=10, type=int,help='display training metrics every # iterations,  =train_epoch_size/2, train_epoch_size = train_total_num/batch_size')  # 20 for one step data show
        self.parser.add_argument('--val_freq', default=10, type=int,help='do validation every # iterations, =train_epoch_size-1')
        self.parser.add_argument('--k_fold', default=4, type=int, help='use k fold validation')
        self.parser.add_argument('--k_fold_start', default=0, type=int, help='start from k fold')
        self.parser.add_argument('--dropout', default=0.5, type=float, help='Dropout rate of Resnet')

        #data cfg
        self.parser.add_argument('--z_dim', default=128, type=int)
        self.parser.add_argument('--resnet_layers', default=50, type=int, choices=[18, 34, 50, 101], help='Number of Resnet layers')
        self.parser.add_argument('--cbt_layers', default=3, type=int, help='Number of Resnet layers')
        self.parser.add_argument('--patchsize', default=4, type=int, help='Number of patch sqrt')
        self.parser.add_argument('--cls_true', action='store_true', help='use surgery after image')
        self.parser.add_argument('--binary', action='store_true', help='use surgery after image')


    def initialize(self, fixed=None):
        # print(fixed)
        self.args = self.parse(fixed)
        print('--------------Options Log-------------')
        print(arg2str(self.args))
        print('--------------Options End-------------')

        #Create save dict
        if not os.path.exists(self.args.save_folder):
            os.makedirs(self.args.save_folder, exist_ok=True)



        return self.args


    def parse(self, fixed=None):

        if fixed is not None:
            args = self.parser.parse_known_args(fixed)[0]
        else:
            args = self.parser.parse_known_args()[0]

        return args


    def save_result(self, acc_list, loss_list, loss_refine_list):
        acc_mean = np.mean(acc_list)
        acc_std = np.std(acc_list)

        loss_mean = np.mean(loss_list)
        loss_std = np.std(loss_list)

        loss_refine_mean = np.mean(loss_refine_list)
        loss_refine_std = np.std(loss_refine_list)



        acc_save_path = os.path.join(self.args.save_folder, self.args.exp_name) + '/acc_{}_std_{}.txt'.format(acc_mean, acc_std)
        loss_save_path = os.path.join(self.args.save_folder, self.args.exp_name) + '/loss_{}_std_{}.txt'.format(loss_mean, loss_std)
        loss_refine_save_path = os.path.join(self.args.save_folder, self.args.exp_name) + '/loss2_{}_std_{}.txt'.format(loss_refine_mean, loss_refine_std)

        with open(acc_save_path, "w") as f:
            for c, ac in enumerate(acc_list):
                f.write('train_{}_acc is {}\n'.format(c, ac))
        f.close()

        with open(loss_save_path, "w") as f:
            for c, ac in enumerate(loss_list):
                f.write('train_{}_loss is {}\n'.format(c, ac))
        f.close()

        with open(loss_refine_save_path, "w") as f:
            for c, ac in enumerate(loss_refine_list):
                f.write('train_{}_loss is {}\n'.format(c, ac))
        f.close()
