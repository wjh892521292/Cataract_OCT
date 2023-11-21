import os, sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import models
from datetime import datetime
from tensorboardX import SummaryWriter

from config.cfg import arg2str
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score

'''
trainer回归差值
trainer2回归真实值
'''

class DefaultTrainer(object):

    def __init__(self, args, k_fold=""):
        self.args = args
        self.batch_size = args.batch_size
        self.lr = self.lr_current = args.lr
        self.start_iter = args.start_iter
        self.max_iter = args.max_iter
        self.warmup_steps = args.warmup_steps
        self.eval_only = args.eval_only
        self.model = getattr(models, args.model_name.lower())(args)
        self.model.cuda()
        self.MAE = nn.L1Loss()
        self.MSE = nn.MSELoss()
        self.k_fold = k_fold
        self.max_acc = 0.0
        self.min_loss = self.min_loss_refine = 1000000

        self.start = 0

        self.optim = getattr(torch.optim, args.optim) \
            (filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=args.weight_decay)

        if args.resume:
            if os.path.isfile(self.args.resume):
                iter, index = self.load_model(args.resume)
                self.start_iter = iter


    def train_iter(self, step, dataloader):
        if self.args.after_true:
            ids, img_a_r, img_a_c, img_b_r, img_b_c, label_a, label_b = dataloader.next()
            img_a_r, img_a_c, img_b_r, img_b_c = img_a_r.float().cuda(), img_a_c.float().cuda(), img_b_r.float().cuda(), img_b_c.float().cuda()
        else:
            ids, img_a_r, img_a_c, label_a, label_b = dataloader.next()
            img_a_r, img_a_c = img_a_r.float().cuda(), img_a_c.float().cuda()



        target = label_b
        target = target.unsqueeze(dim=1).cuda()
        target2 = label_b - label_a
        target2 = target2.unsqueeze(dim=1).cuda()

        self.model.train()
        if self.eval_only:
            self.model.eval()



        if self.args.model_name.lower() == 'resnet' or self.args.model_name.lower() == 'resnet1' :
            score = self.model(img_a_r, img_a_c)
            loss = loss1 = loss2 = loss3 = self.MAE(score, target)
            kl_div = 0

        elif self.args.model_name.lower() == 'resnet2' or self.args.model_name.lower() == 'resnet3' or self.args.model_name.lower() == 'resnet4' :
            score = self.model(img_a_r, img_a_c, label_a.unsqueeze(dim=1).cuda())
            loss = loss1 = loss2 = loss3 = self.MAE(score, target)
            kl_div = 0

        elif self.args.model_name.lower() == 'resnet5':
            score, score_refine = self.model(img_a_r, img_a_c, img_b_r, img_b_c, label_a.unsqueeze(dim=1).cuda())
            loss1 = self.MAE(score, target)
            loss2 = self.MAE(score_refine, target)
            loss = loss3 = loss1 + loss2
            kl_div = 0

        elif self.args.model_name.lower() == 'resvae':
            output_r, output_c, score, mu, logvar = self.model(img_a_r, img_a_c)
            loss1 = self.MSE(output_r, img_b_r)
            loss2 = self.MSE(output_c, img_b_c)
            loss3 = self.MAE(score, target)
            loss = 0.1 * loss1 + 0.1 * loss2 + loss3
            kl_div = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            if step <= self.max_iter / 2:
                loss = loss + kl_div

        elif self.args.model_name.lower() == 'resvae2':
            output_r, output_c, score = self.model(img_a_r, img_a_c, label_a.unsqueeze(dim=1).cuda())
            loss1 = self.MSE(output_r, img_b_r) + self.MSE(output_c, img_b_c)
            loss2 = loss3 = self.MAE(score, target)
            loss = 0.05 * loss1 + loss2
            kl_div = 0

        elif self.args.model_name.lower() == 'resvae3' or self.args.model_name.lower() == 'resvae4':
            output_r, output_c, score, score_refine = self.model(img_a_r, img_a_c, label_a.unsqueeze(dim=1).cuda())
            loss1 = self.MSE(output_r, img_b_r) + self.MSE(output_c, img_b_c)
            loss2 = self.MAE(score, target)
            loss3 = self.MAE(score_refine, target)
            loss = 0.05 * loss1 + loss2 + loss3
            kl_div = 0
        elif 'cbt' in self.args.model_name.lower():
            score = self.model(img_a_r, img_a_c, label_a.unsqueeze(dim=1).cuda())
            score2 = score - label_a.unsqueeze(dim=1).cuda()
            loss = loss1 = loss2 = loss3 = self.MAE(score, target)
            loss3 = self.MAE(score2, target2)
            kl_div = 0
        # print(step, ' ', ids)
        # print('术前视力:', label_a.item())
        # print('预测视力差：', score.item())
        # print('预测视力：', label_a.item() + score.item())
        # print('术后视力:', label_b.item())

        if self.start == 0:
            self.init_writer()
            self.start = 1




        print( 'Training - Step: {} - Loss: {:.4f} - Loss_img: {:.4f} - Loss_score: {:.4f} - Loss_score_refine: {:.4f} - Loss_kl: {:.4f}' \
               .format(step, loss.item(), loss1.item(), loss2.item(), loss3.item(), kl_div))

        loss.backward()
        self.optim.step()
        self.model.zero_grad()



        if step % self.args.display_freq == 0:
            score2 = score - label_a.unsqueeze(dim=1).cuda()
            pred = (score2 >= 0.2).cpu()
            gt = (target2 >= 0.2).cpu()

            acc = accuracy_score(gt, pred)
            recall = recall_score(gt, pred, average='binary')
            precision = precision_score(gt, pred, average='binary')
            f1 = f1_score(gt, pred, average='binary')

            print(
                'Training - Step: {} - Acc: {:.4f} - Precision: {:.4f} - Recall: {:.4f} - f1score:{:.4f} - lr:{:.4f}' \
                .format(step, acc, precision, recall, f1, self.lr_current))

            # scalars = [loss.item(), acc, prec, recall, f1, kap]
            # names = ['loss', 'acc', 'precision', 'recall', 'f1score', 'kappa']
            scalars = [loss.item(), loss1.item(), loss2.item(), loss3.item(), kl_div, acc, precision, recall, f1, self.lr_current]
            names = ['loss', 'loss_img1', 'loss_img2', 'loss_score', 'loss_kl', 'acc', 'precision', 'recall', 'f1score', 'lr']
            write_scalars(self.writer, scalars, names, step, 'train'+self.k_fold)



    def train(self, train_dataloader, valid_dataloader=None):

        train_epoch_size = len(train_dataloader)
        train_iter = iter(train_dataloader)
        val_epoch_size = len(valid_dataloader)

        for step in range(self.start_iter, self.max_iter):

            if step % train_epoch_size == 0:
                print('Epoch: {} ----- step:{} - train_epoch size:{}'.format(step // train_epoch_size, step,
                                                                             train_epoch_size))
                train_iter = iter(train_dataloader)

            self._adjust_learning_rate_iter(step)
            self.train_iter(step, train_iter)

            if (valid_dataloader is not None) and (
                    step % self.args.val_freq == 0 or step == self.args.max_iter - 1) and (step != 0):
                val_iter = iter(valid_dataloader)
                val_acc, val_loss, val_loss_refine = self.validation(step, val_iter, val_epoch_size)
                if val_acc > self.max_acc:
                    self.delete_model(best='best_acc', index=self.max_acc)
                    self.max_acc = val_acc
                    self.save_model(step, best='best_acc', index=self.max_acc, gpus=1)

                if val_loss.item() < self.min_loss:
                    self.delete_model(best='min_loss', index=self.min_loss)
                    self.min_loss = val_loss.item()
                    self.save_model(step, best='min_loss', index=self.min_loss, gpus=1)

                if val_loss_refine.item() < self.min_loss_refine:
                    self.delete_model(best='min_loss_refine', index=self.min_loss_refine)
                    self.min_loss_refine = val_loss_refine.item()
                    self.save_model(step, best='min_loss_refine', index=self.min_loss_refine, gpus=1)

        return self.max_acc, self.min_loss, self.min_loss_refine
        # if step % self.args.save_freq == 0 and step != 0:
        #     self.model.save_model(step, best='step', index=step, gpus=1)


    def validation(self, step, val_iter, val_epoch_size):

        print('============Begin Validation============:step:{}'.format(step))

        self.model.eval()

        total_score = []
        total_score2 = []
        total_target = []
        total_target2 = []
        with torch.no_grad():
            for i in range(val_epoch_size):
                if self.args.after_true:
                    ids, img_a_r, img_a_c, img_b_r, img_b_c, label_a, label_b = next(val_iter)
                    img_a_r, img_a_c, img_b_r, img_b_c = img_a_r.float().cuda(), img_a_c.float().cuda(), img_b_r.float().cuda(), img_b_c.float().cuda()

                else:
                    ids, img_a_r, img_a_c, label_a, label_b = next(val_iter)
                    img_a_r, img_a_c = img_a_r.float().cuda(), img_a_c.float().cuda()

                target = label_b
                target = target.unsqueeze(dim=1).cuda()
                target2 = label_b - label_a
                target2 = target2.unsqueeze(dim=1).cuda()
                if self.args.model_name.lower() == 'resnet' or self.args.model_name.lower() == 'resnet1':
                    score = score2 = self.model(img_a_r, img_a_c)


                elif self.args.model_name.lower() == 'resnet2' or self.args.model_name.lower() == 'resnet3' or self.args.model_name.lower() == 'resnet4' :
                    score = score2 = self.model(img_a_r, img_a_c, label_a.unsqueeze(dim=1).cuda())


                elif  self.args.model_name.lower() == 'resnet5':
                    score, score2 = self.model(img_a_r, img_a_c, img_b_r, img_b_c,
                                                          label_a.unsqueeze(dim=1).cuda())


                elif self.args.model_name.lower() == 'resvae':
                    output_r, output_c, score, mu, logvar = self.model(img_a_r, img_a_c)
                    score2 = score
                elif self.args.model_name.lower() == 'resvae2':
                    output_r, output_c, score = self.model(img_a_r, img_a_c,label_a.unsqueeze(dim=1).cuda())
                    score2 = score
                elif self.args.model_name.lower() == 'resvae3' or self.args.model_name.lower() == 'resvae4':
                    output_r, output_c, score, score2 = self.model(img_a_r, img_a_c,
                                                                         label_a.unsqueeze(dim=1).cuda())
                elif 'cbt' in self.args.model_name.lower():
                    score = score2 = self.model(img_a_r, img_a_c, label_a.unsqueeze(dim=1).cuda())
                    score2 = score2 - label_a.unsqueeze(dim=1).cuda()
                if i == 0:
                    total_score = score
                    total_score2 = score2
                    total_target = target
                    total_target2 = target2
                else:
                    if len(score.shape) == 1:
                        score = score.unsqueeze(0)
                        score2 = score2.unsqueeze(0)


                    if len(target.shape) == 1:
                        target = target.unsqueeze(0)
                        target2 = target2.unsqueeze(0)
                    total_score = torch.cat((total_score, score), 0)
                    total_score2 = torch.cat((total_score2, score2), 0)

                    total_target = torch.cat((total_target, target), 0)
                    total_target2 = torch.cat((total_target2, target2), 0)

        loss = self.MAE(total_score, total_target)
        loss2 = self.MAE(total_score2, total_target2)


        pred = (total_score2 >= 0.2 ).cpu()
        gt = (total_target2 >= 0.2).cpu()

        acc = accuracy_score(gt, pred)
        recall = recall_score(gt, pred, average='binary')
        precision = precision_score(gt, pred, average='binary')
        f1 = f1_score(gt, pred, average='binary')



        print(
            'Valid - Step: {} \n Loss: {:.4f} \n Loss2: {:.4f} \nAcc: {:.4f} \n Precision: {:.4f} \n Recall: {:.4f} \n f1score:{:.4f} ' \
                .format(step, loss.item(), loss2.item(), acc, precision, recall, f1))

        scalars = [loss.item(), loss2.item(), acc, precision, recall, f1]
        names = ['loss_score',  'loss_score_refine', 'acc', 'precision', 'recall', 'f1score']
        write_scalars(self.writer, scalars, names, step, 'val'+self.k_fold)

        return acc, loss, loss2



################

    def _adjust_learning_rate_iter(self, step):
        """Sets the learning rate to the initial LR decayed by 10 at every specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        if step <= self.warmup_steps:  # 增大学习率
            self.lr_current = self.args.lr * float(step) / float(self.warmup_steps)

        if self.args.lr_adjust == 'fix':
            if step in self.args.stepvalues:
                self.lr_current = self.lr_current * self.args.gamma
        elif self.args.lr_adjust == 'poly':
            self.lr_current = self.args.lr * (1 - step / self.args.max_iter) ** 0.9

        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr_current

    def init_writer(self):
        """ Tensorboard writer initialization
            """

        if not os.path.exists(self.args.save_folder):
            os.makedirs(self.args.save_folder, exist_ok=True)


        if self.args.exp_name == 'test':
            log_path = os.path.join(self.args.save_log, self.args.exp_name)
        else:
            log_path = os.path.join(self.args.save_log, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + self.args.exp_name)
        log_config_path = os.path.join(log_path, 'configs.log')

        self.writer = SummaryWriter(log_path)
        with open(log_config_path, 'w') as f:
            f.write(arg2str(self.args))


    def load_model(self, model_path):
        if os.path.exists(model_path):
            load_dict = torch.load(model_path)
            net_state_dict = load_dict['net_state_dict']

            try:
                self.model.load_state_dict(net_state_dict)
            except:
                self.model.module.load_state_dict(net_state_dict)
            self.iter = load_dict['iter'] + 1
            index = load_dict['index']

            print('Model Loaded!')
            return self.iter, index
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    def delete_model(self, best, index):
        if index == 0 or index == 1000000:
            return
        save_fname = '%s_%s_%s.pth' % (self.model.model_name(), best, index)
        save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
        os.remove(save_path)

    def save_model(self, step, best='best_acc', index=None, gpus=1):

        model_save_path = os.path.join(self.args.save_folder, self.args.exp_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)

        if gpus == 1:
            save_fname = '%s_%s_%s.pth' % (self.model.model_name(), best, index)
            save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
            save_dict = {
                'net_state_dict': self.model.state_dict(),
                'exp_name': self.args.exp_name,
                'iter': step,
                'index': index
            }
        else:
            save_fname = '%s_%s_%s.pth' % (self.model.module.model_name(), best, index)
            save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
            save_dict = {
                'net_state_dict': self.model.module.state_dict(),
                'exp_name': self.args.exp_name,
                'iter': step,
                'index': index
            }
        torch.save(save_dict, save_path)
        print(best + ' Model Saved')

def write_scalars(writer, scalars, names, n_iter, tag=None):
    for scalar, name in zip(scalars, names):
        if tag is not None:
            name = '/'.join([tag, name])
        writer.add_scalar(name, scalar, n_iter)