###########################################################################
# PAFNet - PyTorch Implementation
# Created by: Hammond Liu
# Copyright (c) 2021
###########################################################################

import os
import shutil
import sys
import time
import pickle
import random
from PIL import Image
import cv2
import numpy as np
from addict import Dict
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils import data
from tensorboardX import SummaryWriter
import torchvision.transforms as transform

from thop import profile
from thop import clever_format

# Default Work Dir: /scratch/[NetID]/PAFNet/
from model.utils.visualize import show_img

BASE_DIR = os.getcwd()
sys.path.append(BASE_DIR)
# config: sys.argv[1] & sys.argv[2]
# SMY_PATH = os.path.join('./results/', sys.argv[1], sys.argv[2][:sys.argv[2].find('_')], sys.argv[2])
SMY_PATH = './results/nyud/pcgf/pcgf_pc'
# GPU ids (only when there are multiple GPUs)
# GPUS = [0, 1]
# 指定哪张卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from model.model import get_pafnet
from model.datasets import get_dataset
from model.net.loss import SegmentationLoss

import model.utils as utils
from config import get_config


class Trainer():
    def __init__(self, args):
        config = args
        args = config.training
        self.config = config
        self.args = args
        for k, v in config.items():
            print('[%s]: %s' % (k, v))

        # dataset_root = 'sunrgbd'
        # dataset_name = 'pcgf_pc'
        # exp_id = sys.argv[0]
        #
        # dataset_path = os.path.join(dataset_root, dataset_name)

        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),  # convert RGB [0,255] to FloatTensor in range [0, 1]
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])  # mean and std based on imageNet
        if args.dataset in ('nyud', 'nyud_tmp'):
            dep_transform = transform.Compose([
                transform.ToTensor(),
                transform.Normalize(mean=[0.2798], std=[0.1387])  # mean and std for depth
            ])
        elif args.dataset == 'sunrgbd':
            dep_transform = transform.Compose([
                transform.ToTensor(),
                transform.Lambda(lambda x: x.to(torch.float)),
                transform.Normalize(mean=[19025.15], std=[9880.92])  # mean and std for depth
            ])
        else:
            raise ValueError('Unable to transform depth on the selected dataset.')

        # dataset
        data_kwargs = {'transform': input_transform, 'dep_transform': dep_transform,
                       'base_size': args.base_size, 'crop_size': args.crop_size}
        trainset = get_dataset(args.dataset, root='./data/', split=args.train_split, mode='train',
                               **data_kwargs)  # root=sys.argv[3]
        testset = get_dataset(args.dataset, root='./data/', split='val', mode='val', **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
        self.train_step = len(self.trainloader) // 4
        self.val_step = len(self.valloader) // 4
        self.nclass = trainset.num_class

        # model and params
        model = get_pafnet(args.dataset, config=self.config)
        print(model)

        # lr setting
        enc = model.encoder.encoder
        base_modules = [enc.rgb_base, enc.dep_base]
        base_ids = utils.get_param_ids(base_modules)
        base_params = filter(lambda p: id(p) in base_ids, model.parameters())
        other_params = filter(lambda p: id(p) not in base_ids, model.parameters())
        self.optimizer = torch.optim.SGD([{'params': base_params, 'lr': args.lr},
                                          {'params': other_params, 'lr': args.lr * 10}],
                                         lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # lr scheduler
        self.scheduler = utils.LR_Scheduler_Head(
            mode=args.lr_scheduler,
            base_lr=args.lr,
            num_epochs=args.epochs,
            iters_per_epoch=len(self.trainloader),
            warmup_epochs=5
        )
        self.best_pred = (0.0, 0.0)

        # using cuda
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        if args.cuda:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model, device_ids=GPUS)
                self.multi_gpu = True
            else:
                self.multi_gpu = False
        self.model = model.to(self.device)

        # init class weight
        class_wt = None
        wt_dict = {
            'nyud': 'nyud/weight',
            'nyud_tmp': 'nyud_tmp/weight',
            'sunrgbd': 'sunrgbd/weight'
        }
        # if args.class_weight is not None:
        #     fname = 'wt%d.pickle' % args.class_weight
        #     with open(os.path.join(sys.argv[3], wt_dict[args.dataset], fname), 'rb') as handle:
        #         wt = pickle.load(handle)
        #     class_wt = torch.FloatTensor(wt).to(self.device)
        # print('class weight = %s(%s).' % (wt_dict[args.dataset], args.class_weight))

        # criterions
        self.criterion = SegmentationLoss(
            aux=self.config.decoder_args.aux,
            aux_weight=args.aux_weight,
            nclass=self.nclass,
            weight=class_wt
        )

        # writing summary
        if not os.path.isdir(SMY_PATH):
            utils.mkdir(SMY_PATH)
        self.writer = SummaryWriter(SMY_PATH)

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()

        # scaler =GradScaler()        #使用AMP（PF16)减小显存使用

        total_inter, total_union, total_correct, total_label, total_loss = 0, 0, 0, 0, 0
        for i, (image, dep, target) in enumerate(self.trainloader):
            self.scheduler(self.optimizer, i, epoch, sum(self.best_pred))
            self.optimizer.zero_grad()

            image, dep, target = image.to(self.device), dep.to(self.device), target.to(self.device)
            # with torch.cuda.amp.autocast():

            outputs = self.model(image, dep)
            loss = self.criterion(*outputs, target)

            # scaler.scale(loss).backward()
            # scaler.step(self.optimizer)
            # scaler.update()

            loss.backward()
            self.optimizer.step()

            correct, labeled = utils.batch_pix_accuracy(outputs[0].data, target)
            inter, union = utils.batch_intersection_union(outputs[0].data, target, self.nclass)
            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            train_loss += loss.item()

            if (i + 1) % self.train_step == 0:
                print('epoch {}, step {}, loss {}'.format(epoch + 1, i + 1, train_loss / 50))
                self.writer.add_scalar('train_loss', train_loss / 50, epoch * len(self.trainloader) + i)
                train_loss = 0.0

        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IOU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIOU = IOU.mean()
        print('epoch {}, pixel Acc {}, mean IOU {}'.format(epoch + 1, pixAcc, mIOU))
        self.writer.add_scalar("mean_iou/train", mIOU, epoch)
        self.writer.add_scalar("pixel accuracy/train", pixAcc, epoch)
        self.writer.add_scalar('check_info/base_lr0', self.optimizer.param_groups[0]['lr'], epoch)
        self.writer.add_scalar('check_info/other_lr1', self.optimizer.param_groups[1]['lr'], epoch)

    def train_n_evaluate(self):

        results = Dict({'miou': [], 'pix_acc': []})

        for epoch in range(self.args.epochs):
            # run on one epoch
            print("\n===============train epoch {}/{} ==========================\n".format(epoch + 1, self.args.epochs))

            # one full pass over the train set
            self.training(epoch)

            # evaluate for one epoch on the validation set
            print('\n===============start testing, training epoch {} ===============\n'.format(epoch + 1))
            # rgb_input = torch.randn(1, 3, 680, 480).to(self.device)  # 示例RGB输入
            # depth_input = torch.randn(1, 1, 680, 480).to(self.device)  # 示例深度输入，假设深度是单通道
            #
            # # 使用profile计算FLOPs和参数，注意inputs参数应为模型输入
            # flops, params = profile(self.model, inputs=(rgb_input, depth_input,))
            # flops, params = clever_format([flops, params], "%.3f")
            # print("FLOPs:", flops, "Params:", params)

            # start_time = time.time()  # 记录推理开始时间
            pixAcc, mIOU, loss = self.validation(epoch)
            # infer_time = time.time() - start_time  # 计算推理时间
            # print('average_infer_time:',infer_time)
            print('evaluation pixel acc {}, mean IOU {}, loss {}'.format(pixAcc, mIOU, loss))

            results.miou.append(round(mIOU, 6))
            results.pix_acc.append(round(pixAcc, 6))


            # save the best model
            is_best = False
            new_pred = (round(mIOU, 6), round(pixAcc, 6))
            if sum(new_pred) > sum(self.best_pred):
                is_best = True
                self.best_pred = new_pred
                best_state_dict = self.model.module.state_dict() if self.multi_gpu else self.model.state_dict()
            # utils.save_checkpoint({'epoch': epoch + 1,
            #                        'state_dict': self.model.module.state_dict() if self.multi_gpu else self.model.state_dict(),
            #                        'optimizer': self.optimizer.state_dict(),
            #                        'best_pred': self.best_pred}, self.args, is_best)

        final_miou, final_pix_acc = sum(results.miou[-5:]) / 5, sum(results.pix_acc[-5:]) / 5

        final_result = '\nPerformance of last 5 epochs\n[mIoU]: %4f\n[Pixel_Acc]: %4f\n[Best Pred]: %s\n' % (
        final_miou, final_pix_acc,self.best_pred)
        print(final_result)

        # Export weights if needed
        nyu_flag = (self.args.dataset == 'nyud' and final_miou > 0.49)
        sun_flag = (self.args.dataset == 'sunrgbd' and final_miou > 0.47)
        print(nyu_flag, sun_flag, self.args.dataset, round(final_miou, 2))
        if self.args.export or nyu_flag or sun_flag:

            export_info = '_'.join(sys.argv[1:-1] + [str(int(time.time()))])
            torch.save(best_state_dict, os.path.join(SMY_PATH, export_info + '.pth'))
            print('Exported as %s.pth' % export_info)

    # @classmethod
    # def get_class_colors(*args):
    #     def uint82bin(n, count=8):
    #         """returns the binary of integer n, count refers to amount of bits"""
    #         return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])
    #
    #     N = 41
    #     cmap = np.zeros((N, 3), dtype=np.uint8)
    #     for i in range(N):
    #         r, g, b = 0, 0, 0
    #         id = i
    #         for j in range(7):
    #             str_id = uint82bin(id)
    #             r = r ^ (np.uint8(str_id[-1]) << (7 - j))
    #             g = g ^ (np.uint8(str_id[-2]) << (7 - j))
    #             b = b ^ (np.uint8(str_id[-3]) << (7 - j))
    #             id = id >> 3
    #         cmap[i, 0] = r
    #         cmap[i, 1] = g
    #         cmap[i, 2] = b
    #     class_colors = cmap.tolist()
    #     return class_colors


    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, dep, target):
            # model, image, target already moved to gpus
            pred = model(image, dep)        #【6，3，480，480】
            # print("image:",image.shape)
            loss = self.criterion(*pred, target)
            # print("target:",target.shape)
            correct, labeled = utils.batch_pix_accuracy(pred[0].data, target)
            inter, union = utils.batch_intersection_union(pred[0].data, target, self.nclass)

            # save colored result
            # _, predict = torch.max(pred[0].data, 1)
            # predict = predict.cpu().numpy()
            # target = target.cpu().numpy()
            # # print("predict:",predict.shape)
            # result_pred = predict[0]        #[480,480]
            # result_target = target[0]       #[480,480]

            # print("result_pred:", result_pred.shape)
            # print("result_target:", result_target.shape)

            # result_pred_color = utils.color_label_eval(result_pred)
            # result_target_color = utils.color_label_eval(result_target)
            #
            # result_img = Image.fromarray(result_pred_color.astype(np.uint8), mode='RGB')
            # result_label = Image.fromarray(result_target_color.astype(np.uint8),mode='RGB')
            #
            # timestamp = int(time.time())
            # fn = f"predicted_image_{timestamp}.png"
            # result_img.save(os.path.join('pred_result_color', fn))
            # result_label.save(os.path.join('target_color', fn))
            # save raw result
            # cv2.imwrite(os.path.join('pred_view', fn), result_pred)


            return correct, labeled, inter, union, loss


        self.model.eval()
        total_inter, total_union, total_correct, total_label, total_loss = 0, 0, 0, 0, 0
        for i, (image, dep, target) in enumerate(self.valloader):

            image, dep, target = image.to(self.device), dep.to(self.device), target.to(self.device)

            with torch.no_grad():
                correct, labeled, inter, union, loss = eval_batch(self.model, image, dep, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            total_loss += loss.item()
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IOU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIOU = IOU.mean()

            if i % self.val_step == 0:
                print('eval mean IOU {}'.format(mIOU))
            loss = total_loss / len(self.valloader)

            self.writer.add_scalar("mean_iou/val", mIOU, epoch)
            self.writer.add_scalar("pixel accuracy/val", pixAcc, epoch)

        return pixAcc, mIOU, loss


if __name__ == "__main__":
    start_time = time.time()
    print('[args]:', sys.argv)
    print("------- program start ----------")
    # configuration
    args = Dict(get_config('nyud', 'pcgf_pc'))
    args.training.cuda = (args.training.use_cuda and torch.cuda.is_available())
    args.training.resume = None if args.training.resume == 'None' else args.training.resume
    torch.manual_seed(args.training.seed)
    if len(sys.argv) > 0:
        print('debugging mode...')
        args.training.workers = 1
        args.training.batch_size = 4
        args.training.test_batch_size = 4

    trainer = Trainer(args)
    print('Total Epoches:', trainer.config.training.epochs)
    trainer.train_n_evaluate()

    exp_time_mins = int(time.time() - start_time) // 60
    print('[Time]: %.2fh' % (exp_time_mins / 60))
