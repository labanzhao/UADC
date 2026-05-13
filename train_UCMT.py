import os
import sys
import json
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import time
import argparse
import copy
import cv2

import pandas as pd
import matplotlib.pyplot as plt
from easydict import EasyDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Normalize, ToPILImage
from data.dataset import ISICDataset


from data.split import *

import csv
from typing import List, Tuple

from models import deeplabv3
from utils.loss_functions import DSCLoss
from utils.logger import logger as logging
from utils.utils import *
from utils.mask_generator import BoxMaskGenerator, AddMaskParamsToBatch, SegCollate
from utils.ramps import sigmoid_rampup
from utils.torch_utils import seed_torch
from utils.model_init import init_weight

from dataset.dct_transform import *
from copy import deepcopy


import pprint
#from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, AdamW
import yaml

from dataset.semi_dct import SemiDatasetDCT
from dataset.semi import SemiDataset
from model.semseg.segmentor import Segmentor
from supervised import evaluate
from supervised_dct import evaluate as evaluate_dct
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log
from util.dist_helper import setup_distributed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--project', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/runs/UCMT', help='project path for saving results')
    parser.add_argument('--backbone', type=str, default='DeepLabv3p', choices=['DeepLabv3p', 'UNet'], help='segmentation backbone')
    parser.add_argument('--image_root', type=str, default='YOUR_DATA_PATH', help='path_to_image')
    parser.add_argument('--gt_root', type=str, default='YOUR_DATA_PATH', help='path_to_GroundTruth')
    parser.add_argument('--image_size', type=int, default=256, help='the size of images for training and testing')
    parser.add_argument('--labeled_percentage', type=float, default=0.30, help='the percentage of labeled data')
    parser.add_argument('--is_cutmix', type=bool, default=False, help='cut mix')
    parser.add_argument('--mix_prob', type=float, default=0.5, help='probability for amplitude mix')            #？
    parser.add_argument('--topk', type=float, default=2, help='top k')
    parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='number of inputs per batch')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers to use for dataloader')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--num_classes', type=int, default=2, help='number of target categories')
    parser.add_argument('--pretrained', type=bool, default=True, help='using pretrained weights')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')#################################################
    parser.add_argument('--intra_weights', type=list, default=[1., 1.], help='inter classes weighted coefficients in the loss function')
    parser.add_argument('--inter_weight', type=float, default=1., help='inter losses weighted coefficients in the loss function')
    parser.add_argument('--log_freq', type=float, default=10, help='logging frequency of metrics accord to the current iteration')  #？？
    # parser.add_argument('--config', type=str, default="/home/li/桌面/UCMT-main/configs/pascal/segformerb2_4x4.yaml", required=True)
    # parser.add_argument('--config2', type=str, default="/home/li/桌面/UCMT-main/configs/pascal/r50_dct_4x4.yaml", required=True)
    # parser.add_argument('--config3', type=str, default="/home/li/桌面/UCMT-main/configs/pascal/r50_4x4.yaml",required=True)
    parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)
    parser.add_argument('--thr', default=0.95, type=float)
    parser.add_argument('--uw', default=1.0, type=float)
    parser.add_argument('--amp', action="store_true")
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args
    #parser.add_argument('--config3', type=str, required=True)
    #parser.add_argument('--labeled-id-path', type=str, required=True)
    #parser.add_argument('--unlabeled-id-path', type=str, required=True)
    #parser.add_argument('--save-path', type=str, required=True)


def get_dataset_indices(dataset):
    return [i for i in range(len(dataset))]

def save_indices_to_csv(indices, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(indices)

def select_random_images(image_root, output_json_15, output_json_30):
    """
    从指定文件夹中随机选择一定比例的PNG图像，并将图像文件名保存到JSON文件中。

    参数:
    image_root (str): 包含PNG图像的文件夹路径。
    output_json (str): 保存选定图像文件名的JSON文件路径。
    percentage (float): 要选择的图像的比例，默认为0.15（即15%）。
    """
    # 获取文件夹中所有PNG图像的文件名
    image_files = [f for f in os.listdir(image_root) if f.endswith('.png')]

    # 如果文件夹中没有PNG图像，则返回一个空列表
    if not image_files:
        return []

    # 随机选择指定比例的图像
    selected_indices_30 = np.random.choice(len(image_files), size=int(0.3 * len(image_files)), replace=False)
    half_size = int(0.5 * len(selected_indices_30))
    selected_indices_15 = np.random.choice(selected_indices_30, size=half_size, replace=False)


    # 获取被选中图像的文件名
    selected_image_names_30 = [os.path.join(image_root, image_files[i]) for i in selected_indices_30]
    selected_image_names_15 = [os.path.join(image_root, image_files[i]) for i in selected_indices_15]

    # 将选定的图像文件名保存到JSON文件中
    with open(output_json_30, 'w') as f:
        json.dump(selected_image_names_30, f)
    with open(output_json_15, 'w') as f:
        json.dump(selected_image_names_15, f)






def get_data(args):
    # select_random_images(image_root = '/home/li/桌面/UCMT-main/dataset/TrainDataset_cvc_kvasir/image/', output_json_15 = '/home/li/桌面/UCMT-main/dataset/split_15.json', output_json_30 = '/home/li/桌面/UCMT-main/dataset/split_30.json')
    # dataset_unlabeled = PolypDataset(args.image_root, args.gt_root, trainsize=352, split_file_15 = None, split_file_30 = None)#
    # dataset_labeled_15 = PolypDataset(args.image_root, args.gt_root, trainsize=352, split_file_15 = '/home/li/桌面/UCMT-main/dataset/split_15.json', split_file_30 = None)
    # dataset_labeled_30 = PolypDataset(args.image_root, args.gt_root, trainsize=352, split_file_15 = None, split_file_30 = '/home/li/桌面/UCMT-main/dataset/split_30.json')

    train_labeled15_loader = get_loader(args.image_root, args.gt_root, batchsize=args.batch_size, trainsize=args.image_size, stage='train', split_file_15 = '/home/li/桌面/UCMT-main/dataset/split_15.json', split_file_30 = None, is_augmentation=True)
    train_labeled30_loader = get_loader(args.image_root, args.gt_root, batchsize=args.batch_size, trainsize=args.image_size, stage='train', split_file_15 = None, split_file_30 = '/home/li/桌面/UCMT-main/dataset/split_30.json', is_augmentation=True)
    train_unlabeled_loader = get_loader(args.image_root, args.gt_root, batchsize=args.batch_size, trainsize=args.image_size, stage='train', split_file_15=None, split_file_30=None, is_augmentation=True)
    return train_unlabeled_loader, train_labeled15_loader, train_labeled30_loader


def main(is_debug=False):
    args = get_args()
    #print(args.project)
    seed_torch(args.seed)
    # Project Saving Path
    project_path = args.project + '_{}_label_{}/'.format(args.backbone, args.labeled_percentage)
    #print(project_path)
    ensure_dir(project_path)
    save_path = project_path + 'weights/'                                                                               #模型保存路径
    ensure_dir(save_path)

    # Tensorboard & Statistics Results & Logger
    tb_dir = project_path + '/tensorboard{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    writer = SummaryWriter(tb_dir)                                                                                      #SummaryWriter类将条目写入log_dir中的事件文件，以供TensorBoard使用
    metrics = EasyDict()
    metrics.train_loss = []
    metrics.train_loss2 = []
    metrics.val_loss = []
    logger = logging(project_path + 'train_val.log')
    logger.info('PyTorch Version {}\n Experiment{}'.format(torch.__version__, project_path))

    # Load Data
    # if args.is_cutmix:
    #     train_unlabeled_dataloader, train_labeled15_dataloader, train_labeled30_dataloader, aux_loader = get_data(args=args)
    # else:
    #     train_unlabeled_dataloader, train_labeled15_dataloader, train_labeled30_dataloader = get_data(args=args)
    train_unlabeled_dataloader, train_labeled15_dataloader, train_labeled30_dataloader = get_data(args=args)
    iters = len(train_unlabeled_dataloader)           #一共1815张图像，一个batch 4张，iters=454
    #print("iters:",iters)

    # Load Model & EMA
    student1 = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)    #.__dict__字典替换？
    student2 = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
    teacher = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
    init_weight(student1.net.classifier, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')                                     #对classifier中进行kaiming初始化
    init_weight(student2.net.classifier, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')
    init_weight(teacher.net.classifier, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')
    teacher.detach_model()
    # best_model_wts = copy.deepcopy(teacher.state_dict())
    h, w = args.image_size // 16, args.image_size // 16
    s = h
    unfolds  = torch.nn.Unfold(kernel_size=(h, w), stride=s).to(device)                                                 #？？？？？？？？？？？？？？？？？
    folds = torch.nn.Fold(output_size=(args.image_size, args.image_size), kernel_size=(h, w), stride=s).to(device)      #fold,unfold这两个的作用是什么？
    best_epoch = 0
    best_loss = 100
    alpha = 0.1
    total_iters = iters * args.num_epochs

    # Criterion & Optimizer & LR Schedule
    criterion = DSCLoss(num_classes=args.num_classes, intra_weights=args.intra_weights, inter_weight=args.inter_weight, device=device)
    criterion_u = DSCLoss(num_classes=args.num_classes, intra_weights=args.intra_weights, inter_weight=args.inter_weight, device=device)
    criterion_c = DSCLoss(num_classes=args.num_classes, intra_weights=args.intra_weights, inter_weight=args.inter_weight, device=device)
    optimizer1 = optim.AdamW(student1.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    optimizer2 = optim.AdamW(student2.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))





    # cudnn.enabled = True                                                                                                #使用非确定性算法###################################################
    # cudnn.benchmark = True



###############################################################################################################################################

    # Train
    since = time.time()
    logger.info('start training')
    for epoch in range(1, args.num_epochs + 1):
        epoch_metrics = EasyDict()                  #编辑字典，可以以属性的方式访问字典
        epoch_metrics.train_loss = []
        epoch_metrics.train_loss2 = []
        if is_debug:
            pbar = range(10)
        else:
            pbar = range(iters)
        iter_train_labeled_dataloader = iter(train_labeled30_dataloader)
        iter_train_unlabeled_dataloader = iter(train_unlabeled_dataloader)

        ############################
        # Train
        ############################
        student1.train()        #训练模式
        student2.train()
        teacher.train()
        for idx in pbar:
            image, label, imageA1, imageA2 = next(iter_train_labeled_dataloader)        #imageA1, imageA2是image augmentation，返回一个batch的数据
            image, label = image.to(device), label.to(device)
            imageA1, imageA2 = imageA1.to(device), imageA2.to(device)
            uimage, uimageA1, uimageA2 = next(iter_train_unlabeled_dataloader)
            uimage = uimage.to(device)
            uimageA1, uimageA2 = uimageA1.to(device), uimageA2.to(device)

            '''
            Step 1
            '''
            # print(optimizer1)
            # return 0
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            ###########################
                # supervised path #
            ###########################
            pred1s = student1(image)
            # print(pred1s)
            # print("#############")
            pred2s = student2(image)
            preds = teacher(image)
            pred1 = pred1s['out']                                                               #??字典？？预测值
            # print(pred1)
            # print("#############")
            pred2 = pred2s['out']
            pred = preds['out']
            pred1_feature = torch.softmax(pred1, dim=1)
            pred2_feature = torch.softmax(pred2, dim=1)
            pred_feature = torch.softmax(pred, dim=1)
            loss1_sup = criterion(pred1, label.squeeze(1).long())                                                       #[4,2,256,256]  [4,256,256]
            loss2_sup = criterion(pred2, label.squeeze(1).long())
            loss_sup = loss1_sup + loss2_sup

            ###########################
                # unsupervised path #                       #预测伪标签？
            ###########################
            # Estimate the pseudo-labels
            pred1s_u = student1(uimageA1)
            pred2s_u = student2(uimageA2)
            preds_u = teacher(uimage)
            pred1_u = pred1s_u['out']                                                               #这个out到底是什么，为什么有两个out？
            pred2_u = pred2s_u['out']
            pred_u = preds_u['out']
            pred1_u_feature = torch.softmax(pred1_u, dim=1)
            pseudo1 = torch.argmax(pred1_u_feature, dim=1)                                                              #student1预测结果（使用了ueimageA1）（获得每行中最大的数值的位置）
            pred2_u_feature = torch.softmax(pred2_u, dim=1)
            pseudo2 = torch.argmax(pred2_u_feature, dim=1)                                                              #student2预测结果(使用了ueimageA2)
            pred_u_feature = torch.softmax(pred_u, dim=1)
            pseudo = torch.argmax(pred_u_feature, dim=1)                                                                #teacher预测结果(使用了ueimage)

            # CMT loss
            loss1_cmt = criterion_c(pred1_u, pseudo.detach())                           #与教师模型的伪标签损失
            loss2_cmt = criterion_c(pred2_u, pseudo.detach())
            loss_cmt = (loss1_cmt + loss2_cmt) * 0.5

            # CPS loss
            loss1_u = criterion_u(pred1_u, pseudo2.detach())                            #与另一个学生伪标签的交叉损失
            loss2_u = criterion_u(pred2_u, pseudo1.detach())
            loss_cps = (loss1_u + loss2_u) * 0.5
            loss_u = (loss_cps + loss_cmt) * alpha                                                                      #整体的loss_u结合了CMT loss与CPS loss
            lambda_ = sigmoid_rampup(current=idx + len(pbar) * (epoch-1), rampup_length=len(pbar)*5)
            loss = loss_sup + lambda_ * loss_u                                                                          #总损失
            loss.backward()                                                                                             #损失回传
            optimizer1.step()                                                                                           #每个mini-batch更新一次🧑‍🎓
            optimizer2.step()
            teacher.weighted_update(student1, student2, ema_decay=0.99, cur_step=idx + len(pbar) * (epoch-1)) #EMA更新

            writer.add_scalar('train_sup_loss', loss_sup.item(), idx + len(pbar) * (epoch-1))   #写入log tensorboard?
            writer.add_scalar('train_cps_loss', loss_cps.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_cmt_loss', loss_cmt.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_loss', loss.item(), idx + len(pbar) * (epoch-1))
            if idx % args.log_freq == 0:
                logger.info("Train1: Epoch/Epochs {}/{}\t"
                            "iter/iters {}/{}\t"
                            "loss {:.3f}, loss_sup {:.3f}, loss_cps {:.3f}, loss_cmt {:.3f}, lambda {}".format(epoch, args.num_epochs, idx, len(pbar),
                                                                                  loss.item(), loss_sup.item(), loss_cps.item(), loss_cmt.item(), lambda_))
            epoch_metrics.train_loss.append(loss.item())

            '''
            Step 2
            '''
            # dct_transform = DCTTransform()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # optimizer11.zero_grad()
            # optimizer22.zero_grad()
            topk = args.topk
            ###########################
                # supervised path #
            ###########################
            #Estimate the uncertainty map
            with torch.no_grad():
                uncertainty_map11 = torch.mean(torch.stack([pred1_feature, pred_feature]), dim=0)                       #[4 2 256 256]
                uncertainty_map11 = -1.0 * torch.sum(uncertainty_map11*torch.log(uncertainty_map11 + 1e-6), dim=1, keepdim=True)#[4 1 256 256]
                uncertainty_map22 = torch.mean(torch.stack([pred2_feature, pred_feature]), dim=0)
                uncertainty_map22 = -1.0 * torch.sum(uncertainty_map22*torch.log(uncertainty_map22 + 1e-6), dim=1, keepdim=True)

                B, C = image.shape[0], image.shape[1]                                                                   #batch channel 4 3
                # for student 1
                x11 = unfolds(uncertainty_map11)                                                                        # B x C*kernel_size[0]*kernel_size[1] x L [4 256 256] (这里的C是1,与上面image的C不一样)    [4 256 256]将每个滑动窗口内的数据列成单独一列
                #print(x11.size())
                x11 = x11.view(B, 1, h, w, -1)                                                                          # B x C x h x w x L [4 1 16 16 256]
                #print(x11.size())
                x11_mean = torch.mean(x11, dim=(1, 2, 3))                                                               # B x L  [4 256]
                _, x11_max_index = torch.sort(x11_mean, dim=1, descending=True)                                         # B x L B x L [4 256] 排序结果   结果对应的数字在原来tensor中的index
                # for student 2
                x22 = unfolds(uncertainty_map22)  # B x C*kernel_size[0]*kernel_size[1] x L  (这里的C是1,与上面image的C不一样)    [4 256 256]将每个滑动窗口内的数据列成单独一列
                x22 = x22.view(B, 1, h, w, -1)  # B x C x h x w x L (这里的C是1,与上面image的C不一样) [4 1 16 16 256]
                x22_mean = torch.mean(x22, dim=(1, 2, 3))  # B x L [4 256]
                _, x22_max_index = torch.sort(x22_mean, dim=1, descending=True)  # B x L B x L [4 256]
                img_unfold = unfolds(imageA1).view(B, C, h, w, -1)  # B x C x h x w x L
                lab_unfold = unfolds(label.float()).view(B, 1, h, w, -1)  # B x C x h x w x L
                for i in range(B):
                    img_unfold[i, :, :, :, x11_max_index[i, :topk]] = img_unfold[i, :, :, :, x22_max_index[i, -topk:]]
                    img_unfold[i, :, :, :, x22_max_index[i, :topk]] = img_unfold[i, :, :, :, x11_max_index[i, -topk:]]
                    lab_unfold[i, :, :, :, x11_max_index[i, :topk]] = lab_unfold[i, :, :, :, x22_max_index[i, -topk:]]
                    lab_unfold[i, :, :, :, x22_max_index[i, :topk]] = lab_unfold[i, :, :, :, x11_max_index[i, -topk:]]
                image2 = folds(img_unfold.view(B, C*h*w, -1))        #[4 3 256 256]                                                   #image2是交换后的图像？
                label2 = folds(lab_unfold.view(B, 1*h*w, -1))       #[4 1 256 256]
            pred1s = student1(image2)
            pred2s = student2(image2)
            pred1 = pred1s['out']   #[4 2 256 256]
            pred2 = pred2s['out']

            # print(label2.squeeze(1).long().shape)
            # print("####################################")
            # return 0
            loss1_sup = criterion(pred1, label2.squeeze(1).long())

            loss2_sup = criterion(pred2, label2.squeeze(1).long())

            loss_sup = loss1_sup + loss2_sup

            ###########################
                # unsupervised path #
            ###########################
            # Estimate the uncertainty map
            with torch.no_grad():
                uncertainty_map1 = torch.mean(torch.stack([pred1_u_feature, pred_u_feature]), dim=0)
                uncertainty_map1 = -1.0 * torch.sum(uncertainty_map1*torch.log(uncertainty_map1 + 1e-6), dim=1, keepdim=True)
                uncertainty_map2 = torch.mean(torch.stack([pred2_u_feature, pred_u_feature]), dim=0)
                uncertainty_map2 = -1.0 * torch.sum(uncertainty_map2*torch.log(uncertainty_map2 + 1e-6), dim=1, keepdim=True)

                B, C = uimage.shape[0], uimage.shape[1]
                # for student 1
                x1 = unfolds(uncertainty_map1)  # B x C*kernel_size[0]*kernel_size[1] x L
                x1 = x1.view(B, 1, h, w, -1)  # B x C x h x w x L
                x1_mean = torch.mean(x1, dim=(1, 2, 3))  # B x L
                _, x1_max_index = torch.sort(x1_mean, dim=1, descending=True)  # B x L B x L
                # for student 2
                x2 = unfolds(uncertainty_map2)  # B x C*kernel_size[0]*kernel_size[1] x L
                x2 = x2.view(B, 1, h, w, -1)  # B x C x h x w x L
                x2_mean = torch.mean(x2, dim=(1, 2, 3))  # B x L
                _, x2_max_index = torch.sort(x2_mean, dim=1, descending=True)  # B x L B x L
                imgu_unfold = unfolds(uimageA1).view(B, C, h, w, -1)  # B x C x h x w x L
                pseudo_unfold = unfolds(pseudo.unsqueeze(1).float()).view(B, 1, h, w, -1)  # B x C x h x w x
                for i in range(B):
                    imgu_unfold[i, :, :, :, x1_max_index[i, :topk]] = imgu_unfold[i, :, :, :, x2_max_index[i, -topk:]]
                    imgu_unfold[i, :, :, :, x2_max_index[i, :topk]] = imgu_unfold[i, :, :, :, x1_max_index[i, -topk:]]
                    pseudo_unfold[i, :, :, :, x1_max_index[i, :topk]] = pseudo_unfold[i, :, :, :, x2_max_index[i, -topk:]]
                    pseudo_unfold[i, :, :, :, x2_max_index[i, :topk]] = pseudo_unfold[i, :, :, :, x1_max_index[i, -topk:]]
                uimage2 = folds(imgu_unfold.view(B, C * h * w, -1))
                pseudo = folds(pseudo_unfold.view(B, 1 * h * w, -1)).squeeze(1).long()


            #无监督的路径下得到uimage2，pseudo#######################################################################################
            # Re-Estimate the pseudo-labels on the new uimages
            pred1_u = student1(uimage2)['out']
            pred2_u = student2(uimage2)['out']


            pseudo1 = torch.softmax(pred1_u, dim=1)
            pseudo1 = torch.argmax(pseudo1, dim=1)
            pseudo2 = torch.softmax(pred2_u, dim=1)
            pseudo2 = torch.argmax(pseudo2, dim=1)

            # CMT loss
            loss1_cmt = criterion_c(pred1_u, pseudo.detach())
            loss2_cmt = criterion_c(pred2_u, pseudo.detach())
            loss_cmt = (loss1_cmt + loss2_cmt ) * 0.5

            # CPS loss
            loss1_u = criterion_u(pred1_u, pseudo2.detach())
            loss2_u = criterion_u(pred2_u, pseudo1.detach())
            loss_cps = (loss1_u + loss2_u) * 0.5
            #torch.distributed.barrier()
            loss_u = (loss_cps + loss_cmt) * alpha
            lambda_ = sigmoid_rampup(current=idx + len(pbar) * (epoch - 1), rampup_length=len(pbar) * 5)
            loss = loss_sup + lambda_ * loss_u
            loss.backward()
            optimizer1.step()
            optimizer2.step()


            teacher.weighted_update(student1, student2, ema_decay=0.99, cur_step=idx + len(pbar) * (epoch - 1))

            writer.add_scalar('train_sup_loss', loss_sup.item(), idx + len(pbar) * (epoch - 1))
            writer.add_scalar('train_cps_loss', loss_cps.item(), idx + len(pbar) * (epoch - 1))
            writer.add_scalar('train_cmt_loss', loss_cmt.item(), idx + len(pbar) * (epoch - 1))
            writer.add_scalar('train_loss', loss.item(), idx + len(pbar) * (epoch - 1))
            if idx % args.log_freq == 0:#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                logger.info("Train2: Epoch/Epochs {}/{}\t"
                            "iter/iters {}/{}\t"
                            "loss {:.3f}, loss_sup {:.3f}, loss_cps {:.3f}, loss_cmt {:.3f}, lambda {}".format(
                    epoch, args.num_epochs, idx, len(pbar),
                    loss.item(), loss_sup.item(), loss_cps.item(), loss_cmt.item(), lambda_))
            # if idx % args.log_freq == 0:
            #     logger.info("Train2: Epoch/Epochs {}/{}\t"
            #                 "iter/iters {}/{}\t"
            #                 "loss {:.3f}, loss_sup {:.3f}, loss_cps2 {:.3f}, lambda {}".format(
            #         epoch, args.num_epochs, idx, len(pbar),
            #         loss.item(), loss_sup.item(), loss_cps.item(), lambda_))
            epoch_metrics.train_loss2.append(loss.item())

        metrics.train_loss.append(np.mean(epoch_metrics.train_loss))
        metrics.train_loss2.append(np.mean(epoch_metrics.train_loss2))
        logger.info("Average: Epoch/Epoches {}/{}\t"#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                    "train1 epoch loss {:.3f}\t"
                    "train2 epoch loss {:.3f}\n".format(epoch, args.num_epochs,
                                                        np.mean(epoch_metrics.train_loss),
                                                        np.mean(epoch_metrics.train_loss2), ))
        # logger.info("Average: Epoch/Epoches {}/{}\t"
        #             "train2 epoch loss {:.3f}\n".format(epoch, args.num_epochs,
        #                                                 np.mean(epoch_metrics.train_loss2), ))
        if np.mean(epoch_metrics.train_loss2) <= best_loss:                                                             #保存的是teacher模型
            best_loss = np.mean(epoch_metrics.train_loss2)
            torch.save(teacher.state_dict(), save_path + 'best.pth'.format(best_epoch))#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            torch.save(student1.state_dict(), save_path + 'student1_best.pth'.format(best_epoch))
            torch.save(student2.state_dict(), save_path + 'student2_best.pth'.format(best_epoch))


        torch.save(teacher.state_dict(), save_path + 'last.pth'.format(best_epoch))#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        torch.save(student1.state_dict(), save_path + 'student1_last.pth'.format(best_epoch))
        torch.save(student2.state_dict(), save_path + 'student2_last.pth'.format(best_epoch))
    ############################
    # Save Metrics
    ############################
    data_frame = pd.DataFrame(
        data={'loss': metrics.train_loss,
                'loss2': metrics.train_loss2},
        index=range(1, args.num_epochs + 1))#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    # data_frame = pd.DataFrame(
    #     data={'loss2': metrics.train_loss2},
    #     index=range(1, args.num_epochs + 1))
    data_frame.to_csv(project_path + 'train_loss.csv', index_label='Epoch')
    plt.figure()
    plt.title("Loss")
    plt.plot(metrics.train_loss, label="Train")#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    plt.plot(metrics.train_loss2, label="Train2")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(project_path + 'train_loss.png')

    time_elapsed = time.time() - since
    logger.info('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info(project_path)
    logger.info('TRAINING FINISHED!')
if __name__ == '__main__':
    main()
