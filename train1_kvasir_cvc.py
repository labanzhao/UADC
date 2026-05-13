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
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='number of inputs per batch')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers to use for dataloader')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--num_classes', type=int, default=2, help='number of target categories')
    parser.add_argument('--pretrained', type=bool, default=True, help='using pretrained weights')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')#################################################
    parser.add_argument('--intra_weights', type=list, default=[1., 1.], help='inter classes weighted coefficients in the loss function')
    parser.add_argument('--inter_weight', type=float, default=1., help='inter losses weighted coefficients in the loss function')
    parser.add_argument('--log_freq', type=float, default=10, help='logging frequency of metrics accord to the current iteration')  #？？
    parser.add_argument('--config', type=str, default="/home/li/桌面/UCMT-main/configs/pascal/segformerb2_4x4.yaml", required=True)
    parser.add_argument('--config2', type=str, default="/home/li/桌面/UCMT-main/configs/pascal/r50_dct_4x4.yaml", required=True)
    parser.add_argument('--config3', type=str, default="/home/li/桌面/UCMT-main/configs/pascal/r50_4x4.yaml",required=True)
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

    train_labeled15_loader = get_loader(args.image_root, args.gt_root, batchsize=args.batch_size, trainsize=args.image_size, stage = 'train', split_file_15 = '/home/li/桌面/UCMT-main/dataset/split_15.json', split_file_30 = None, is_augmentation=True)
    train_labeled30_loader = get_loader(args.image_root, args.gt_root, batchsize=args.batch_size, trainsize=args.image_size, stage = 'train', split_file_15 = None, split_file_30 = '/home/li/桌面/UCMT-main/dataset/split_30.json', is_augmentation=True)
    train_unlabeled_loader = get_loader(args.image_root, args.gt_root, batchsize=args.batch_size, trainsize=args.image_size, stage = 'train', split_file_15=None, split_file_30=None, is_augmentation=True)
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
    if args.is_cutmix:
        train_unlabeled_dataloader, train_labeled15_dataloader, train_labeled30_dataloader, aux_loader = get_data(args=args)
    else:
        # train_labeled_dataloader, train_unlabeled_dataloader = get_data(args=args)
        train_unlabeled_dataloader, train_labeled15_dataloader, train_labeled30_dataloader= get_data(args=args)
    iters = len(train_unlabeled_dataloader)           #
    print("iters:",iters)
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2")
    # return 0
    # Load Model & EMA
    # student1 = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)    #.__dict__字典替换？
    # #print(student1)
    # student2 = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
    # teacher = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
    #student3=deeplabv3.to(device)
    #print(1111)
    #print(student1.net)
    #print(student1.net.classifier)
    #print(isinstance(student1.net.classifier, tuple))
    # for name, m in student1.net.classifier.named_modules():
    #     print(name)
    #     print("---------------------------")
    #     print(m)
    #     print("==================================")
    # print(student1.net.classifier.named_modules())
    # print(11111111111111111111)
    # init_weight(student1.net.classifier, nn.init.kaiming_normal_,
    #             nn.BatchNorm2d, 1e-5, 0.1,
    #             mode='fan_in', nonlinearity='relu')                                     #对classifier中进行kaiming初始化
    # init_weight(student2.net.classifier, nn.init.kaiming_normal_,
    #             nn.BatchNorm2d, 1e-5, 0.1,
    #             mode='fan_in', nonlinearity='relu')
    # init_weight(teacher.net.classifier, nn.init.kaiming_normal_,
    #             nn.BatchNorm2d, 1e-5, 0.1,
    #             mode='fan_in', nonlinearity='relu')
    # teacher.detach_model()
    # best_model_wts = copy.deepcopy(teacher.state_dict())
    h, w = args.image_size // 16, args.image_size // 16
    s = h
    unfolds  = torch.nn.Unfold(kernel_size=(h, w), stride=s).to(device)                                                 #对于PolypDataset设置的是多少？？？？？？？？？？？
    folds = torch.nn.Fold(output_size=(args.image_size, args.image_size), kernel_size=(h, w), stride=s).to(device)
    best_epoch = 0
    best_loss = 100
    alpha = 1.0#################################################################################################################################################################
    total_iters = iters * args.num_epochs

    # Criterion & Optimizer & LR Schedule
    criterion = DSCLoss(num_classes=args.num_classes, intra_weights=args.intra_weights, inter_weight=args.inter_weight, device=device)
    criterion_u = DSCLoss(num_classes=args.num_classes, intra_weights=args.intra_weights, inter_weight=args.inter_weight, device=device)
    criterion_c = DSCLoss(num_classes=args.num_classes, intra_weights=args.intra_weights, inter_weight=args.inter_weight, device=device)
    # optimizer1 = optim.AdamW(student1.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    # optimizer2 = optim.AdamW(student2.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))


    #diverse cotraining##################################################################################################################################################
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg2 = yaml.load(open(args.config2, "r"), Loader=yaml.Loader)
    cfg3 = yaml.load(open(args.config3, "r"), Loader=yaml.Loader)
    cfg['conf_thresh'] = args.thr
    #logger = init_log('global', logging.INFO)
    #logger.propagate = 0

#    rank, word_size = setup_distributed(port=args.port)



    # if rank == 0:
    #     all_args = {**cfg, **vars(args), 'ngpus': word_size}
    #     logger.info('{}\n'.format(pprint.pformat(all_args)))
    # all_args = {**cfg, **vars(args), 'ngpus': word_size}
    # logger.info('{}\n'.format(pprint.pformat(all_args)))
    # if rank == 0:
    #     os.makedirs(args.save_path, exist_ok=True)
    # os.makedirs(args.save_path, exist_ok=True)



    # cudnn.enabled = True                                                                                                #使用非确定性算法
    # cudnn.benchmark = True
    entropies1 = []
    entropies2 = []
    entropies3 = []
    model1 = Segmentor(cfg).to(device)
    model2 = Segmentor(cfg2).to(device)
    model3 = Segmentor(cfg3).to(device)
    # if rank == 0:
    #     logger.info('Total params: {:.1f}M\n'.format(count_params(model1)))

    param_groups1 = [{'params': model1.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model1.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}]
    param_groups2 = [{'params': model2.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model2.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}]
    param_groups3 = [{'params': model3.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model3.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}]

    if cfg["optim"] == "SGD":
        optimizer11 = SGD(param_groups1, lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    elif cfg["optim"] == "AdamW":
        # optimizer11 = AdamW(param_groups1, lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.999))
        optimizer11 = AdamW(model1.parameters(), lr=cfg['lr'], betas=(0.9, 0.999))
    if cfg2["optim"] == "SGD":
        optimizer22 = SGD(param_groups2, lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    elif cfg2["optim"] == "AdamW":
        # optimizer22 = AdamW(param_groups2, lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.999))
        optimizer22 = AdamW(model2.parameters(), lr=cfg['lr'], betas=(0.9, 0.999))
    if cfg3["optim"] == "SGD":
        optimizer1 = SGD(param_groups3, lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    elif cfg3["optim"] == "AdamW":
        # optimizer1 = AdamW(param_groups3, lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.999))
        optimizer1 = AdamW(model3.parameters(), lr=cfg['lr'], betas=(0.9, 0.999))

    # local_rank = int(os.environ["LOCAL_RANK"])
    # model1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model1).cuda()
    # model2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model2).cuda()
    #model3 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model3).cuda()

    # model1 = torch.nn.parallel.DistributedDataParallel(model1, device_ids=[local_rank],
    #                                                    output_device=local_rank, find_unused_parameters=False)
    # model2 = torch.nn.parallel.DistributedDataParallel(model2, device_ids=[local_rank],
    #                                                    output_device=local_rank, find_unused_parameters=False)
    # model3 = torch.nn.parallel.DistributedDataParallel(model3, device_ids=[local_rank],
    #                                                    output_device=local_rank, find_unused_parameters=False)
    # if cfg['criterion']['name'] == 'CELoss':
    #     criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    # elif cfg['criterion']['name'] == 'OHEM':
    #     criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    # else:
    #     raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    #criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    # trainset_u = SemiDatasetDCT(cfg['dataset'], cfg['data_root'], 'train_u',
    #                             cfg['crop_size'], args.unlabeled_id_path)
    # trainset_l = SemiDatasetDCT(cfg['dataset'], cfg['data_root'], 'train_l',
    #                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    # valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    # valset_dct = SemiDatasetDCT(cfg['dataset'], cfg['data_root'], 'val')
    #
    # trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)                                        #分布式采样器 一般与数据并行训练一起使用
    # trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
    #                            pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler_l)
    #
    # trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    # trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
    #                            pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler_u)
    # valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    # valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
    #                        drop_last=False, sampler=valsampler)
    # valsampler_dct = torch.utils.data.distributed.DistributedSampler(valset_dct)
    # valloader_dct = DataLoader(valset_dct, batch_size=1, pin_memory=True, num_workers=2,
    #                            drop_last=False, sampler=valsampler_dct)

    # total_iters = len(trainloader_u) * cfg['epochs']
    # previous_best1, previous_best2, previous_best3 = 0.0, 0.0, 0.0
    # scaler1 = torch.cuda.amp.GradScaler()                                                                               #一个用于自动混合精度训练的 PyTorch 工具，它可以帮助加速模型训练并减少显存使用量。具体来说，GradScaler 可以将梯度缩放到较小的范围，以避免数值下溢或溢出的问题，同时保持足够的精度以避免模型的性能下降
    # scaler2 = torch.cuda.amp.GradScaler()
    # scaler3 = torch.cuda.amp.GradScaler()

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
        #print(1,next(iter_train_labeled_dataloader))
        # return 0
        ############################
        # Train
        ############################
        model1.train()
        model2.train()
        model3.train()
        # i=0
        # for idx in pbar:
        #     image, label, imageA1, imageA2 = next(iter_train_labeled_dataloader)
        #     i=i+1
        # print(i)
        # return 0
        for idx in pbar:
            image, label, imageA1, imageA2= next(iter_train_labeled_dataloader)        #imageA1, imageA2是image augmentation，返回一个batch的数据
            # print(image)
            # print(label)
            # print(imageA1)
            # print(imageA2)
            # print("#########################")
            image, label = image.to(device), label.to(device)
            torch.set_printoptions(edgeitems=256)
            # print(label.shape)
            # print("#################################################################")
            # return 0
            # print(image.shape)
            # print(label.shape)
            # return 0
            imageA1, imageA2 = imageA1.to(device), imageA2.to(device)
            uimage, uimageA1, uimageA2= next(iter_train_unlabeled_dataloader)
            uimage = uimage.to(device)
            uimageA1, uimageA2 = uimageA1.to(device), uimageA2.to(device)



#更改部分####################################################################################################################
            #train1
            '''
            Step 1
            '''
            dct_transform = DCTTransform()
            optimizer11.zero_grad()
            optimizer22.zero_grad()
            optimizer1.zero_grad()

            image_dct_expanded_to_concat = []
            j = image.size(0)
            for i in range(j):
                single_image = image[i]  # type torch.tensor   shape [3 256 256]
                # single_image = single_image.numpy()
                pil_image = ToPILImage()(single_image)
                uimage2_single_dct = dct_transform.__call__(deepcopy(pil_image))
                uimage2_dct_expanded = uimage2_single_dct.unsqueeze(0)
                image_dct_expanded_to_concat.append(uimage2_dct_expanded)
            image_dct = torch.cat(image_dct_expanded_to_concat, dim=0)  ############得到dct转换后的tensor uimage2_dct
            image_dct = image_dct.cuda()

            # optimizer2.zero_grad()
            ###########################
            # supervised path #
            ###########################

            # pred1s = student1(image)
            # # print(pred1s)
            # # print("#############")
            # # pred2s = student2(image)
            # # preds = teacher(image)
            # pred1 = pred1s['out']                                                               #??字典？？预测值
            # print(pred1)
            # print("#############")
            # pred2 = pred2s['out']
            # pred = preds['out']
            # pred1_feature = torch.softmax(pred1, dim=1)
            pred11 = model1(image)  ##############################################[4 2 256 256]                                   模型的更新放在哪里?
            pred11_feature = torch.softmax(pred11, dim=1)
            # print(type(label2.squeeze(1).long()))
            # return 0
            pred22 = model2(image_dct, ([256, 256]))
            pred22_feature = torch.softmax(pred22, dim=1)
            loss11_sup = criterion(pred11, label.squeeze(1).long())  ###################################################修改后的loss,损失函数好要用criterion_l？？？？？？？？？？？？？？？
            loss22_sup = criterion(pred22, label.squeeze(1).long())  #################################################修改后的loss,损失函数好要用criterion_l？？？？？？？？？？？？？？？？
            # pred2_feature = torch.softmax(pred2, dim=1)
            # pred_feature = torch.softmax(pred, dim=1)
            pred1 = model3(image)  ##############################################[4 2 256 256]                                   模型的更新放在哪里?
            pred1_feature = torch.softmax(pred1, dim=1)
            loss1_sup = criterion(pred1, label.squeeze(1).long())  # [4,2,256,256]  [4,256,256]
            loss_sup1 = loss11_sup + loss22_sup + loss1_sup*6

            pseudo11_l = torch.argmax(pred11_feature, dim=1)
            pseudo22_l = torch.argmax(pred22_feature, dim=1)
            pseudo1_l = torch.argmax(pred1_feature, dim=1)
            loss11_cps_l1 = (criterion_c(pred11, pseudo22_l.detach()) + criterion_c(pred11, pseudo1_l.detach())) / 2.
            loss22_cps_l1 = (criterion_c(pred22, pseudo11_l.detach()) + criterion_c(pred22, pseudo1_l.detach())) / 2.
            loss1_cps_l1 = (criterion_c(pred1, pseudo11_l.detach()) + criterion_c(pred1, pseudo22_l.detach())) / 2.
            loss_cps_l1 = loss11_cps_l1 + loss22_cps_l1 + loss1_cps_l1
            # pseudo11_l = torch.argmax(pred11_feature, dim=1)#[4,256,256]

            # #########
            # #可视化伪标签与对应图像
            # #########
            # black_white_image_h = (pseudo11_l == 1).float()  # 这将返回一个形状相同的浮点张量，其中1表示白色，0表示黑色
            #
            # # 如果需要，可以将浮点张量缩放到0-255的整数范围
            # black_white_image_h_uint8 = (black_white_image_h * 255).byte()
            # # 选择一个图像进行可视化（如果有多个图像在批量中）
            # image_to_save_h = black_white_image_h_uint8[0]  # 取第一个图像作为示例
            # image_to_save_h = image_to_save_h.cpu()
            # save_folder_h = '/home/li/桌面/UCMT-main/images/h/'  # 确保这个文件夹已经存在
            # image_path_h = save_folder_h + str(idx)+'black_white_image.jpg'
            # # 使用matplotlib保存图像为jpg格式
            # plt.imsave(image_path_h, image_to_save_h, cmap='gray')
            #
            # black_white_image_f2 = (pseudo22_l == 1).float()  # 这将返回一个形状相同的浮点张量，其中1表示白色，0表示黑色
            # # 如果需要，可以将浮点张量缩放到0-255的整数范围
            # black_white_image_f2_uint8 = (black_white_image_f2 * 255).byte()
            # # 选择一个图像进行可视化（如果有多个图像在批量中）
            # image_to_save_f2 = black_white_image_f2_uint8[0]  # 取第一个图像作为示例
            # image_to_save_f2 = image_to_save_f2.cpu()
            # save_folder_f2 = '/home/li/桌面/UCMT-main/images/f2/'  # 确保这个文件夹已经存在        f2是pred22,f3是pred1
            # image_path_f2 = save_folder_f2 + str(idx) + 'black_white_image.jpg'
            # # 使用matplotlib保存图像为jpg格式
            # plt.imsave(image_path_f2, image_to_save_f2, cmap='gray')
            #
            # black_white_image_f3 = (pseudo1_l == 1).float()  # 这将返回一个形状相同的浮点张量，其中1表示白色，0表示黑色
            # # 如果需要，可以将浮点张量缩放到0-255的整数范围
            # black_white_image_f3_uint8 = (black_white_image_f3 * 255).byte()
            # # 选择一个图像进行可视化（如果有多个图像在批量中）
            # image_to_save_f3 = black_white_image_f3_uint8[0]  # 取第一个图像作为示例
            # image_to_save_f3 = image_to_save_f3.cpu()
            # save_folder_f3 = '/home/li/桌面/UCMT-main/images/f3/'  # 确保这个文件夹已经存在
            # image_path_f3 = save_folder_f3 + str(idx) + 'black_white_image.jpg'
            # # 使用matplotlib保存图像为jpg格式
            # plt.imsave(image_path_f3, image_to_save_f3, cmap='gray')
            #
            # image_image = image.cpu()
            # # 如果 image 不是 (C, H, W) 格式，而是 (B, C, H, W)（其中 B 是批大小），则选择一个图像来保存
            # # 例如，选择第一个图像
            # image_image = image_image[0]
            # # 如果图像数据是在 [0, 1] 范围内，并且你想将其转换为 [0, 255] 以保存为JPEG
            # image_image = image_image.numpy() * 255
            # image_image = image_image.astype(np.uint8)
            # # 如果图像是 RGB 图像，确保通道顺序正确
            # # 对于 PyTorch，通常通道顺序是 (C, H, W)，但 PIL 期望的是 (H, W, C)
            # if image_image.shape[0] == 3:
            #     image_image = np.transpose(image_image, (1, 2, 0))
            # # 将 numpy 数组转换为 PIL 图像
            # pil_image = Image.fromarray(image_image)
            # # 指定保存文件的路径和名称
            # save_path = '/home/li/桌面/UCMT-main/images/image/'
            # image_path = save_path + str(idx) + 'image.jpg'
            # # 保存图像为 JPEG 格式
            # pil_image.save(image_path, 'JPEG')

            # loss2_sup = criterion(pred2, label.squeeze(1).long())

            ###########################
            # unsupervised path #                       #预测伪标签？
            ###########################
            # Estimate the pseudo-labels
            # 无cmt的训练################################################################################################################
            uimage_dct_expanded_to_concat = []
            j = uimage.size(0)
            for i in range(j):
                single_image = uimage[i]  # type torch.tensor   shape [3 256 256]
                # single_image = single_image.numpy()
                pil_image = ToPILImage()(single_image)
                uimage2_single_dct = dct_transform.__call__(deepcopy(pil_image))
                uimage2_dct_expanded = uimage2_single_dct.unsqueeze(0)
                uimage_dct_expanded_to_concat.append(uimage2_dct_expanded)
            uimage_dct = torch.cat(uimage_dct_expanded_to_concat, dim=0)  ############得到dct转换后的tensor uimage2_dct
            uimage_dct = uimage_dct.to(device)

            pred11_u = model1(uimageA1)  ##################################################？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？uimageA1
            pred22_u = model2(uimage_dct, ([256, 256]))  #################################################
            pred1_u = model3(uimageA2)  ###############################################################################？？？？？？？？？？？？？？？？？？？？uimageA2

            # pred_u_w1 = pred11_u.detach()  # bs, c, h, w                                                            #无法继续向前传播梯度的副本（model1）
            pred_u_w1 = pred11_u
            pred11_u_feature = pred11_u.softmax(dim=1)
            # print(pred11_u_feature.shape)
            conf_u_w1 = pred11_u.softmax(dim=1).max(dim=1)[0]  # 保留最大值
            mask_u_w1 = pred11_u_feature.argmax(dim=1)  # bs, h, w                                                          #得到最大值的index

            # pred_u_w2 = pred22_u.detach()  # bs, c, h, w                                                            #与前面一样，dct版本（model2）
            pred_u_w2 = pred22_u
            pred22_u_feature = pred22_u.softmax(dim=1)
            conf_u_w2 = pred22_u.softmax(dim=1).max(dim=1)[0]  # bs, h, w
            mask_u_w2 = pred22_u_feature.argmax(dim=1)  # bs, h, w 伪标签?

            # pred_u_w3 = pred1.detach()  # bs, c, h, w                                                            #与第一部分一样，model3训练得出的数据
            pred_u_w3 = pred1_u
            pred1_u_feature = pred1_u.softmax(dim=1)
            conf_u_w3 = pred1_u.softmax(dim=1).max(dim=1)[0]  # bs, h, w
            # mask_u_w3 = pred_u_w3.argmax(dim=1)  # bs, h, w
            mask_u_w3 = pred1_u_feature.argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w1.clone(), conf_u_w1.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w2.clone(), conf_u_w2.clone()
            mask_u_w_cutmixed3, conf_u_w_cutmixed3 = mask_u_w3.clone(), conf_u_w3.clone()

            # pred1s_u = student1(uimageA1)
            # # pred2s_u = student2(uimageA2)
            # # preds_u = teacher(uimage)
            # pred1_u = pred1s_u['out']                                                               #这个out到底是什么，为什么有两个out？
            # # pred2_u = pred2s_u['out']
            # # pred_u = preds_u['out']
            # pred1_u_feature = torch.softmax(pred1_u, dim=1)

            pseudo1 = mask_u_w3
            pseudo11 = mask_u_w1
            pseudo22 = mask_u_w2
            # pseudo1 = mask_u_w_cutmixed3
            # pseudo11=mask_u_w_cutmixed1
            # pseudo22=mask_u_w_cutmixed2

            # pred2_u_feature = torch.softmax(pred2_u, dim=1)
            # pseudo2 = torch.argmax(pred2_u_feature, dim=1)                                                              #student2预测结果(使用了ueimageA2)
            # pred_u_feature = torch.softmax(pred_u, dim=1)
            # pseudo = torch.argmax(pred_u_feature, dim=1)                                                                #teacher预测结果(使用了ueimage)

            # # # CMT loss
            # loss1_cmt = criterion_c(pred1_u, pseudo11.detach())                           #与教师模型的伪标签损失
            # loss2_cmt = criterion_c(pred22_u, pseudo11.detach())
            # loss_cmt1 = (loss1_cmt + loss2_cmt) * 0.5

            # CPS loss
            loss1_u = criterion_u(pred1_u, pseudo11.detach())  # 与另一个学生伪标签的交叉损失
            loss122_u = criterion_u(pred1_u, pseudo22.detach())
            loss1_u_total = (loss1_u + loss122_u) / 2.

            loss11_u = criterion_u(pred11_u, pseudo1.detach())  # 与另一个学生伪标签的交叉损失
            loss1122_u = criterion_u(pred11_u, pseudo22.detach())
            loss11_u_total = (loss11_u + loss1122_u) / 2.

            loss22_u = criterion_u(pred22_u, pseudo1.detach())
            loss2211_u = criterion_u(pred22_u, pseudo11.detach())
            loss22_u_total = (loss22_u + loss2211_u) / 2.

            loss_cps1 = loss1_u_total + loss11_u_total + loss22_u_total
            # loss_u = (loss_cps1 + loss_cmt1) * alpha                                                                      #整体的loss_u结合了CMT loss与CPS loss
            loss_u = (loss_cps1) * alpha  #################################################################################0.1还是1.0？？？？？？？？？？？
            # lambda_ = sigmoid_rampup(current=idx + len(pbar) * (epoch-1), rampup_length=len(pbar)*5)
            # loss1 = loss_sup1 + lambda_ * loss_u                                                                          #总损失
            # loss1 = loss_sup1  + loss_cps_l1+ loss_u
            loss1 = loss_sup1 + loss_cps_l1 + loss_u
            loss1.backward()  # 损失回传
            optimizer1.step()  # 每个mini-batch更新一次🧑‍🎓
            optimizer11.step()
            optimizer22.step()

            # teacher.weighted_update(student1, student2, ema_decay=0.99, cur_step=idx + len(pbar) * (epoch-1)) #EMA更新################################################
            def entropy(probs):
                eps = 1e-6
                entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
                return entropy.mean(dim=(1, 2))

            # pseudo-label uncertainty calculate
            entropy1 = entropy(pred11_u_feature)
            entropies1.append(entropy1.mean().item())
            entropy2 = entropy(pred22_u_feature)
            entropies2.append(entropy2.mean().item())
            entropy3 = entropy(pred1_u_feature)
            entropies3.append(entropy3.mean().item())

            writer.add_scalar('train_sup1_loss', loss_sup1.item(), idx + len(pbar) * (epoch - 1))  # 写入log tensorboard?
            writer.add_scalar('train_cps1_loss', loss_cps1.item(), idx + len(pbar) * (epoch - 1))
            # writer.add_scalar('train_cmt1_loss', loss_cmt1.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_loss1', loss1.item(), idx + len(pbar) * (epoch - 1))
            # if idx % args.log_freq == 0:
            #     logger.info("Train1: Epoch/Epochs {}/{}\t"
            #                 "iter/iters {}/{}\t"
            #                 "loss1 {:.3f}, loss_sup1 {:.3f}, loss_cps1 {:.3f}, loss_cmt1 {:.3f}, lambda {}".format(epoch, args.num_epochs, idx, len(pbar),
            #                                                                       loss1.item(), loss_sup1.item(), loss_cps1.item(), loss_cmt1.item(), lambda_))
            # if idx % args.log_freq == 0:
            #     logger.info("Train1: Epoch/Epochs {}/{}\t"
            #                 "iter/iters {}/{}\t"
            #                 "loss1 {:.3f}, loss_sup1 {:.3f}, loss_cps1 {:.3f}, lambda {}".format(epoch, args.num_epochs, idx, len(pbar),
            #                                                                       loss1.item(), loss_sup1.item(), loss_cps1.item(), lambda_))
            if idx % args.log_freq == 0:
                logger.info("Train1: Epoch/Epochs {}/{}\t"
                            "iter/iters {}/{}\t"
                            "loss1 {:.3f}, loss_sup1 {:.3f}, loss_cps1 {:.3f}".format(epoch, args.num_epochs, idx,
                                                                                      len(pbar),
                                                                                      loss1.item(), loss_sup1.item(),
                                                                                      loss_cps1.item()))
            epoch_metrics.train_loss.append(loss1.item())
            # '''
            # Step 2
            # '''
            # dct_transform = DCTTransform()
            # optimizer1.zero_grad()
            # # optimizer2.zero_grad()
            # optimizer11.zero_grad()
            # optimizer22.zero_grad()
            # topk = args.topk
            # # 有监督
            # with torch.no_grad():
            #     uncertainty_map11 = torch.mean(torch.stack([pred1_feature, pred11_feature]), dim=0)  # [4 2 256 256]
            #     uncertainty_map11 = -1.0 * torch.sum(uncertainty_map11 * torch.log(uncertainty_map11 + 1e-6), dim=1, keepdim=True)  # [4 1 256 256]
            #     uncertainty_map22 = torch.mean(torch.stack([pred22_feature, pred11_feature]), dim=0)
            #     uncertainty_map22 = -1.0 * torch.sum(uncertainty_map22 * torch.log(uncertainty_map22 + 1e-6), dim=1, keepdim=True)
            #
            #     B, C = image.shape[0], image.shape[1]  # batch channel 4 3
            #     # for student 1
            #     x11 = unfolds(uncertainty_map11)  # B x C*kernel_size[0]*kernel_size[1] x L [4 256 256] (这里的C是1,与上面image的C不一样)    [4 256 256]将每个滑动窗口内的数据列成单独一列
            #     # print(x11.size())
            #     x11 = x11.view(B, 1, h, w, -1)  # B x C x h x w x L [4 1 16 16 256]
            #     # print(x11.size())
            #     x11_mean = torch.mean(x11, dim=(1, 2, 3))  # B x L  [4 256]
            #     _, x11_max_index = torch.sort(x11_mean, dim=1,
            #                                   descending=True)  # B x L B x L [4 256] 排序结果   结果对应的数字在原来tensor中的index
            #     # for student 2
            #     x22 = unfolds(uncertainty_map22)  # B x C*kernel_size[0]*kernel_size[1] x L  (这里的C是1,与上面image的C不一样)    [4 256 256]将每个滑动窗口内的数据列成单独一列
            #     x22 = x22.view(B, 1, h, w, -1)  # B x C x h x w x L (这里的C是1,与上面image的C不一样) [4 1 16 16 256]
            #     x22_mean = torch.mean(x22, dim=(1, 2, 3))  # B x L [4 256]
            #     _, x22_max_index = torch.sort(x22_mean, dim=1, descending=True)  # B x L B x L [4 256]
            #     img_unfold = unfolds(imageA1).view(B, C, h, w,
            #                                        -1)  # B x C x h x w x L######################################################################################################
            #     lab_unfold = unfolds(label.float()).view(B, 1, h, w, -1)  # B x C x h x w x L\
            #     # print(label.shape)
            #     # return 0
            #     for i in range(B):
            #         img_unfold[i, :, :, :, x11_max_index[i, :topk]] = img_unfold[i, :, :, :, x22_max_index[i, -topk:]]
            #         img_unfold[i, :, :, :, x22_max_index[i, :topk]] = img_unfold[i, :, :, :, x11_max_index[i, -topk:]]
            #         lab_unfold[i, :, :, :, x11_max_index[i, :topk]] = lab_unfold[i, :, :, :, x22_max_index[i, -topk:]]
            #         lab_unfold[i, :, :, :, x22_max_index[i, :topk]] = lab_unfold[i, :, :, :, x11_max_index[i, -topk:]]
            #     image2 = folds(img_unfold.view(B, C * h * w, -1))  # [4 3 256 256]                                                   #image2是交换后的图像？
            #     label2 = folds(lab_unfold.view(B, 1 * h * w, -1))  # [4 1 256 256]
            #     image2_dct_expanded_to_concat = []
            #     l = image2.size(0)
            #     # print(image2.size(0))
            #     # tensors = torch.chunk(image2, 4, dim=0)
            #     for i in range(l):
            #         single_image = image2[i]  # type torch.tensor   shape [3 256 256]
            #         # print(single_image.shape)
            #         # single_image = single_image.numpy()
            #         pil_image = ToPILImage()(single_image)
            #         # print(type(pil_image))
            #         image2_single_dct = dct_transform.__call__(deepcopy(pil_image))
            #         image2_dct_expanded = image2_single_dct.unsqueeze(0)
            #         image2_dct_expanded_to_concat.append(image2_dct_expanded)
            #     image2_dct = torch.cat(image2_dct_expanded_to_concat, dim=0)  ############得到dct转换后的tensor image2_dct
            #     image2_dct = image2_dct.cuda()
            # pred11_stage2 = model1(image2)  ##############################################[4 2 256 256]                                   模型的更新放在哪里?
            # # print(type(label2.squeeze(1).long()))
            # # return 0
            # pred22_stage2 = model2(image2_dct, ([256, 256]))
            # pred1_stage2 = model3(image2)
            #
            # pred11_stage2_feature = torch.softmax(pred11_stage2, dim=1)
            # pred22_stage2_feature = torch.softmax(pred22_stage2, dim=1)
            # pred1_stage2_feature = torch.softmax(pred1_stage2, dim=1)
            # pseudo11_l2 = torch.argmax(pred11_stage2_feature, dim=1)
            # pseudo22_l2 = torch.argmax(pred22_stage2_feature, dim=1)
            # pseudo1_l2 = torch.argmax(pred1_stage2_feature, dim=1)
            # loss11_cps_l2 = (criterion_c(pred11_stage2, pseudo22_l2.detach()) + criterion_c(pred11_stage2, pseudo1_l2.detach())) / 2.
            # loss22_cps_l2 = (criterion_c(pred22_stage2, pseudo11_l2.detach()) + criterion_c(pred22_stage2, pseudo1_l2.detach())) / 2.
            # loss1_cps_l2 = (criterion_c(pred1_stage2, pseudo11_l2.detach()) + criterion_c(pred1_stage2, pseudo22_l2.detach())) / 2.
            # loss_cps_l2 = loss11_cps_l2 + loss22_cps_l2 + loss1_cps_l2
            #
            # loss11_sup = criterion(pred11_stage2, label2.squeeze(1).long())  ###################################################修改后的loss,损失函数好要用criterion_l？？？？？？？？？？？？？？？
            # loss22_sup = criterion(pred22_stage2, label2.squeeze(1).long())  #################################################修改后的loss,损失函数好要用criterion_l？？？？？？？？？？？？？？？？
            # loss1_sup = criterion(pred1_stage2, label2.squeeze(1).long())
            # loss_sup = loss11_sup + loss22_sup + loss1_sup*9
            # # 无监督
            # with torch.no_grad():
            #     uncertainty_map1 = torch.mean(torch.stack([pred1_u_feature, pred11_u_feature]), dim=0)
            #     uncertainty_map1 = -1.0 * torch.sum(uncertainty_map1 * torch.log(uncertainty_map1 + 1e-6), dim=1, keepdim=True)
            #     # print(pred1_u_feature.shape)#[4,1,256,256]
            #     # print((uncertainty_map1*torch.log(uncertainty_map1 + 1e-6)).shape)#[4,2,256,256]
            #     # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            #     uncertainty_map2 = torch.mean(torch.stack([pred22_u_feature, pred11_u_feature]), dim=0)
            #     uncertainty_map2 = -1.0 * torch.sum(uncertainty_map2 * torch.log(uncertainty_map2 + 1e-6), dim=1, keepdim=True)
            #
            #     B, C = uimage.shape[0], uimage.shape[1]
            #     # for student 1
            #     x1 = unfolds(uncertainty_map1)  # B x C*kernel_size[0]*kernel_size[1] x L
            #     x1 = x1.view(B, 1, h, w, -1)  # B x C x h x w x L
            #     x1_mean = torch.mean(x1, dim=(1, 2, 3))  # B x L
            #     _, x1_max_index = torch.sort(x1_mean, dim=1, descending=True)  # B x L B x L
            #     # for student 2
            #     x2 = unfolds(uncertainty_map2)  # B x C*kernel_size[0]*kernel_size[1] x L
            #     x2 = x2.view(B, 1, h, w, -1)  # B x C x h x w x L
            #     x2_mean = torch.mean(x2, dim=(1, 2, 3))  # B x L
            #     _, x2_max_index = torch.sort(x2_mean, dim=1, descending=True)  # B x L B x L
            #     imgu_unfold = unfolds(uimageA1).view(B, C, h, w,
            #                                          -1)  # B x C x h x w x L  [4,3,16,16,256]#########################################################################
            #     # print(imgu_unfold.shape)
            #     pseudo_unfold = unfolds(pseudo11.unsqueeze(1).float()).view(B, 1, h, w, -1)  # B x C x h x w x
            #     for i in range(B):
            #         imgu_unfold[i, :, :, :, x1_max_index[i, :topk]] = imgu_unfold[i, :, :, :, x2_max_index[i, -topk:]]
            #         imgu_unfold[i, :, :, :, x2_max_index[i, :topk]] = imgu_unfold[i, :, :, :, x1_max_index[i, -topk:]]
            #         pseudo_unfold[i, :, :, :, x1_max_index[i, :topk]] = pseudo_unfold[i, :, :, :, x2_max_index[i, -topk:]]
            #         pseudo_unfold[i, :, :, :, x2_max_index[i, :topk]] = pseudo_unfold[i, :, :, :, x1_max_index[i, -topk:]]
            #     uimage2 = folds(imgu_unfold.view(B, C * h * w, -1))
            #     pseudo = folds(pseudo_unfold.view(B, 1 * h * w, -1)).squeeze(1).long()
            #     uimage2_dct_expanded_to_concat = []
            #     j = uimage2.size(0)
            #     for i in range(j):
            #         single_image = uimage2[i]  # type torch.tensor   shape [3 256 256]
            #         # single_image = single_image.numpy()
            #         pil_image = ToPILImage()(single_image)
            #         uimage2_single_dct = dct_transform.__call__(deepcopy(pil_image))
            #         uimage2_dct_expanded = uimage2_single_dct.unsqueeze(0)
            #         uimage2_dct_expanded_to_concat.append(uimage2_dct_expanded)
            #     uimage2_dct = torch.cat(uimage2_dct_expanded_to_concat, dim=0)  ############得到dct转换后的tensor uimage2_dct
            #     uimage2_dct = uimage2_dct.cuda()
            # # uimage2_dct_expanded_to_concat=[]
            # # j = uimageA2.size(0)
            # # for i in range(j):
            # #     single_image = uimageA2[i]  # type torch.tensor   shape [3 256 256]
            # #     # single_image = single_image.numpy()
            # #     pil_image = ToPILImage()(single_image)
            # #     uimage2_single_dct = dct_transform.__call__(deepcopy(pil_image))
            # #     uimage2_dct_expanded = uimage2_single_dct.unsqueeze(0)
            # #     uimage2_dct_expanded_to_concat.append(uimage2_dct_expanded)
            # # uimage2_dct = torch.cat(uimage2_dct_expanded_to_concat, dim=0)  ############得到dct转换后的tensor uimage2_dct
            # # uimage2_dct = uimage2_dct.cuda()
            # pred11_u_stage2 = model1(uimage2)  ##################################################
            # pred22_u_stage2 = model2(uimage2_dct, ([256, 256]))  #################################################
            # pred1_u_stage2 = model3(uimage2)
            #
            # # pred_u_w1 = pred11_u_stage2.detach()  # bs, c, h, w                                                            #无法继续向前传播梯度的副本（model1）
            # pred_u_w1 = pred11_u_stage2
            # pred11_u_feature = pred11_u_stage2.softmax(dim=1)
            # conf_u_w1 = pred11_u_stage2.softmax(dim=1).max(dim=1)[0]  # 保留最大值
            # # mask_u_w1 = pred_u_w1.argmax(dim=1)  # bs, h, w                                                          #得到最大值的index
            # mask_u_w1 = pred11_u_feature.argmax(dim=1)
            #
            # # pred_u_w2 = pred22_u.detach()  # bs, c, h, w                                                            #与前面一样，dct版本（model2）
            # pred_u_w2 = pred22_u_stage2
            # pred22_u_feature = pred22_u_stage2.softmax(dim=1)
            # conf_u_w2 = pred22_u_stage2.softmax(dim=1).max(dim=1)[0]  # bs, h, w
            # # mask_u_w2 = pred_u_w2.argmax(dim=1)  # bs, h, w 伪标签?
            # mask_u_w2 = pred22_u_feature.argmax(dim=1)
            #
            # # pred_u_w3 = pred1.detach()  # bs, c, h, w                                                            #与第一部分一样，model3训练得出的数据
            # pred_u_w3 = pred1_u_stage2
            # pred1_u_feature = pred1_u_stage2.softmax(dim=1)
            # conf_u_w3 = pred1_u_stage2.softmax(dim=1).max(dim=1)[0]  # bs, h, w
            # # mask_u_w3 = pred_u_w3.argmax(dim=1)  # bs, h, w
            # mask_u_w3 = pred1_u_feature.argmax(dim=1)
            #
            # # pred1s_u = student1(uimage2)
            # # pred1_u = pred1s_u['out']
            # # pred1_u_feature = torch.softmax(pred1_u, dim=1)
            # # pseudo1 = torch.argmax(pred1_u_feature, dim=1)
            #
            # mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w1.clone(), conf_u_w1.clone()
            # mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w2.clone(), conf_u_w2.clone()
            # mask_u_w_cutmixed3, conf_u_w_cutmixed3 = mask_u_w3.clone(), conf_u_w3.clone()
            #
            # pseudo1 = mask_u_w3
            # pseudo11 = mask_u_w1
            # pseudo22 = mask_u_w2
            # # pseudo1 = mask_u_w_cutmixed3
            # # pseudo11 = mask_u_w_cutmixed1
            # # pseudo22 = mask_u_w_cutmixed2
            #
            # # # CMT loss
            # # loss11_cmt = criterion_c(pred1_u, pseudo11.detach())                                                     #！！！！！！！！！！！pesudo要重新定义
            # # loss22_cmt = criterion_c(pred22_u, pseudo11.detach())
            # # loss_cmt = (loss11_cmt + loss22_cmt) * 0.5
            #
            # # loss11_cmt = criterion_c(pred1_u, pseudo.detach())                                                     #！！！！！！！！！！！这样可以吗？？？？？？？？？
            # # loss22_cmt = criterion_c(pred22_u, pseudo.detach())
            # # loss_cmt = (loss11_cmt + loss22_cmt) * 0.5
            #
            # # CPS loss
            # loss1_u = criterion_u(pred1_u_stage2, pseudo11.detach())  # 与另一个学生伪标签的交叉损失
            # loss122_u = criterion_u(pred1_u_stage2, pseudo22.detach())
            # loss1_u_total = (loss1_u + loss122_u) / 2.
            #
            # loss11_u = criterion_u(pred11_u_stage2, pseudo1.detach())  # 与另一个学生伪标签的交叉损失
            # loss1122_u = criterion_u(pred11_u_stage2, pseudo22.detach())
            # loss11_u_total = (loss11_u + loss1122_u) / 2.
            #
            # loss22_u = criterion_u(pred22_u_stage2, pseudo1.detach())
            # loss2211_u = criterion_u(pred22_u_stage2, pseudo11.detach())
            # loss22_u_total = (loss22_u + loss2211_u) / 2.
            #
            # loss_cps2 = loss1_u_total + loss11_u_total + loss22_u_total
            #
            # # loss_u_s12 = criterion_u(pred11_u, mask_u_w_cutmixed2)
            #
            # ## loss_u_s12 = loss_u_s12 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            # ## loss_u_s12 = torch.sum(loss_u_s12) / torch.sum(ignore_mask_cutmixed1 != 255).item()
            #
            # # loss11_u = loss_u_s12
            #
            # # loss_u_s21 = criterion_u(pred22_u, mask_u_w_cutmixed1)
            #
            # ## loss_u_s21 = loss_u_s21 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            # ## loss_u_s21 = torch.sum(loss_u_s21) / torch.sum(ignore_mask_cutmixed2 != 255).item()
            # # loss22_u = loss_u_s21
            # # loss_cps2 = (loss11_u + loss22_u) * 0.5
            # # torch.distributed.barrier()
            # loss_u = (loss_cps2) * alpha
            # # lambda_ = sigmoid_rampup(current=idx + len(pbar) * (epoch - 1), rampup_length=len(pbar) * 5)
            # # loss = loss_sup + lambda_ * loss_u
            # # loss = loss_sup +loss_cps_l2 +loss_u
            # loss = loss_sup + loss_cps_l2 + loss_u
            # loss.backward()
            # optimizer1.step()
            # optimizer11.step()
            # optimizer22.step()
            #
            # entropy1 = entropy(pred11_u_feature)
            # entropies1.append(entropy1.mean().item())
            # entropy2 = entropy(pred22_u_feature)
            # entropies2.append(entropy2.mean().item())
            # entropy3 = entropy(pred1_u_feature)
            # entropies3.append(entropy3.mean().item())


            # writer.add_scalar('train_sup_loss', loss_sup.item(), idx + len(pbar) * (epoch - 1))
            # writer.add_scalar('train_cps_loss', loss_cps2.item(), idx + len(pbar) * (epoch - 1))
            # # writer.add_scalar('train_cmt_loss', loss_cmt.item(), idx + len(pbar) * (epoch - 1))
            # writer.add_scalar('train_loss', loss.item(), idx + len(pbar) * (epoch - 1))




            # if idx % args.log_freq == 0:
            #     logger.info("Train2: Epoch/Epochs {}/{}\t"
            #                 "iter/iters {}/{}\t"
            #                 "loss {:.3f}, loss_sup {:.3f}, loss_cps2 {:.3f}".format(
            #         epoch, args.num_epochs, idx, len(pbar),
            #         loss.item(), loss_sup.item(), loss_cps2.item()))
            # epoch_metrics.train_loss2.append(loss.item())

        metrics.train_loss.append(np.mean(epoch_metrics.train_loss))
        # metrics.train_loss2.append(np.mean(epoch_metrics.train_loss2))
        # logger.info(
        #     "Average: Epoch/Epoches {}/{}\t"  # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        #     "train1 epoch loss {:.3f}\t"
        #     "train2 epoch loss {:.3f}\n".format(epoch, args.num_epochs,
        #                                         np.mean(epoch_metrics.train_loss),
        #                                         np.mean(epoch_metrics.train_loss2), ))
        logger.info(
            "Average: Epoch/Epoches {}/{}\t"  # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            "train1 epoch loss {:.3f}\n".format(epoch, args.num_epochs, np.mean(epoch_metrics.train_loss),))
        # logger.info("Average: Epoch/Epoches {}/{}\t"
        #             "train2 epoch loss {:.3f}\n".format(epoch, args.num_epochs,
        #                                                 np.mean(epoch_metrics.train_loss2), ))
        if np.mean(epoch_metrics.train_loss) <= best_loss:  # 保存的是teacher模型
            best_loss = np.mean(epoch_metrics.train_loss)
            # torch.save(teacher.state_dict(), save_path + 'best.pth'.format(best_epoch))#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            # torch.save(student1.state_dict(), save_path + 'student1_best.pth'.format(best_epoch))
            # torch.save(student2.state_dict(), save_path + 'student2_best.pth'.format(best_epoch))

            # if mIOU1 > previous_best1 and rank == 0:
            #     if previous_best1 != 0:
            #         pre_path = os.path.join(args.save_path, 'm1_%s_%.2f.pth' % (cfg['backbone'], previous_best1))
            #         if os.path.exists(pre_path):
            #             os.remove(pre_path)
            #     previous_best1 = mIOU1
            torch.save(model1.state_dict(), save_path + 'model1_best.pth'.format(best_epoch))
            #                     os.path.join(args.save_path, 'm1_%s_%.2f.pth' % (cfg['backbone'], mIOU1)))
            #
            #     if mIOU2 > previous_best2 and rank == 0:
            #         if previous_best2 != 0:
            #             pre_path = os.path.join(args.save_path, 'm2_%s_%.2f.pth' % (cfg['backbone'], previous_best2))
            #             if os.path.exists(pre_path):
            #                 os.remove(pre_path)
            #         previous_best2 = mIOU2
            torch.save(model2.state_dict(), save_path + 'model2_best.pth'.format(best_epoch))
            torch.save(model3.state_dict(), save_path + 'model3_best.pth'.format(best_epoch))
        #                     os.path.join(args.save_path, 'm2_%s_%.2f.pth' % (cfg['backbone'], mIOU2)))
        #
        #     if mIOU3 > previous_best3 and rank == 0:
        #         if previous_best3 != 0:
        #             pre_path = os.path.join(args.save_path, 'm3_%s_%.2f.pth' % (cfg['backbone'], previous_best3))
        #             if os.path.exists(pre_path):
        #                 os.remove(pre_path)
        #         previous_best3 = mIOU3
        #         torch.save(model3.module.state_dict(),
        #                     os.path.join(args.save_path, 'm3_%s_%.2f.pth' % (cfg['backbone'], mIOU3)))

        # torch.save(teacher.state_dict(), save_path + 'last.pth'.format(best_epoch))#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        torch.save(model1.state_dict(), save_path + 'model1_last.pth'.format(best_epoch))
        torch.save(model2.state_dict(), save_path + 'model2_last.pth'.format(best_epoch))
        torch.save(model3.state_dict(), save_path + 'model3_last.pth'.format(best_epoch))
        # torch.save(student1.state_dict(), save_path + 'student1_last.pth'.format(best_epoch))
        # torch.save(student2.state_dict(), save_path + 'student2_best.pth'.format(best_epoch))
        ############################
        # Save Metrics
        ############################
    # data_frame = pd.DataFrame(
    #     data={'loss': metrics.train_loss,
    #           'loss2': metrics.train_loss2},
    #     index=range(1, args.num_epochs + 1))  # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

    data_frame = pd.DataFrame(
        data={'loss': metrics.train_loss},
        index=range(1, args.num_epochs + 1))
    # data_frame = pd.DataFrame(
    #     data={'loss2': metrics.train_loss2},
    #     index=range(1, args.num_epochs + 1))
    data_frame.to_csv(project_path + 'train_loss.csv', index_label='Epoch')

    # uncertainty save to csv
    with open(os.path.join(project_path, 'entropies1.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Entropy'])
        for i, entropy in enumerate(entropies1):
            writer.writerow([i, entropy])
    with open(os.path.join(project_path, 'entropies2.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Entropy'])
        for i, entropy in enumerate(entropies2):
            writer.writerow([i, entropy])
    with open(os.path.join(project_path, 'entropies3.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Entropy'])
        for i, entropy in enumerate(entropies3):
            writer.writerow([i, entropy])

    plt.figure()
    plt.title("Loss")
    plt.plot(metrics.train_loss, label="Train")  # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    plt.plot(metrics.train_loss2, label="Train2")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(project_path + 'train_loss.png')
    plt.close()

    # pseudo-labels uncertainty
    plt.figure()
    plt.plot(range(len(entropies1)), entropies1)
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')
    plt.title('Pseudo-labels Uncertainty')
    plt.savefig(project_path + 'Uncertainty1.png')
    # plt.close()

    # plt.figure()
    plt.plot(range(len(entropies2)), entropies2)
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')
    plt.title('Pseudo-labels Uncertainty')
    plt.savefig(project_path + 'Uncertainty2.png')
    # plt.close()

    # plt.figure()
    plt.plot(range(len(entropies3)), entropies3)
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')
    plt.title('Pseudo-labels Uncertainty')
    plt.savefig(project_path + 'Uncertainty3.png')
    plt.close()

    time_elapsed = time.time() - since
    logger.info('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info(project_path)
    logger.info('TRAINING FINISHED!')


if __name__ == '__main__':
    main()