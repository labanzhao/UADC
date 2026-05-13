import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import time
import argparse
import copy
import cv2
import numpy as np
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
    parser.add_argument('--data_path', type=str, default='YOUR_DATA_PATH', help='path to the data')
    parser.add_argument('--image_size', type=int, default=256, help='the size of images for training and testing')
    parser.add_argument('--labeled_percentage', type=float, default=0.05, help='the percentage of labeled data')
    parser.add_argument('--is_cutmix', type=bool, default=False, help='cut mix')
    parser.add_argument('--mix_prob', type=float, default=0.5, help='probability for amplitude mix')            #？
    parser.add_argument('--topk', type=float, default=2, help='top k')
    parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs')
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

def get_data(args):
    train_set = ISICDataset(image_path=args.data_path, stage='train', image_size=args.image_size, is_augmentation=True)
    labeled_train_set, unlabeled_train_set = random_split(train_set, [int(len(train_set) * args.labeled_percentage),
                                                       len(train_set) - int(len(train_set) * args.labeled_percentage)])         #训练集分成标记数据与无标记数据
    print('before:', len(labeled_train_set), len(train_set))                                #90,1815
    # repeat the labeled set to have a equal length with the unlabeled set (dataset)
    labeled_ratio = len(train_set) // len(labeled_train_set)                                #20
    labeled_train_set = ConcatDataset([labeled_train_set for i in range(labeled_ratio)])    #1800
    # print(len(train_set))
    # print("###########################")
    # print(len(labeled_train_set))
    # print("###########################")
    labeled_train_set = ConcatDataset([labeled_train_set,
                                       Subset(labeled_train_set, range(len(train_set) - len(labeled_train_set)))])  #1815,重复labeled_train_dataset至train_set长度一致的数据集
    # print(len(labeled_train_set))
    assert len(labeled_train_set) == len(train_set)
    print('after:', len(labeled_train_set), len(train_set))
    train_labeled_dataloder = DataLoader(dataset=labeled_train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    train_unlabeled_dataloder = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)  #这里train_set也包含了有标记？
    if args.is_cutmix:
        mask_generator = BoxMaskGenerator(prop_range=(0.25, 0.5),
                                          n_boxes=3,
                                          random_aspect_ratio=True,
                                          prop_by_area=True,
                                          within_bounds=True,
                                          invert=True)

        add_mask_params_to_batch = AddMaskParamsToBatch(mask_generator)
        mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)
        aux_dataloder = DataLoader(dataset=labeled_train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=mask_collate_fn)
        return train_labeled_dataloder, train_unlabeled_dataloder, aux_dataloder
    return train_labeled_dataloder, train_unlabeled_dataloder


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
        train_labeled_dataloader, train_unlabeled_dataloader, aux_loader = get_data(args=args)
    else:
        train_labeled_dataloader, train_unlabeled_dataloader = get_data(args=args)
    iters = len(train_labeled_dataloader)           #一共1815张图像，一个batch 4张，iters=454
    #print("iters:",iters)

    # Load Model & EMA
    student1 = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)    #.__dict__字典替换？
    #print(student1)
    student2 = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
    teacher = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
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
    alpha = 2.0
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
        optimizer11 = AdamW(param_groups1, lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.999))
    if cfg2["optim"] == "SGD":
        optimizer22 = SGD(param_groups2, lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    elif cfg2["optim"] == "AdamW":
        optimizer22 = AdamW(param_groups2, lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.999))
    if cfg3["optim"] == "SGD":
        optimizer1 = SGD(param_groups3, lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    elif cfg3["optim"] == "AdamW":
        optimizer1 = AdamW(param_groups3, lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.999))

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
        iter_train_labeled_dataloader = iter(train_labeled_dataloader)
        iter_train_unlabeled_dataloader = iter(train_unlabeled_dataloader)
        #print(1,next(iter_train_labeled_dataloader))
        # return 0
        ############################
        # Train
        ############################
        student1.train()        #训练模式
        student2.train()
        teacher.train()
        model1.train()
        model2.train()
        model3.train()
        for idx in pbar:
            image, label, imageA1, imageA2 = next(iter_train_labeled_dataloader)        #imageA1, imageA2是image augmentation，返回一个batch的数据
            # print(image)
            # print(label)
            # print(imageA1)
            # print(imageA2)
            # print("#########################")
            image, label = image.to(device), label.to(device)
            imageA1, imageA2 = imageA1.to(device), imageA2.to(device)
            uimage, _, uimageA1, uimageA2 = next(iter_train_unlabeled_dataloader)
            uimage = uimage.to(device)
            uimageA1, uimageA2 = uimageA1.to(device), uimageA2.to(device)

            # '''
            # Step 1
            # '''
            # optimizer1.zero_grad()
            # optimizer2.zero_grad()
            # ###########################
            #     # supervised path #
            # ###########################
            # pred1s = student1(image)
            # # print(pred1s)
            # # print("#############")
            # pred2s = student2(image)
            # preds = teacher(image)
            # pred1 = pred1s['out']                                                               #??字典？？预测值
            # # print(pred1)
            # # print("#############")
            # pred2 = pred2s['out']
            # pred = preds['out']
            # pred1_feature = torch.softmax(pred1, dim=1)
            # pred2_feature = torch.softmax(pred2, dim=1)
            # pred_feature = torch.softmax(pred, dim=1)
            # loss1_sup = criterion(pred1, label.squeeze(1).long())                                                       #[4,2,256,256]  [4,256,256]
            # loss2_sup = criterion(pred2, label.squeeze(1).long())
            # loss_sup = loss1_sup + loss2_sup
            #
            # ###########################
            #     # unsupervised path #                       #预测伪标签？
            # ###########################
            # # Estimate the pseudo-labels
            # pred1s_u = student1(uimageA1)
            # pred2s_u = student2(uimageA2)
            # preds_u = teacher(uimage)
            # pred1_u = pred1s_u['out']                                                               #这个out到底是什么，为什么有两个out？
            # pred2_u = pred2s_u['out']
            # pred_u = preds_u['out']
            # pred1_u_feature = torch.softmax(pred1_u, dim=1)
            # pseudo1 = torch.argmax(pred1_u_feature, dim=1)                                                              #student1预测结果（使用了ueimageA1）（获得每行中最大的数值的位置）
            # pred2_u_feature = torch.softmax(pred2_u, dim=1)
            # pseudo2 = torch.argmax(pred2_u_feature, dim=1)                                                              #student2预测结果(使用了ueimageA2)
            # pred_u_feature = torch.softmax(pred_u, dim=1)
            # pseudo = torch.argmax(pred_u_feature, dim=1)                                                                #teacher预测结果(使用了ueimage)
            #
            # # CMT loss
            # loss1_cmt = criterion_c(pred1_u, pseudo.detach())                           #与教师模型的伪标签损失
            # loss2_cmt = criterion_c(pred2_u, pseudo.detach())
            # loss_cmt = (loss1_cmt + loss2_cmt) * 0.5
            #
            # # CPS loss
            # loss1_u = criterion_u(pred1_u, pseudo2.detach())                            #与另一个学生伪标签的交叉损失
            # loss2_u = criterion_u(pred2_u, pseudo1.detach())
            # loss_cps = (loss1_u + loss2_u) * 0.5
            # loss_u = (loss_cps + loss_cmt) * alpha                                                                      #整体的loss_u结合了CMT loss与CPS loss
            # lambda_ = sigmoid_rampup(current=idx + len(pbar) * (epoch-1), rampup_length=len(pbar)*5)
            # loss = loss_sup + lambda_ * loss_u                                                                          #总损失
            # loss.backward()                                                                                             #损失回传
            # optimizer1.step()                                                                                           #每个mini-batch更新一次🧑‍🎓
            # optimizer2.step()
            # teacher.weighted_update(student1, student2, ema_decay=0.99, cur_step=idx + len(pbar) * (epoch-1)) #EMA更新
            #
            # writer.add_scalar('train_sup_loss', loss_sup.item(), idx + len(pbar) * (epoch-1))   #写入log tensorboard?
            # writer.add_scalar('train_cps_loss', loss_cps.item(), idx + len(pbar) * (epoch-1))
            # writer.add_scalar('train_cmt_loss', loss_cmt.item(), idx + len(pbar) * (epoch-1))
            # writer.add_scalar('train_loss', loss.item(), idx + len(pbar) * (epoch-1))
            # if idx % args.log_freq == 0:
            #     logger.info("Train1: Epoch/Epochs {}/{}\t"
            #                 "iter/iters {}/{}\t"
            #                 "loss {:.3f}, loss_sup {:.3f}, loss_cps {:.3f}, loss_cmt {:.3f}, lambda {}".format(epoch, args.num_epochs, idx, len(pbar),
            #                                                                       loss.item(), loss_sup.item(), loss_cps.item(), loss_cmt.item(), lambda_))
            # epoch_metrics.train_loss.append(loss.item())

            # '''
            # Step 2
            # '''
            # dct_transform = DCTTransform()
            # optimizer1.zero_grad()
            # optimizer2.zero_grad()
            # optimizer11.zero_grad()
            # optimizer22.zero_grad()
            # topk = args.topk
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
            pred1 = model3(image)  ##############################################[4 2 256 256]                                   模型的更新放在哪里?
            pred1_feature = torch.softmax(pred1, dim=1)
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
            loss1_sup = criterion(pred1, label.squeeze(1).long())                                                       #[4,2,256,256]  [4,256,256]
            # loss2_sup = criterion(pred2, label.squeeze(1).long())
            loss_sup = loss1_sup + loss22_sup +loss11_sup

            pseudo11_l = torch.argmax(pred11_feature, dim=1)
            # print(pseudo11_l.shape)
            # return 0
            pseudo22_l = torch.argmax(pred22_feature, dim=1)
            pseudo1_l = torch.argmax(pred1_feature, dim=1)
            loss11_cps_l1 = (criterion_u(pred11, pseudo22_l.detach()) + criterion_u(pred11, pseudo1_l.detach())) / 2.
            loss22_cps_l1 = (criterion_u(pred22, pseudo11_l.detach()) + criterion_u(pred22, pseudo1_l.detach())) / 2.
            loss1_cps_l1 = (criterion_u(pred1, pseudo11_l.detach()) + criterion_u(pred1, pseudo22_l.detach())) / 2.
            loss_cps_l1 = (loss11_cps_l1 + loss22_cps_l1 + loss1_cps_l1)

            ###########################
                # unsupervised path #                       #预测伪标签？
            ###########################
            # Estimate the pseudo-labels
#无cmt的训练################################################################################################################
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
            uimage_dct = uimage_dct.cuda()

            pred11_u = model1(uimage)  ##################################################
            pred22_u = model2(uimage_dct, ([256, 256]))  #################################################
            pred1_u = model3(uimage)

            pred_u_w1 = pred11_u.detach()  # bs, c, h, w                                                            #无法继续向前传播梯度的副本（model1）
            pred11_u_feature = pred_u_w1.softmax(dim=1)
            conf_u_w1 = pred_u_w1.softmax(dim=1).max(dim=1)[0]  # 保留最大值
            mask_u_w1 = pred_u_w1.argmax(dim=1)  # bs, h, w                                                          #得到最大值的index

            pred_u_w2 = pred22_u.detach()  # bs, c, h, w                                                            #与前面一样，dct版本（model2）
            pred22_u_feature = pred_u_w2.softmax(dim=1)
            conf_u_w2 = pred_u_w2.softmax(dim=1).max(dim=1)[0]  # bs, h, w
            mask_u_w2 = pred_u_w2.argmax(dim=1)  # bs, h, w 伪标签?

            pred_u_w3 = pred1.detach()  # bs, c, h, w                                                            #与第一部分一样，model3训练得出的数据
            pred1_u_feature = pred_u_w3.softmax(dim=1)
            conf_u_w3 = pred_u_w3.softmax(dim=1).max(dim=1)[0]  # bs, h, w
            mask_u_w3 = pred_u_w3.argmax(dim=1)  # bs, h, w

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
            pseudo1 = mask_u_w_cutmixed3
            pseudo11=mask_u_w_cutmixed1
            pseudo22=mask_u_w_cutmixed2
            # pred2_u_feature = torch.softmax(pred2_u, dim=1)
            # pseudo2 = torch.argmax(pred2_u_feature, dim=1)                                                              #student2预测结果(使用了ueimageA2)
            # pred_u_feature = torch.softmax(pred_u, dim=1)
            # pseudo = torch.argmax(pred_u_feature, dim=1)                                                                #teacher预测结果(使用了ueimage)


            # CMT loss
            loss1_cmt = criterion_c(pred1_u, pseudo11.detach())                           #与教师模型的伪标签损失
            loss2_cmt = criterion_c(pred22_u, pseudo11.detach())
            loss_cmt = (loss1_cmt + loss2_cmt) * 0.5

            # CPS loss
            loss1_u = criterion_u(pred1_u, pseudo11.detach())                            #与另一个学生伪标签的交叉损失
            loss122_u=criterion_u(pred1_u, pseudo22.detach())
            loss1_u_total=(loss1_u+loss122_u)/2

            loss11_u = criterion_u(pred11_u, pseudo1.detach())  # 与另一个学生伪标签的交叉损失
            loss1122_u = criterion_u(pred11_u, pseudo22.detach())
            loss11_u_total=(loss11_u+loss1122_u)/2

            loss22_u = criterion_u(pred22_u, pseudo1.detach())
            loss2211_u=criterion_u(pred22_u, pseudo11.detach())
            loss22_u_total = (loss22_u + loss2211_u) / 2

            loss_cps = (loss1_u_total + loss11_u_total+loss22_u_total) /3
            # loss_u = (loss_cps1 + loss_cmt) * alpha                                                                      #整体的loss_u结合了CMT loss与CPS loss
            loss_u = (loss_cps + loss_cmt) * alpha #################################################################################0.1还是1.0？？？？？？？？？？？
            lambda_ = sigmoid_rampup(current=idx + len(pbar) * (epoch-1), rampup_length=len(pbar)*5)
            loss = loss_sup + lambda_ * loss_u                                                                          #总损失
            loss.backward()                                                                                             #损失回传
            optimizer1.step()                                                                                           #每个mini-batch更新一次🧑‍🎓
            optimizer11.step()
            optimizer22.step()
#一直注释到这里###########################################################################################################################################################
            writer.add_scalar('train_sup_loss', loss_sup.item(), idx + len(pbar) * (epoch - 1))
            writer.add_scalar('train_cps_loss', loss_cps.item(), idx + len(pbar) * (epoch - 1))
            writer.add_scalar('train_cmt1_loss', loss_cmt.item(), idx + len(pbar) * (epoch - 1))
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

        # metrics.train_loss.append(np.mean(epoch_metrics.train_loss))
        metrics.train_loss2.append(np.mean(epoch_metrics.train_loss2))
        # logger.info("Average: Epoch/Epoches {}/{}\t"#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        #             "train1 epoch loss {:.3f}\t"
        #             "train2 epoch loss {:.3f}\n".format(epoch, args.num_epochs,
        #                                                 np.mean(epoch_metrics.train_loss),
        #                                                 np.mean(epoch_metrics.train_loss2), ))
        logger.info("Average: Epoch/Epoches {}/{}\t"
                    "train2 epoch loss {:.3f}\n".format(epoch, args.num_epochs,
                                                        np.mean(epoch_metrics.train_loss2), ))
        if np.mean(epoch_metrics.train_loss2) <= best_loss:                                                             #保存的是teacher模型
            best_loss = np.mean(epoch_metrics.train_loss2)
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
            torch.save(model2.state_dict(),save_path + 'model2_best.pth'.format(best_epoch))
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
    #             'loss2': metrics.train_loss2},
    #     index=range(1, args.num_epochs + 1))#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    data_frame = pd.DataFrame(
        data={'loss2': metrics.train_loss2},
        index=range(1, args.num_epochs + 1))
    data_frame.to_csv(project_path + 'train_loss.csv', index_label='Epoch')
    plt.figure()
    plt.title("Loss")
    # plt.plot(metrics.train_loss, label="Train")#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
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


