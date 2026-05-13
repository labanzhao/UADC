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
import csv
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
from PIL import Image


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
    parser.add_argument('--labeled_percentage', type=float, default=0.1, help='the percentage of labeled data')
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





def get_data(args):
    train_set = ISICDataset(image_path=args.data_path, stage='train', image_size=args.image_size, is_augmentation=True)
    labeled_train_set, unlabeled_train_set = random_split(train_set, [int(len(train_set) * args.labeled_percentage),
                                                       len(train_set) - int(len(train_set) * args.labeled_percentage)])         #训练集分成标记数据与无标记数据
    print('before:', len(labeled_train_set), len(train_set))                                #90,1815   181,1815
    # repeat the labeled set to have a equal length with the unlabeled set (dataset)
    labeled_ratio = len(train_set) // len(labeled_train_set)                                #20    10
    labeled_train_set = ConcatDataset([labeled_train_set for i in range(labeled_ratio)])    #1800
    # print(labeled_ratio)
    # return 0
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
    def entropy(probs):
        eps = 1e-6
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
        return entropy.mean(dim=(1, 2))
    args = get_args()
    #print(args.project)
    seed_torch(args.seed)
    # Project Saving Path
    project_path = args.project + '_{}_label_{}_train1_UADC/'.format(args.backbone, args.labeled_percentage)
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
    # entropy=entropy()

    # Load Data
    if args.is_cutmix:
        train_labeled_dataloader, train_unlabeled_dataloader, aux_loader = get_data(args=args)
    else:
        train_labeled_dataloader, train_unlabeled_dataloader = get_data(args=args)
    iters = len(train_labeled_dataloader)           #一共1815张图像，一个batch 4张，iters=454
    #print("iters:",iters)

    # # Load Model & EMA
    # student1 = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)    #.__dict__字典替换？
    # #print(student1)
    # student2 = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
    # teacher = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)

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
        optimizer11 = AdamW(param_groups1, lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.999))
    if cfg2["optim"] == "SGD":
        optimizer22 = SGD(param_groups2, lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    elif cfg2["optim"] == "AdamW":
        optimizer22 = AdamW(param_groups2, lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.999))
    if cfg3["optim"] == "SGD":
        optimizer1 = SGD(param_groups3, lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    elif cfg3["optim"] == "AdamW":
        optimizer1 = AdamW(param_groups3, lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.999))



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
        # student1.train()        #训练模式
        # student2.train()
        # teacher.train()
        model1.train()
        model2.train()
        model3.train()
        # print(model2)
        # print("#################################################################################################")
        # print(model3)
        # return 0
        for idx in pbar:
            image, label, imageA1, imageA2 = next(iter_train_labeled_dataloader)        #imageA1, imageA2是image augmentation，返回一个batch的数据
            # print(image)
            torch.set_printoptions(edgeitems=256)
            # print(label)
            # print(imageA1)
            # print(imageA2)
            # print("#########################")

            image, label = image.to(device), label.to(device)




            #print(label)
            # return 0
            imageA1, imageA2 = imageA1.to(device), imageA2.to(device)
            uimage, _, uimageA1, uimageA2 = next(iter_train_unlabeled_dataloader)
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
            loss_sup1 = loss11_sup * 2 + loss22_sup + loss1_sup




            pseudo11_l = torch.argmax(pred11_feature, dim=1)
            pseudo22_l = torch.argmax(pred22_feature, dim=1)
            pseudo1_l = torch.argmax(pred1_feature, dim=1)
            loss11_cps_l1=(criterion_c(pred11, pseudo22_l.detach())+criterion_c(pred11, pseudo1_l.detach()))/2.
            loss22_cps_l1 =(criterion_c(pred22, pseudo11_l.detach())+criterion_c(pred22, pseudo1_l.detach()))/2.
            loss1_cps_l1 = (criterion_c(pred1, pseudo11_l.detach())+criterion_c(pred1, pseudo22_l.detach()))/2.
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
            uimage_dct = uimage_dct.to(device)

            pred11_u = model1(uimage)  ##################################################？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？uimageA1
            pred22_u = model2(uimage_dct, ([256, 256]))  #################################################
            pred1_u = model3(uimage)###############################################################################？？？？？？？？？？？？？？？？？？？？uimageA2

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
            loss1_u = criterion_u(pred1_u, pseudo11.detach())                            #与另一个学生伪标签的交叉损失
            loss122_u=criterion_u(pred1_u, pseudo22.detach())
            loss1_u_total=(loss1_u+loss122_u)/2.

            loss11_u = criterion_u(pred11_u, pseudo1.detach())  # 与另一个学生伪标签的交叉损失
            loss1122_u = criterion_u(pred11_u, pseudo22.detach())
            loss11_u_total=(loss11_u+loss1122_u)/2.

            loss22_u = criterion_u(pred22_u, pseudo1.detach())
            loss2211_u=criterion_u(pred22_u, pseudo11.detach())
            loss22_u_total = (loss22_u + loss2211_u) / 2.

            loss_cps1 = loss1_u_total + loss11_u_total + loss22_u_total
            # loss_u = (loss_cps1 + loss_cmt1) * alpha                                                                      #整体的loss_u结合了CMT loss与CPS loss
            loss_u = (loss_cps1) * alpha #################################################################################0.1还是1.0？？？？？？？？？？？
            # lambda_ = sigmoid_rampup(current=idx + len(pbar) * (epoch-1), rampup_length=len(pbar)*5)
            # loss1 = loss_sup1 + lambda_ * loss_u                                                                          #总损失
            # loss1 = loss_sup1  + loss_cps_l1+ loss_u
            loss1 = loss_sup1 +loss_cps_l1 + loss_u
            loss1.backward()                                                                                             #损失回传
            optimizer1.step()
            optimizer11.step()
            optimizer22.step()


            # pseudo-label uncertainty calculate
            entropy1 = entropy(pred11_u_feature)
            entropies1.append(entropy1.mean().item())
            entropy2 = entropy(pred22_u_feature)
            entropies2.append(entropy2.mean().item())
            entropy3 = entropy(pred1_u_feature)
            entropies3.append(entropy3.mean().item())

            writer.add_scalar('train_sup1_loss', loss_sup1.item(), idx + len(pbar) * (epoch-1))   #写入log tensorboard?
            writer.add_scalar('train_cps1_loss', loss_cps1.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_loss1', loss1.item(), idx + len(pbar) * (epoch-1))

            if idx % args.log_freq == 0:
                logger.info("Train1: Epoch/Epochs {}/{}\t"
                            "iter/iters {}/{}\t"
                            "loss1 {:.3f}, loss_sup1 {:.3f}, loss_cps1 {:.3f}".format(epoch, args.num_epochs, idx, len(pbar),
                                                                                  loss1.item(), loss_sup1.item(), loss_cps1.item()))
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
            # #有监督
            # with torch.no_grad():
            #     uncertainty_map11 = torch.mean(torch.stack([pred1_feature, pred11_feature]), dim=0)                       #[4 2 256 256]
            #     uncertainty_map11 = -1.0 * torch.sum(uncertainty_map11*torch.log(uncertainty_map11 + 1e-6), dim=1, keepdim=True)#[4 1 256 256]
            #     uncertainty_map22 = torch.mean(torch.stack([pred22_feature, pred11_feature]), dim=0)
            #     uncertainty_map22 = -1.0 * torch.sum(uncertainty_map22*torch.log(uncertainty_map22 + 1e-6), dim=1, keepdim=True)
            #
            #     B, C = image.shape[0], image.shape[1]                                                                   #batch channel 4 3
            #     # for student 1
            #     x11 = unfolds(uncertainty_map11)                                                                        # B x C*kernel_size[0]*kernel_size[1] x L [4 256 256] (这里的C是1,与上面image的C不一样)    [4 256 256]将每个滑动窗口内的数据列成单独一列
            #     #print(x11.size())
            #     x11 = x11.view(B, 1, h, w, -1)                                                                          # B x C x h x w x L [4 1 16 16 256]
            #     #print(x11.size())
            #     x11_mean = torch.mean(x11, dim=(1, 2, 3))                                                               # B x L  [4 256]
            #     _, x11_max_index = torch.sort(x11_mean, dim=1, descending=True)                                         # B x L B x L [4 256] 排序结果   结果对应的数字在原来tensor中的index
            #     # for student 2
            #     x22 = unfolds(uncertainty_map22)  # B x C*kernel_size[0]*kernel_size[1] x L  (这里的C是1,与上面image的C不一样)    [4 256 256]将每个滑动窗口内的数据列成单独一列
            #     x22 = x22.view(B, 1, h, w, -1)  # B x C x h x w x L (这里的C是1,与上面image的C不一样) [4 1 16 16 256]
            #     x22_mean = torch.mean(x22, dim=(1, 2, 3))  # B x L [4 256]
            #     _, x22_max_index = torch.sort(x22_mean, dim=1, descending=True)  # B x L B x L [4 256]
            #     img_unfold = unfolds(imageA1).view(B, C, h, w, -1)  # B x C x h x w x L######################################################################################################
            #     lab_unfold = unfolds(label.float()).view(B, 1, h, w, -1)  # B x C x h x w x L\
            #     # print(label.shape)
            #     # return 0
            #     for i in range(B):
            #         img_unfold[i, :, :, :, x11_max_index[i, :topk]] = img_unfold[i, :, :, :, x22_max_index[i, -topk:]]
            #         img_unfold[i, :, :, :, x22_max_index[i, :topk]] = img_unfold[i, :, :, :, x11_max_index[i, -topk:]]
            #         lab_unfold[i, :, :, :, x11_max_index[i, :topk]] = lab_unfold[i, :, :, :, x22_max_index[i, -topk:]]
            #         lab_unfold[i, :, :, :, x22_max_index[i, :topk]] = lab_unfold[i, :, :, :, x11_max_index[i, -topk:]]
            #     image2 = folds(img_unfold.view(B, C*h*w, -1))        #[4 3 256 256]                                                   #image2是交换后的图像？
            #     label2 = folds(lab_unfold.view(B, 1*h*w, -1))       #[4 1 256 256]
            #     image2_dct_expanded_to_concat = []
            #     l = image2.size(0)
            # # print(image2.size(0))
            # # tensors = torch.chunk(image2, 4, dim=0)
            #     for i in range(l):
            #         single_image = image2[i]  # type torch.tensor   shape [3 256 256]
            #     # print(single_image.shape)
            #     # single_image = single_image.numpy()
            #         pil_image = ToPILImage()(single_image)
            #     # print(type(pil_image))
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
            #
            # loss11_sup = criterion(pred11_stage2, label2.squeeze(1).long())  ###################################################修改后的loss,损失函数好要用criterion_l？？？？？？？？？？？？？？？
            # loss22_sup = criterion(pred22_stage2, label2.squeeze(1).long())  #################################################修改后的loss,损失函数好要用criterion_l？？？？？？？？？？？？？？？？
            # loss1_sup = criterion(pred1_stage2, label2.squeeze(1).long())
            # loss_sup =  loss11_sup*2 + loss22_sup + loss1_sup
            # #无监督
            # with torch.no_grad():
            #     uncertainty_map1 = torch.mean(torch.stack([pred1_u_feature, pred11_u_feature]), dim=0)
            #     uncertainty_map1 = -1.0 * torch.sum(uncertainty_map1*torch.log(uncertainty_map1 + 1e-6), dim=1, keepdim=True)
            #     # print(pred1_u_feature.shape)#[4,1,256,256]
            #     # print((uncertainty_map1*torch.log(uncertainty_map1 + 1e-6)).shape)#[4,2,256,256]
            #     # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            #     uncertainty_map2 = torch.mean(torch.stack([pred22_u_feature, pred11_u_feature]), dim=0)
            #     uncertainty_map2 = -1.0 * torch.sum(uncertainty_map2*torch.log(uncertainty_map2 + 1e-6), dim=1, keepdim=True)
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
            #     imgu_unfold = unfolds(uimageA1).view(B, C, h, w, -1)  # B x C x h x w x L  [4,3,16,16,256]#########################################################################
            #     # print(imgu_unfold.shape)
            #     pseudo_unfold = unfolds(pseudo11.unsqueeze(1).float()).view(B, 1, h, w, -1)  # B x C x h x w x
            #     for i in range(B):
            #         imgu_unfold[i, :, :, :, x1_max_index[i, :topk]] = imgu_unfold[i, :, :, :, x2_max_index[i, -topk:]]
            #         imgu_unfold[i, :, :, :, x2_max_index[i, :topk]] = imgu_unfold[i, :, :, :, x1_max_index[i, -topk:]]
            #         pseudo_unfold[i, :, :, :, x1_max_index[i, :topk]] = pseudo_unfold[i, :, :, :, x2_max_index[i, -topk:]]
            #         pseudo_unfold[i, :, :, :, x2_max_index[i, :topk]] = pseudo_unfold[i, :, :, :, x1_max_index[i, -topk:]]
            #     uimage2 = folds(imgu_unfold.view(B, C*h*w, -1))
            #     pseudo = folds(pseudo_unfold.view(B, 1*h*w, -1)).squeeze(1).long()
            #     uimage2_dct_expanded_to_concat = []
            #     j=uimage2.size(0)
            #     for i in range(j):
            #         single_image = uimage2[i]  # type torch.tensor   shape [3 256 256]
            #         # single_image = single_image.numpy()
            #         pil_image = ToPILImage()(single_image)
            #         uimage2_single_dct = dct_transform.__call__(deepcopy(pil_image))
            #         uimage2_dct_expanded = uimage2_single_dct.unsqueeze(0)
            #         uimage2_dct_expanded_to_concat.append(uimage2_dct_expanded)
            #     uimage2_dct = torch.cat(uimage2_dct_expanded_to_concat, dim=0)  ############得到dct转换后的tensor uimage2_dct
            #     uimage2_dct = uimage2_dct.cuda()
            #
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
            #
            #
            # mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w1.clone(), conf_u_w1.clone()
            # mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w2.clone(), conf_u_w2.clone()
            # mask_u_w_cutmixed3, conf_u_w_cutmixed3 = mask_u_w3.clone(), conf_u_w3.clone()
            #
            # pseudo1 = mask_u_w3
            # pseudo11 = mask_u_w1
            # pseudo22 = mask_u_w2
            #
            #
            #
            #
            #
            #
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
            #
            #
            #
            # loss_u = (loss_cps2) * alpha
            # # lambda_ = sigmoid_rampup(current=idx + len(pbar) * (epoch - 1), rampup_length=len(pbar) * 5)
            # # loss = loss_sup + lambda_ * loss_u
            # # loss = loss_sup +loss_cps_l2 +loss_u
            # loss = loss_sup + loss_cps_l2 + loss_u
            # loss.backward()
            # optimizer1.step()
            # optimizer11.step()
            # optimizer22.step()

            # entropy1 = entropy(pred11_u_feature)
            # entropies1.append(entropy1.mean().item())
            # entropy2 = entropy(pred22_u_feature)
            # entropies2.append(entropy2.mean().item())
            # entropy3 = entropy(pred1_u_feature)
            # entropies3.append(entropy3.mean().item())
            #
            # writer.add_scalar('train_sup_loss', loss_sup.item(), idx + len(pbar) * (epoch - 1))
            # writer.add_scalar('train_cps_loss', loss_cps2.item(), idx + len(pbar) * (epoch - 1))
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
        # logger.info("Average: Epoch/Epoches {}/{}\t"#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        #             "train1 epoch loss {:.3f}\t"
        #             "train2 epoch loss {:.3f}\n".format(epoch, args.num_epochs,
        #                                                 np.mean(epoch_metrics.train_loss),
        #                                                 np.mean(epoch_metrics.train_loss2), ))
        logger.info("Average: Epoch/Epoches {}/{}\t"
                    "train1 epoch loss {:.3f}\n".format(epoch, args.num_epochs,
                                                        np.mean(epoch_metrics.train_loss),))

        if np.mean(epoch_metrics.train_loss) <= best_loss:                                                             #保存的是teacher模型
            best_loss = np.mean(epoch_metrics.train_loss)
            torch.save(model1.state_dict(), save_path + 'model1_best.pth'.format(best_epoch))
            torch.save(model2.state_dict(), save_path + 'model2_best.pth'.format(best_epoch))
            torch.save(model3.state_dict(), save_path + 'model3_best.pth'.format(best_epoch))

        torch.save(model1.state_dict(), save_path + 'model1_last.pth'.format(best_epoch))
        torch.save(model2.state_dict(), save_path + 'model2_last.pth'.format(best_epoch))
        torch.save(model3.state_dict(), save_path + 'model3_last.pth'.format(best_epoch))

    ############################
    # Save Metrics
    ############################
    data_frame = pd.DataFrame(
        data={'loss': metrics.train_loss},
        index=range(1, args.num_epochs + 1))#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    # data_frame = pd.DataFrame(
    #     data={'loss2': metrics.train_loss2},
    #     index=range(1, args.num_epochs + 1))
    data_frame.to_csv(project_path + 'train_loss.csv', index_label='Epoch')


    # uncertainty save to csv
    with open(os.path.join(project_path, 'entropies1.csv'),'w',newline='') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(['Iteration','Entropy'])
        for i,entropy in enumerate(entropies1):
            writer.writerow([i,entropy])
    with open(os.path.join(project_path, 'entropies2.csv'),'w',newline='') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(['Iteration','Entropy'])
        for i,entropy in enumerate(entropies2):
            writer.writerow([i,entropy])
    with open(os.path.join(project_path, 'entropies3.csv'),'w',newline='') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(['Iteration','Entropy'])
        for i,entropy in enumerate(entropies3):
            writer.writerow([i,entropy])


    plt.figure()
    plt.title("Loss")
    plt.plot(metrics.train_loss, label="Train")#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    # plt.plot(metrics.train_loss2, label="Train2")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(project_path + 'train_loss.png')
    plt.close()


    #pseudo-labels uncertainty
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




