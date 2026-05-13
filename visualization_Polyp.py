import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import argparse
from skimage import io
from skimage import color
from data.split import *
import torch
import torch.nn.functional as FU
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from data.dataset import ISICDataset
from models import deeplabv3
from utils.utils import ensure_dir

from model.semseg.segmentor import Segmentor
import yaml
from torchvision.transforms import Normalize, ToPILImage
from dataset.dct_transform import *
from copy import deepcopy
import pandas as pd
sep = '\\' if sys.platform[:3] == 'win' else '/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--project', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/runs/UCMT',
                        help='project path for saving results')
    parser.add_argument('--backbone', type=str, default='DeepLabv3p', choices=['DeepLabv3p', 'UNet'],
                        help='segmentation backbone')
    parser.add_argument('--image_root', type=str, default='YOUR_DATA_PATH', help='path_to_image')
    parser.add_argument('--gt_root', type=str, default='YOUR_DATA_PATH', help='path_to_GroundTruth')
    parser.add_argument('--is_cutmix', type=bool, default=False, help='cut mix')
    parser.add_argument('--labeled_percentage', type=float, default=0.15, help='the percentage of labeled data')
    parser.add_argument('--image_size', type=int, default=256, help='the size of images for training and testing')
    parser.add_argument('--batch_size', type=int, default=1, help='number of inputs per batch')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers to use for dataloader')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--num_classes', type=int, default=2, help='number of target categories')
    parser.add_argument('--model_weights', type=str, default='model3_best.pth', help='model weights')
    parser.add_argument('--config', type=str, default="/home/li/桌面/UCMT-main/configs/pascal/segformerb2_4x4.yaml", required=True)
    parser.add_argument('--config2', type=str, default="/home/li/桌面/UCMT-main/configs/pascal/r50_dct_4x4.yaml", required=True)
    parser.add_argument('--config3', type=str, default="/home/li/桌面/UCMT-main/configs/pascal/r50_4x4.yaml", required=True)
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def get_data(args):
    dataset = PolypDataset(args.image_root, args.gt_root, trainsize=args.image_size, stage='test', split_file_15=None, split_file_30=None, is_augmentation=False)
    test_dataloder = get_loader(args.image_root, args.gt_root, batchsize=args.batch_size, trainsize=args.image_size, stage='test', split_file_15=None, split_file_30=None, is_augmentation=False, shuffle=False, pin_memory=True)
    return test_dataloder, len(dataset)


def load_model(model_weights, in_channels, num_classes, backbone):
    model = deeplabv3.__dict__[backbone](in_channels=in_channels, out_channels=num_classes).to(device)
    print('#parameters:', sum(param.numel() for param in model.parameters()))
    model.load_state_dict(torch.load(model_weights))
    return model

def load_segformer(config,model_weights):
    cfg = yaml.load(open(config, "r"), Loader=yaml.Loader)
    model= Segmentor(cfg).to(device)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    print('#parameters:', sum(param.numel() for param in model.parameters()))
    model.load_state_dict(torch.load(model_weights))
    return model

def vis(is_debug=False):
    args = get_args()
    # Project Saving Path
    project_path = args.project + '_{}_label_{}/'.format(args.backbone, args.labeled_percentage)
    save_path = project_path + 'visualization_ba/'
    ensure_dir(save_path)
    # Load Data
    test_dataloader, length = get_data(args=args)
    iters = len(test_dataloader)
    iter_test_dataloader = iter(test_dataloader)
    if is_debug:
        pbar = range(100)
    else:
        pbar = range(iters)
    # Load model
    weights_path = project_path + 'weights/' + args.model_weights
    # model = load_model(model_weights=weights_path, in_channels=args.in_channels, num_classes=args.num_classes, backbone=args.backbone)
    # model = load_segformer(config=args.config2, model_weights=weights_path)
    # model = load_segformer(config=args.config, model_weights=weights_path)
    model = load_segformer(config=args.config3, model_weights=weights_path)
    model.eval()
    ############################
    # Evaluation
    ############################
    print('start evaluation')
    un_norm = Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    with torch.no_grad():
        for idx in pbar:
            image, label = iter_test_dataloader.next()
            image, label = image.to(device), label.to(device)
            # print(image.shape)
            # return 0
            pred = model(image)
            # pred = model(image)['out']
            B, C, H, W = label.shape
            pred = FU.interpolate(pred, size=[H, W], mode='bilinear', align_corners=False)
            image = FU.interpolate(image, size=[H, W], mode='bilinear', align_corners=False)
            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy() * 255. / 3.
            # torch.set_printoptions(profile="full")  # set print the whole tensor
            # print(label)
            # return 0
            label = label.squeeze(0).squeeze(0).long().cpu().numpy() * 255. / 3.

            # df = pd.DataFrame(label)
            # print(df)


            # np.set_printoptions(threshold=np.inf, edgeitems=50)
            # print(label)
            # return 0


            # print(image.dtype)
            # print("##################")
            # print(label.dtype)
            image = un_norm(image)
            image = image.squeeze(0).cpu().numpy() * 255.
            # print(image.shape)
            # return 0
            image = image.transpose([1, 2, 0])
            # print("@@@@@@@@@@@@@@@")
            # print(image.dtype)
            # print("!!!!!!!!!!!!!!")
            # print(image.astype('uint8').dtype)
            # print("$$$$$$$$$$$$$$$$$")
            # print(label.astype('uint8').dtype)
            # return 0
            io.imsave(save_path + str(idx) + '_img.png', image.astype('uint8'))

            # n_colors = len(colors)  # 颜色列表中的颜色数量
            # label_rgb = color.label2rgb(label, colors=colors, image_alpha=0)
            # label_rgb = color.label2rgb((label).astype('uint8'),colors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            # colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for red, green, blue
            #
            # # 将标签图像转换为RGB图像
            # label_rgb = color.label2rgb(label, image=None, colors=colors).astype(np.uint8)
            # io.imsave(save_path + str(idx) + '_lbl.png',label_rgb)
            colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for red, green, blue

            # 将标签图像转换为RGB图像
            label_rgb = color.label2rgb(label, colors=colors)

            # 确保图像数据是 uint8 类型
            label_rgb = label_rgb.astype(np.uint8)
            io.imsave(save_path + str(idx) + '_lbl.png', label_rgb)

            pred_rgb = color.label2rgb(pred, colors=colors)

            # 确保图像数据是 uint8 类型
            pred_rgb = pred_rgb.astype(np.uint8)
            io.imsave(save_path + str(idx) + '_prd.png',pred_rgb)
            # io.imsave(save_path + str(idx) + '_prd.png',
            #           color.label2rgb((pred),
            #                           colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]).astype('uint8'))
            print('itr/itrs: {}/{}'.format(idx + 1, len(pbar)))

    print('EVAL FINISHED!')


if __name__ == '__main__':
    vis()
