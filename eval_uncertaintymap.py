import os
import sys
import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as FU
from torch.utils.data import DataLoader
from data.dataset import ISICDataset
from models import deeplabv3
from utils.utils import dice_score_batch

from model.semseg.segmentor import Segmentor
import yaml
from torchvision.transforms import Normalize, ToPILImage
from dataset.dct_transform import *
from copy import deepcopy
from PIL import Image
sep = '\\' if sys.platform[:3] == 'win' else '/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--project', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/runs/UCMT', help='project path for saving results')
    parser.add_argument('--backbone', type=str, default='UNet', choices=['DeepLabv3p', 'UNet'], help='segmentation backbone')
    parser.add_argument('--data_path', type=str, default='YOUR_DATA_PATH', help='path to the data')
    parser.add_argument('--is_cutmix', type=bool, default=False, help='cut mix')
    parser.add_argument('--labeled_percentage', type=float, default=0.05, help='the percentage of labeled data')
    parser.add_argument('--image_size', type=int, default=256, help='the size of images for training and testing')
    parser.add_argument('--batch_size', type=int, default=4, help='number of inputs per batch')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers to use for dataloader')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--num_classes', type=int, default=2, help='number of target categories')
    parser.add_argument('--model1_weights', type=str, default='model1_best.pth', help='model weights')
    parser.add_argument('--model2_weights', type=str, default='model2_best.pth', help='model weights')
    parser.add_argument('--model3_weights', type=str, default='model3_best.pth', help='model weights')
    parser.add_argument('--config', type=str, default="/home/li/桌面/UCMT-main/configs/pascal/segformerb2_4x4.yaml",required=True)
    parser.add_argument('--config2', type=str, default="/home/li/桌面/UCMT-main/configs/pascal/r50_dct_4x4.yaml",required=True)
    parser.add_argument('--config3', type=str, default="/home/li/桌面/UCMT-main/configs/pascal/r50_4x4.yaml",required=True)
    parser.add_argument('--topk', type=float, default=2, help='top k')
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def get_data(args):
    test_set = ISICDataset(image_path=args.data_path, stage='test', image_size=args.image_size, is_augmentation=False)
    test_dataloder = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    return test_dataloder, len(test_set)


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


def eval(is_debug=False):
    args = get_args()
    h, w = args.image_size // 16, args.image_size // 16
    s = h
    topk = args.topk
    unfolds = torch.nn.Unfold(kernel_size=(h, w), stride=s).to(device)  # ？？？？？？？？？？？？？？？？？
    folds = torch.nn.Fold(output_size=(args.image_size, args.image_size), kernel_size=(h, w), stride=s).to(
        device)  # fold,unfold这两个的作用是什么？
    # Project Saving Path
    project_path = args.project + '_{}_label_{}/'.format(args.backbone, args.labeled_percentage)
    # Load Data
    test_dataloader, length = get_data(args=args)
    iters = len(test_dataloader)
    iter_test_dataloader = iter(test_dataloader)
    if is_debug:
        pbar = range(10)
        length = 10 * args.batch_size
    else:
        pbar = range(iters)
    # Load model
    weights1_path = project_path + 'weights/' + args.model1_weights
    weights2_path = project_path + 'weights/' + args.model2_weights
    weights3_path = project_path + 'weights/' + args.model3_weights
    # model = load_model(model_weights=weights_path, in_channels=args.in_channels, num_classes=args.num_classes, backbone=args.backbone)
    model2 = load_segformer(config=args.config2, model_weights=weights2_path)
    model1=load_segformer(config=args.config, model_weights=weights1_path)
    model3 = load_segformer(config=args.config3, model_weights=weights3_path)
    dct_transform = DCTTransform()

    model1.eval()
    model2.eval()
    model3.eval()
    ############################
    # Evaluation
    ############################
    print('start evaluation')
    results = {i: [] for i in range(args.num_classes)}
    with torch.no_grad():
        for idx in pbar:
            image, label = next(iter_test_dataloader)
            l = image.size(0)
            image2_dct_expanded_to_concat = []
            for i in range(l):
                #print(i)
                single_image = image[i]  # type torch.tensor   shape [3 256 256]
                # print(single_image.shape)
                # print("11111111111111111111111")
                #return 0
                # single_image = single_image.numpy()
                pil_image = ToPILImage()(single_image)
                # print(type(pil_image))
                image2_single_dct = dct_transform.__call__(deepcopy(pil_image))
                image2_dct_expanded = image2_single_dct.unsqueeze(0)
                image2_dct_expanded_to_concat.append(image2_dct_expanded)
            image2_dct = torch.cat(image2_dct_expanded_to_concat, dim=0)  ############得到dct转换后的tensor image2_dct
            image2_dct = image2_dct.to(device)
            image, label = image.to(device), label.to(device)
            # pred = model(image)['out']
            pred22 = model2(image2_dct,([256,256]))
            pred11 = model1(image)
            pred1 = model3(image)
            B, C, H, W = label.shape
            #print(label.shape)#[4 1 512 512]
            #print(image.shape)#[4 3 256 256]
            # print(image2_dct.shape)#[4 64 64 64]
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(pred.shape)#[4 2 256 256]
            # print("########################")
            # return 0
            pred11 = FU.interpolate(pred11, size=[H, W], mode='bilinear', align_corners=False)
            pred22 = FU.interpolate(pred22, size=[H, W], mode='bilinear', align_corners=False)
            pred1 = FU.interpolate(pred1, size=[H, W], mode='bilinear', align_corners=False)
            # print("###########################################################################")
            # print(pred)
            # print(pred.shape)#[4 2 256 256]
            # return 0
            pred11_feature = torch.softmax(pred11, dim=1)
            pred11 = torch.argmax(pred11_feature, dim=1)
            pred22_feature = torch.softmax(pred22, dim=1)
            pred22 = torch.argmax(pred22_feature, dim=1)
            pred1_feature = torch.softmax(pred1, dim=1)
            pred1 = torch.argmax(pred1_feature, dim=1)
            label = label.squeeze(1).long()
            label_onehot = torch.nn.functional.one_hot(label, num_classes=args.num_classes).permute(0, 3, 1, 2).contiguous()
            pred11_onehot = torch.nn.functional.one_hot(pred11, num_classes=args.num_classes).permute(0, 3, 1, 2).contiguous()
            pred22_onehot = torch.nn.functional.one_hot(pred22, num_classes=args.num_classes).permute(0, 3, 1, 2).contiguous()
            pred1_onehot = torch.nn.functional.one_hot(pred1, num_classes=args.num_classes).permute(0, 3, 1, 2).contiguous()




            uncertainty_map11 = torch.mean(torch.stack([pred1_feature, pred11_feature]), dim=0)  # [4 2 256 256]
            # uncertainty_map11 = -1.0 * torch.sum(uncertainty_map11*torch.log(uncertainty_map11 + 1e-6), dim=1, keepdim=True)#[4 1 256 256]
            uncertainty_map22 = torch.mean(torch.stack([pred22_feature, pred11_feature]), dim=0)
            # uncertainty_map22 = -1.0 * torch.sum(uncertainty_map22*torch.log(uncertainty_map22 + 1e-6), dim=1, keepdim=True)
            # print(image.shape)
            # return 0
            # for i in range(8):
            #     uncertainty_map11_single = uncertainty_map11[i, 0, :, :]  # 取第一个批次和第一个通道
            #     uncertainty_map22_single = uncertainty_map22[i, 0, :, :]  # 取第一个批次和第一个通道
            #
            #     # 确保不确定性图的数据类型是uint8，范围在0-255之间
            #     uncertainty_map11_single = (uncertainty_map11_single * 255).cpu().numpy().astype(np.uint8)
            #     uncertainty_map22_single = (uncertainty_map22_single * 255).cpu().numpy().astype(np.uint8)
            #
            #     # 使用OpenCV的colormap进行可视化
            #     colormap11 = cv2.applyColorMap(uncertainty_map11_single, cv2.COLORMAP_JET)  # JET只是一个例子，你可以选择其他colormap
            #     colormap22 = cv2.applyColorMap(uncertainty_map22_single, cv2.COLORMAP_JET)
            #
            #     # 显示或保存图像
            #     # cv2.imshow('Uncertainty Map 11', colormap11)
            #     # cv2.imshow('Uncertainty Map 22', colormap22)
            #     # cv2.waitKey(0)  # 等待任意键按下
            #     # cv2.destroyAllWindows()  # 关闭所有窗口
            #     save_folder_uncertainty_map11 = '/home/li/桌面/UCMT-main/images/uncertainty_map11/'
            #     image_path_uncertainty_map11 = save_folder_uncertainty_map11 + str(idx) + str(i) + 'black_white_image.jpg'
            #     save_folder_uncertainty_map22 = '/home/li/桌面/UCMT-main/images/uncertainty_map22/'
            #     image_path_uncertainty_map22 = save_folder_uncertainty_map22 + str(idx) + str(i) + 'black_white_image.jpg'
            #     # 如果你想要保存图像
            #     cv2.imwrite(image_path_uncertainty_map11, colormap11)
            #     cv2.imwrite(image_path_uncertainty_map22, colormap22)
            B, C = image.shape[0], image.shape[1]  # batch channel 4 3
            # print(B)
                # for student 1
            x11 = unfolds(uncertainty_map11)  # B x C*kernel_size[0]*kernel_size[1] x L [4 256 256] (这里的C是1,与上面image的C不一样)    [4 256 256]将每个滑动窗口内的数据列成单独一列
                # print(x11.size())
            x11 = x11.view(B, 1, h, w, -1)  # B x C x h x w x L [4 1 16 16 256]
                # print(x11.size())
            x11_mean = torch.mean(x11, dim=(1, 2, 3))  # B x L  [4 256]
            _, x11_max_index = torch.sort(x11_mean, dim=1, descending=True)  # B x L B x L [4 256] 排序结果   结果对应的数字在原来tensor中的index
                # for student 2
            x22 = unfolds(uncertainty_map22)  # B x C*kernel_size[0]*kernel_size[1] x L  (这里的C是1,与上面image的C不一样)    [4 256 256]将每个滑动窗口内的数据列成单独一列
            x22 = x22.view(B, 1, h, w, -1)  # B x C x h x w x L (这里的C是1,与上面image的C不一样) [4 1 16 16 256]
            x22_mean = torch.mean(x22, dim=(1, 2, 3))  # B x L [4 256]
            _, x22_max_index = torch.sort(x22_mean, dim=1, descending=True)  # B x L B x L [4 256]
            img_unfold = unfolds(image).view(B, C, h, w, -1)  # B x C x h x w x L######################################################################################################
            # lab_unfold = unfolds(label.float()).view(B, 1, h, w, -1)  # B x C x h x w x L\
                # print(label.shape)
                # return 0
            for i in range(B):
                img_unfold[i, :, :, :, x11_max_index[i, :topk]] = img_unfold[i, :, :, :, x22_max_index[i, -topk:]]
                img_unfold[i, :, :, :, x22_max_index[i, :topk]] = img_unfold[i, :, :, :, x11_max_index[i, -topk:]]
                # lab_unfold[i, :, :, :, x11_max_index[i, :topk]] = lab_unfold[i, :, :, :, x22_max_index[i, -topk:]]
                # lab_unfold[i, :, :, :, x22_max_index[i, :topk]] = lab_unfold[i, :, :, :, x11_max_index[i, -topk:]]
            image2 = folds(img_unfold.view(B, C * h * w, -1))  # [8 3 256 256]                                                   #image2是交换后的图像？
            img = image2.clone()
            # for i in range(4):
            #     print(img[i].shape)

            # img=image2.clone()
            # img=img.cpu()
            # print(img.shape)
            # return 0
            # label2 = folds(lab_unfold.view(B, 1 * h * w, -1))  # [4 1 256 256]
            save_dir = '/home/li/桌面/UCMT-main/images/image2/'
            print(image2)
            # for i in range(4):
            #     # 提取当前图像
            #     img_img=image2.cpu()
            #     img = img_img[i].permute(1, 2, 0).numpy()  # 将通道维度移到最后，并转换为numpy数组
            #     image_path =  str(idx) + str(i) + 'image.png'
            #
            #     # 创建文件名
            #     filename = os.path.join(save_dir, image_path)
            #
            #     # 使用 Matplotlib 保存图像
            #     plt.imsave(filename, img, cmap='rgb')  # 对于彩色图像，不需要指定 cmap
            return 0
                # for j in range(4):
                #     image2_to_visualize = image2[j].cpu().numpy()  # 假设image2在GPU上，我们需要将其移动到CPU
                #     image2_to_visualize = np.transpose(image2_to_visualize,
                #                                        (1, 2, 0))  # 将通道维度移动到末尾，以匹配matplotlib的期望顺序(H, W, C)
                #
                #     # 如果你处理的是彩色图像，确保数据类型和值范围正确
                #     if image2_to_visualize.dtype == np.float32:
                #         image2_to_visualize = (image2_to_visualize - image2_to_visualize.min()) / (
                #                 image2_to_visualize.max() - image2_to_visualize.min())  # 归一化到[0, 1]
                #         image2_to_visualize = np.clip(image2_to_visualize, 0, 1)  # 确保值在[0, 1]范围内
                #
                #     # plt.imsave('/home/li/桌面/UCMT-main/images/image2/image_{}.png'.format(j), image2_to_visualize)
                #
                #     # 显示图像
                #     # plt.imshow(image2_to_visualize)
                #     # plt.axis('off')  # 关闭坐标轴显示
                #     # plt.show()  # 显示图像

                # for i in range(8):
            #     black_white_image_h = (pred == 1).float()  # 这将返回一个形状相同的浮点张量，其中1表示白色，0表示黑色
            #     # 如果需要，可以将浮点张量缩放到0-255的整数范围
            #     black_white_image_h_uint8 = (black_white_image_h * 255).byte()
            #     # 选择一个图像进行可视化（如果有多个图像在批量中）
            #     image_to_save_h = black_white_image_h_uint8[i]  # 取第一个图像作为示例
            #     image_to_save_h = image_to_save_h.cpu()
            #     save_folder_h = '/home/li/桌面/UCMT-main/images/f2/'  # 确保这个文件夹已经存在
            #     image_path_h = save_folder_h + str(idx) + str(i) + 'black_white_image.jpg'
            #     # 使用matplotlib保存图像为jpg格式
            #     plt.imsave(image_path_h, image_to_save_h, cmap='gray')

                # image_image = image.cpu()
                # # 如果 image 不是 (C, H, W) 格式，而是 (B, C, H, W)（其中 B 是批大小），则选择一个图像来保存
                # # 例如，选择第一个图像
                # image_image = image_image[i]
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
                # image_path = save_path + str(idx) + str(i) + 'image.jpg'
                # # 保存图像为 JPEG 格式
                # # pil_image.save(image_path, 'JPEG')
                # pil_image.save(image_path)



                # print(uncertainty_map11.shape)
                # return 0















    #         dices = dice_score_batch(prediction=pred11_onehot, target=label_onehot).cpu().numpy()
    #
    #         for b in range(len(dices)):
    #             for i in range(args.num_classes):
    #                 results[i].append(dices[b][i])
    #         print('itr/itrs: {}/{}, label: {}, pred: {}'.format(idx + 1, len(pbar), label.shape, pred.shape))
    # # save results
    # data_frame = pd.DataFrame(
    #     data={i: results[i] for i in range(args.num_classes)},
    #     index=range(1, length + 1))
    # data_frame.to_csv(project_path + '/' + 'evaluation.csv', index_label='Index')
    # result = data_frame.values
    # avg_score = np.mean(result, axis=0)
    # with open(project_path+'/performance.txt', 'w') as f:
    #     f.writelines('metric is {} \n'.format(avg_score[1:]))
    # print('AVG Score:', avg_score[1:])
    # print('EVAL FINISHED!')


if __name__ == '__main__':
    eval()