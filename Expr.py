# # import torch.cuda as cuda
# #
# # from utils.utils import *
# # from run import *
# #
# # if __name__ == "__main__":
# #     args = parse_args()
# #     opt = load_config(args.config)
# #
# #     train(opt, args)
# #     cuda.empty_cache()
# #     if args.local_rank <= 0:
# #         test(opt, args)
# #         evaluate(opt, args)
#
#
#
# import os
# import torch
# import tqdm
# import sys
# import logging
# import cv2
# import torch.nn as nn
# import torch.cuda as cuda
# import torch.distributed as dist
#
# from torch.optim import Adam, SGD
# from torch.cuda.amp import GradScaler, autocast
# from torch.utils.data.distributed import DistributedSampler
# from datetime import datetime
# filepath = os.path.split(os.path.abspath(__file__))[0]
# repopath = os.path.split(filepath)[0]
# sys.path.append(repopath)
#
# from utils.dataloader import *
# from lib.optim import *
# from lib import *
#
# def set_logging(opt):
#     # logging
#     log_dir = opt.Train.Checkpoint.checkpoint_dir
#     os.makedirs(log_dir, exist_ok=True)
#     # 生成带时间戳的日志文件名
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_filename = os.path.join(log_dir, f"training_{timestamp}.log")
#
#     # 基础配置
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_filename, encoding='utf-8'),  # 文件处理器
#             logging.StreamHandler()  # 控制台处理器
#         ]
#     )
#
#     # 获取logger
#     logger = logging.getLogger(__name__)
#     return logger
#
#
#
# def calculate_iou(pred, target):
#     """
#     计算二值化掩码的IOU
#
#     Args:
#         pred: 预测的二值化掩码 (0或1)
#         target: 真值二值化掩码 (0或1)
#
#     Returns:
#         iou: IOU值
#     """
#     # 处理不同的维度
#     if pred.dim() == 4:
#         pred = pred.squeeze(1)
#     if target.dim() == 4:
#         target = target.squeeze(1)
#
#     # 确保都是二值图像
#     pred_bin = (pred > 0.5).float()
#     target_bin = (target > 0.5).float()
#
#     # 计算交集和并集
#     intersection = (pred_bin * target_bin).sum(dim=(1, 2))
#     union = (pred_bin + target_bin).clamp(0, 1).sum(dim=(1, 2))
#
#     # 避免除零
#     iou = torch.where(union == 0,
#                       torch.ones_like(intersection),  # 当union==0时，返回1
#                       intersection / (union + 1e-6))  # 否则正常计算IOU
#
#
#     return iou  # 返回标量值
#
# def Eval_model(model, val_loader):
#     with torch.no_grad():
#         model.eval()
#         IOU = []
#         for sample in val_loader:
#             gt = sample['gt'].cuda()
#             del sample['gt']
#             sample = to_cuda(sample)
#             out = model(sample)
#             pred = out['pred']
#             pred = torch.sigmoid(pred)
#             IOUs = calculate_iou(pred, gt)
#             IOU+=(IOUs.cpu().tolist())
#
#     ret_IOU = np.mean(IOU)
#     return ret_IOU
#
# def train(opt, args):
#
#
#     out_dir = 'results/'
#
#     #训练集加载type: "PatchDataset" root: ""PatchesDataset/"
#     train_dataset = eval(opt.Train.Dataset.type)(
#         root=opt.Train.Dataset.root, transform_list=opt.Train.Dataset.transform_list,type = 'train')
#
#
#     val_dataset = eval(opt.Eval.Dataset.type)(
#         root=opt.Eval.Dataset.root, transform_list=opt.Eval.Dataset.transform_list, type='val')
#
#
#
#     train_sampler = None
#
#     train_loader = data.DataLoader(dataset=train_dataset,
#                                     batch_size=opt.Train.Dataloader.batch_size,
#                                     shuffle=True,
#                                     sampler=train_sampler,
#                                     num_workers=opt.Train.Dataloader.num_workers,
#                                     pin_memory=opt.Train.Dataloader.pin_memory,
#                                     drop_last=True)
#
#     val_loader = data.DataLoader(dataset=val_dataset,
#                                    batch_size=opt.Train.Dataloader.batch_size,
#                                    shuffle=True,
#                                    num_workers=opt.Train.Dataloader.num_workers,
#                                    drop_last=False)
#     #模型初始化
#     model = eval(opt.Model.name)(channels=opt.Model.channels,
#                                  output_stride=opt.Model.output_stride,
#                                  pretrained=opt.Model.pretrained)
#     model.load_state_dict(torch.load('checkpoints/UACAPatchNet-1/6.pth'), strict=True)
#     model = model.cuda()
#
#     model.eval()
#
#     with torch.no_grad():
#         save_dir = out_dir+'train_patches'
#         os.makedirs(save_dir, exist_ok=True)
#         step_iter = enumerate(train_loader, start=1)
#         num = 0
#         for i, sample in step_iter:
#             if i==4:
#                 break
#             del sample['gt']
#             sample = to_cuda(sample)
#             out = model(sample)
#             pred = out['pred'].squeeze(1)
#             pred = (torch.sigmoid(pred)>0.5).float()
#             for _ in range(len(pred)):
#                 num+=1
#                 cv2.imwrite(os.path.join(save_dir,str(num)+'.png'), pred[_].cpu().numpy().astype(np.uint8) * 255)
#         num = 0
#         save_dir = out_dir + 'val_patches'
#         os.makedirs(save_dir, exist_ok=True)
#         step_iter = enumerate(val_loader, start=1)
#
#         for i, sample in step_iter:
#             if i == 4:
#                 break
#             del sample['gt']
#             sample = to_cuda(sample)
#             out = model(sample)
#             pred  =out['pred'].squeeze(1)
#             pred = (torch.sigmoid(pred)>0.5).float()
#             for _ in range(len(pred)):
#                 num+=1
#                 cv2.imwrite(os.path.join(save_dir,str(num)+'.png'), pred[_].cpu().numpy().astype(np.uint8) * 255)
#
# if __name__ == '__main__':
#     args = parse_args()   #--verbose --debug
#     config= 'configs/UACAPatchNet.yaml' #训练设置
#     opt = load_config(config)
#     train(opt, args)
# #CUDA_VISIBLE_DEVICES=0 python run/Train_patch.py --verbose --debug
import cv2
import numpy as np
path = "/home/yassine/projects/UACANet-main/results_cl/paper_results/UACAPatchNet-15-pranet-fusion-3/CVC-300/149.png"
print(np.unique(cv2.imread(path)))


