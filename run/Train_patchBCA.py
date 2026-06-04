import os
import torch
import tqdm
import sys
import logging
import cv2
import torch.nn as nn
import torch.cuda as cuda
import torch.distributed as dist

from torch.optim import AdamW, SGD
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime
filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.dataloader import *
from lib.optim import *
from lib import *

def set_logging(opt):
    # logging
    log_dir = opt.Train.Checkpoint.checkpoint_dir
    os.makedirs(log_dir, exist_ok=True)
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"training_{timestamp}.log")

    # 基础配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),  # 文件处理器
            logging.StreamHandler()  # 控制台处理器
        ]
    )

    # 获取logger
    logger = logging.getLogger(__name__)
    return logger



def calculate_iou(pred, target):
    """
    计算二值化掩码的IOU

    Args:
        pred: 预测的二值化掩码 (0或1)
        target: 真值二值化掩码 (0或1)

    Returns:
        iou: IOU值
    """
    # 处理不同的维度
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    # 确保都是二值图像
    pred_bin = (pred > 0.5).float()
    target_bin = (target > 0.5).float()

    # 计算交集和并集
    intersection = (pred_bin * target_bin).sum(dim=(1, 2))
    union = (pred_bin + target_bin).clamp(0, 1).sum(dim=(1, 2))

    # 避免除零
    iou = torch.where(union == 0,
                      torch.ones_like(intersection),  # 当union==0时，返回1
                      intersection / (union + 1e-6))  # 否则正常计算IOU


    return iou  # 返回标量值

def Eval_model(model, val_loader):
    with torch.no_grad():
        model.eval()
        IOU = []
        for sample in val_loader:
            gt = sample['gt'].cuda()
            del sample['gt']
            sample = to_cuda(sample)
            out = model(sample)
            pred = out['pred']
            pred = torch.sigmoid(pred)
            IOUs = calculate_iou(pred, gt)
            IOU+=(IOUs.cpu().tolist())

    ret_IOU = np.mean(IOU)
    return ret_IOU

def train(opt, args):
    os.makedirs(opt.Train.Checkpoint.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(
        opt.Train.Checkpoint.checkpoint_dir, 'debug'), exist_ok=True)

    logger = set_logging(opt)

    logger.info(opt)

    #训练集加载type: "PatchDataset" root: ""PatchesDataset/"
    train_dataset = eval(opt.Train.Dataset.type)(
        root=opt.Train.Dataset.root, transform_list=opt.Train.Dataset.transform_list,type = 'train')
    logger.info('number of train scenes: {}'.format(len(train_dataset)))

    val_dataset = eval(opt.Eval.Dataset.type)(
        root=opt.Eval.Dataset.root, transform_list=opt.Eval.Dataset.transform_list, type='val')
    logger.info('number of val scenes: {}'.format(len(val_dataset)))

    if args.device_num > 1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.device_num)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=opt.Train.Dataloader.batch_size,
                                    shuffle=True,
                                    sampler=train_sampler,
                                    num_workers=opt.Train.Dataloader.num_workers,
                                    pin_memory=opt.Train.Dataloader.pin_memory,
                                    drop_last=True)

    val_loader = data.DataLoader(dataset=val_dataset,
                                   batch_size=opt.Train.Dataloader.batch_size,
                                   shuffle=False,
                                   num_workers=opt.Train.Dataloader.num_workers,
                                   drop_last=False)
    #模型初始化
    model = eval(opt.Model.name)(channels=opt.Model.channels,
                                 output_stride=opt.Model.output_stride,
                                 pretrained=opt.Model.pretrained)

    for name, param in model.named_parameters():
        logger.info("{} {}".format(name, param.shape))

    if args.device_num > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    else:
        model = model.cuda()

    #优化器
    backbone_params = nn.ParameterList()
    decoder_params = nn.ParameterList()

    for name, param in model.named_parameters():
        if 'backbone' in name:
            if 'backbone.layer' in name:
                backbone_params.append(param)
            else:
                pass
        else:
            decoder_params.append(param)

    params_list = [{'params': backbone_params}, {
        'params': decoder_params, 'lr': opt.Train.Optimizer.lr * 10}]
    optimizer = eval(opt.Train.Optimizer.type)(
        params_list, opt.Train.Optimizer.lr, weight_decay=opt.Train.Optimizer.weight_decay)


    if opt.Train.Optimizer.mixed_precision is True:
        scaler = GradScaler()
    else:
        scaler = None

    #学习率
    scheduler = eval(opt.Train.Scheduler.type)(optimizer, gamma=opt.Train.Scheduler.gamma,
                                               minimum_lr=opt.Train.Scheduler.minimum_lr,
                                               max_iteration=len(
                                                   train_loader) * opt.Train.Scheduler.epoch,
                                               warmup_iteration=opt.Train.Scheduler.warmup_iteration)


    if args.local_rank <= 0 and args.verbose is True:
        epoch_iter = tqdm.tqdm(range(1, opt.Train.Scheduler.epoch + 1), desc='Epoch', total=opt.Train.Scheduler.epoch,
                               position=0, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
    else:
        epoch_iter = range(1, opt.Train.Scheduler.epoch + 1)


    best = 0
    for epoch in epoch_iter:
        model.train()
        loss_ = []
        #loss_BCE = []
        #loss_boundary =[]
        if args.local_rank <= 0 and args.verbose is True:
            step_iter = tqdm.tqdm(enumerate(train_loader, start=1), desc='Iter', total=len(
                train_loader), position=1, leave=False, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
            if args.device_num > 1:
                train_sampler.set_epoch(epoch)
        else:
            step_iter = enumerate(train_loader, start=1)

        for i, sample in step_iter:
            optimizer.zero_grad()
            if opt.Train.Optimizer.mixed_precision is True:
                with autocast():
                    sample = to_cuda(sample)
                    out = model(sample)

                    scaler.scale(out['loss']).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
            else:
                sample = to_cuda(sample)
                out = model(sample)
                out['loss'].backward()
                optimizer.step()
                scheduler.step()

            if args.local_rank <= 0 and args.verbose is True:
                step_iter.set_postfix({'loss': out['loss'].item(),'debug_loss':out['debug_loss']})
            loss_.append(out['loss'].item())
            #loss_BCE.append(out['function_loss']['Dice_BCE_loss'])
            #loss_boundary.append(out['function_loss']['Boundary_loss'])
        logger.info(f"epoch:{epoch},loss:{np.mean(loss_)},debug_loss:{out['debug_loss']}")

        if args.local_rank <= 0:

            if epoch % opt.Train.Checkpoint.checkpoint_epoch == 0:
                logging.info('Starting eval...')
                logging.info('Running testing in epoch {}'.format(epoch))


                cur_metric = Eval_model(model, val_loader)

                logging.info(
                    '============================== current metric is {} ================================='.format(cur_metric))
                #
                logging.info('Eval done...')
                if cur_metric > best:
                    best = cur_metric
                    logging.info(
                        '======================================================================================')
                    logging.info(
                        '============================== best metric is {} ================================='.format(
                            best))
                    logging.info(
                        '======================================================================================')

                save_dir = os.path.join(opt.Train.Checkpoint.checkpoint_dir, str(epoch)+'.pth')
                torch.save(model.module.state_dict() if args.device_num > 1 else model.state_dict(), save_dir)

            if args.debug is True:
                debout = debug_tile(out)
                cv2.imwrite(os.path.join(
                    opt.Train.Checkpoint.checkpoint_dir, 'debug', str(epoch) + '.png'), debout)


if __name__ == '__main__':
    args = parse_args()   #--verbose --debug
    config= 'configs/UACAPatchNet_BCAASPP.yaml' #训练设置
    opt = load_config(config)
    train(opt, args)
#CUDA_VISIBLE_DEVICES=1 python run/Train_patch.py --verbose --debug

