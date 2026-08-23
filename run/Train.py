import os
import torch
import tqdm
import sys

import cv2
import torch.nn as nn
import torch.cuda as cuda
import torch.distributed as dist

from torch.optim import Adam, SGD
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.dataloader import *
from lib.optim import *
from lib import *

def train(opt, args):
    # PolypDataset in this repo takes explicit img_root/mask_root (it was
    # repurposed for the patch refiner). Derive them from Train.Dataset.root
    # + subdir names ('images'/'masks' by default; override via
    # img_subdir/mask_subdir in the config if your layout differs).
    _root = opt.Train.Dataset.root
    _img_sub = getattr(opt.Train.Dataset, 'img_subdir', 'images')
    _mask_sub = getattr(opt.Train.Dataset, 'mask_subdir', 'masks')
    train_dataset = eval(opt.Train.Dataset.type)(
        img_root=os.path.join(_root, _img_sub),
        mask_root=os.path.join(_root, _mask_sub),
        transform_list=opt.Train.Dataset.transform_list)

    if args.device_num > 1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.device_num)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=opt.Train.Dataloader.batch_size,
                                    shuffle=train_sampler is None,
                                    sampler=train_sampler,
                                    num_workers=opt.Train.Dataloader.num_workers,
                                    pin_memory=opt.Train.Dataloader.pin_memory,
                                    drop_last=True)

    _model_kwargs = dict(channels=opt.Model.channels,
                         output_stride=opt.Model.output_stride,
                         pretrained=opt.Model.pretrained)
    # Forward FCT knobs only when the config sets them, so vanilla models
    # (UACANet: 3 args) still build; UACANet_FCT picks these up.
    for _k in ('fct_weight', 'fct_warmup_iters', 'fct_use_vflip',
               'use_flip_consistency', 'fct_supervise_flips'):
        if hasattr(opt.Model, _k):
            _model_kwargs[_k] = getattr(opt.Model, _k)
    model = eval(opt.Model.name)(**_model_kwargs)

    if args.device_num > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    else:
        model = model.cuda()

    backbone_params = nn.ParameterList()
    decoder_params = nn.ParameterList()

    for name, param in model.named_parameters():
        # Match both naming conventions: UACANet/BACFR name the encoder
        # `resnet.*`; older models use `backbone.*`. Backbone stages (layer*)
        # train at base lr, the stem stays frozen, and the decoder (everything
        # else) trains at 10x lr. Without the `resnet` case the whole UACANet
        # backbone would land in decoder_params and train at 10x lr.
        if 'backbone' in name or 'resnet' in name:
            if 'layer' in name:
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

    scheduler = eval(opt.Train.Scheduler.type)(optimizer, gamma=opt.Train.Scheduler.gamma,
                                               minimum_lr=opt.Train.Scheduler.minimum_lr,
                                               max_iteration=len(
                                                   train_loader) * opt.Train.Scheduler.epoch,
                                               warmup_iteration=opt.Train.Scheduler.warmup_iteration)
    model.train()

    if args.local_rank <= 0 and args.verbose is True:
        epoch_iter = tqdm.tqdm(range(1, opt.Train.Scheduler.epoch + 1), desc='Epoch', total=opt.Train.Scheduler.epoch,
                               position=0, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
    else:
        epoch_iter = range(1, opt.Train.Scheduler.epoch + 1)

    for epoch in epoch_iter:
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
                step_iter.set_postfix({'loss': out['loss'].item()})

        if args.local_rank <= 0:
            os.makedirs(opt.Train.Checkpoint.checkpoint_dir, exist_ok=True)
            os.makedirs(os.path.join(
                opt.Train.Checkpoint.checkpoint_dir, 'debug'), exist_ok=True)
            if epoch % opt.Train.Checkpoint.checkpoint_epoch == 0:
                torch.save(model.module.state_dict() if args.device_num > 1 else model.state_dict(
                ), os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'))

            if args.debug is True:
                debout = debug_tile(out)
                cv2.imwrite(os.path.join(
                    opt.Train.Checkpoint.checkpoint_dir, 'debug', str(epoch) + '.png'), debout)

    if args.local_rank <= 0:
        torch.save(model.module.state_dict() if args.device_num > 1 else model.state_dict(
        ), os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'latest.pth'))


if __name__ == '__main__':
    args = parse_args()
    opt = load_config(args.config)
    train(opt, args)
