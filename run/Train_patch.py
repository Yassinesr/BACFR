import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import tqdm
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.distributed as dist
import torch.utils.data as data

from torch.optim import Adam, AdamW, SGD
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime



from utils.utils import *
from utils.dataloader import *
from utils.utils import parse_args, load_config, to_cuda
from lib.optim import *
from lib import *


def set_logging(opt):
    log_dir = opt.Train.Checkpoint.checkpoint_dir
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    return logger


def calculate_iou(pred, target):
    """Calculate IoU for binary masks."""
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    pred_bin = (pred > 0.5).float()
    target_bin = (target > 0.5).float()

    intersection = (pred_bin * target_bin).sum(dim=(1, 2))
    union = (pred_bin + target_bin).clamp(0, 1).sum(dim=(1, 2))

    iou = torch.where(
        union == 0,
        torch.ones_like(intersection),
        intersection / (union + 1e-6)
    )

    return iou


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
            IOU += (IOUs.cpu().tolist())

    ret_IOU = np.mean(IOU)
    return ret_IOU


def train(opt, args):
    os.makedirs(opt.Train.Checkpoint.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'debug'), exist_ok=True)

    logger = set_logging(opt)
    logger.info(opt)

    # Load datasets
    train_dataset = eval(opt.Train.Dataset.type)(
        root=opt.Train.Dataset.root,
        transform_list=opt.Train.Dataset.transform_list,
        type='train'
    )
    logger.info('Number of train scenes: {}'.format(len(train_dataset)))

    val_dataset = eval(opt.Eval.Dataset.type)(
        root=opt.Eval.Dataset.root,
        transform_list=opt.Eval.Dataset.transform_list,
        type='val'
    )
    logger.info('Number of val scenes: {}'.format(len(val_dataset)))

    # Distributed setup
    if args.device_num > 1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.device_num)
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    # Data loaders
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.Train.Dataloader.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=opt.Train.Dataloader.num_workers,
        pin_memory=opt.Train.Dataloader.pin_memory,
        drop_last=True
    )

    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=opt.Train.Dataloader.batch_size,
        shuffle=False,
        num_workers=opt.Train.Dataloader.num_workers,
        drop_last=False
    )

    # Model initialization
    model = eval(opt.Model.name)(
        channels=opt.Model.channels,
        output_stride=opt.Model.output_stride,
        pretrained=opt.Model.pretrained,
        use_mccpb=getattr(opt.Model, 'use_mccpb', False),
        use_dual_heads=getattr(opt.Model, 'use_dual_heads', False),
        use_boundary_contrast=getattr(opt.Model, 'use_boundary_contrast', False),
        use_hf_gate=getattr(opt.Model, 'use_hf_gate', False),
        use_flip_consistency=getattr(opt.Model, 'use_flip_consistency', False),
        edge_dist_mode=getattr(opt.Model, 'edge_dist_mode', 'cdist'),
    )

    # Pass loss weights from config (kept for backward compatibility)
    if hasattr(opt.Model, 'bg_loss_weight'):
        model.bg_loss_weight = opt.Model.bg_loss_weight

    if args.local_rank <= 0:
        for name, param in model.named_parameters():
            logger.info("{} {}".format(name, param.shape))

    # Multi-GPU setup
    if args.device_num > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda()
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=True
        )
    else:
        model = model.cuda()

    # Optimizer setup. Two strategies:
    #   PVT path  (model name contains 'PolypPVT' or 'pvt'): unified LR, with
    #     weight-decay excluded from norms / biases. This matches the
    #     published Polyp-PVT AdamW recipe.
    #   ResNet path (BACFR / UACANet / PraNet / etc.): legacy split, backbone
    #     stages at base lr, decoder at 10x. Stem (resnet.conv1/bn1/maxpool)
    #     stays frozen — original BACFR design.
    _model_name = getattr(opt.Model, 'name', '')
    _is_pvt = ('PolypPVT' in _model_name) or ('pvt' in _model_name.lower())

    if _is_pvt:
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            n_lower = name.lower()
            if param.ndim <= 1 or name.endswith('.bias') or 'norm' in n_lower or 'bn' in n_lower:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        params_list = [
            {'params': decay_params,
             'weight_decay': opt.Train.Optimizer.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
    else:
        backbone_params = []
        decoder_params = []
        _stage_keys = ('layer', 'block', 'patch_embed', 'norm')
        for name, param in model.named_parameters():
            if 'resnet' in name or 'backbone' in name:
                if any(k in name for k in _stage_keys):
                    backbone_params.append(param)
            else:
                decoder_params.append(param)
        params_list = [
            {'params': backbone_params},
            {'params': decoder_params, 'lr': opt.Train.Optimizer.lr * 10},
        ]

    optimizer = eval(opt.Train.Optimizer.type)(
        params_list,
        opt.Train.Optimizer.lr,
        weight_decay=opt.Train.Optimizer.weight_decay
    )

    # Mixed precision setup
    if opt.Train.Optimizer.mixed_precision is True:
        scaler = GradScaler()
    else:
        scaler = None

    # Scheduler
    scheduler = eval(opt.Train.Scheduler.type)(
        optimizer,
        gamma=opt.Train.Scheduler.gamma,
        minimum_lr=opt.Train.Scheduler.minimum_lr,
        max_iteration=len(train_loader) * opt.Train.Scheduler.epoch,
        warmup_iteration=opt.Train.Scheduler.warmup_iteration
    )

    # Progress bars
    if args.local_rank <= 0 and args.verbose is True:
        epoch_iter = tqdm.tqdm(
            range(1, opt.Train.Scheduler.epoch + 1),
            desc='Epoch',
            total=opt.Train.Scheduler.epoch,
            position=0,
            bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}'
        )
    else:
        epoch_iter = range(1, opt.Train.Scheduler.epoch + 1)

    best = 0
    for epoch in epoch_iter:
        # Propagate epoch to model so lambda_aux / lambda_BC schedules advance.
        # Safe no-op if the model doesn't define set_epoch.
        target_model = model.module if args.device_num > 1 else model
        if hasattr(target_model, 'set_epoch'):
            target_model.set_epoch(epoch - 1)  # 0-indexed internally

        model.train()

        epoch_losses = {
            'total': [],
            'scale2': [],
            'scale3': [],
            'scale4': [],
        }

        # Set epoch for distributed sampler
        if args.device_num > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Progress bar for steps
        if args.local_rank <= 0 and args.verbose is True:
            step_iter = tqdm.tqdm(
                enumerate(train_loader, start=1),
                desc='Iter',
                total=len(train_loader),
                position=1,
                leave=False,
                bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}'
            )
        else:
            step_iter = enumerate(train_loader, start=1)

        # Training loop
        for i, sample in step_iter:
            optimizer.zero_grad()

            if opt.Train.Optimizer.mixed_precision is True:
                with autocast():
                    sample = to_cuda(sample)
                    out = model(sample)
                    loss = out['loss']

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sample = to_cuda(sample)
                out = model(sample)
                loss = out['loss']
                loss.backward()
                optimizer.step()

            scheduler.step()

            epoch_losses['total'].append(loss.item())

            if 'debug_loss' in out and len(out['debug_loss']) >= 3:
                epoch_losses['scale2'].append(out['debug_loss'][0])
                epoch_losses['scale3'].append(out['debug_loss'][1])
                epoch_losses['scale4'].append(out['debug_loss'][2])

            # Update progress bar
            if args.local_rank <= 0 and args.verbose is True:
                step_iter.set_postfix({
                    'loss': loss.item(),
                    'lr': optimizer.param_groups[0]['lr']
                })

        # Epoch logging
        mean_total_loss = np.mean(epoch_losses['total'])
        log_msg = f"Epoch {epoch}/{opt.Train.Scheduler.epoch} | Total Loss: {mean_total_loss:.4f}"

        if len(epoch_losses['scale2']) > 0:
            log_msg += f" | Scale2: {np.mean(epoch_losses['scale2']):.4f}"
            log_msg += f" | Scale3: {np.mean(epoch_losses['scale3']):.4f}"
            log_msg += f" | Scale4: {np.mean(epoch_losses['scale4']):.4f}"

        logger.info(log_msg)

        # Evaluation and checkpointing
        if args.local_rank <= 0:
            if epoch % opt.Train.Checkpoint.checkpoint_epoch == 0:
                logger.info('Starting evaluation...')

                cur_metric = Eval_model(model, val_loader)

                logger.info('=' * 80)
                logger.info(f'Epoch {epoch} | IoU: {cur_metric:.4f}')
                logger.info('=' * 80)

                # Save best model
                if cur_metric > best:
                    best = cur_metric
                    logger.info('=' * 80)
                    logger.info(f'NEW BEST IoU: {best:.4f}')
                    logger.info('=' * 80)

                    best_path = os.path.join(opt.Train.Checkpoint.checkpoint_dir, 'best.pth')
                    torch.save(
                        model.module.state_dict() if args.device_num > 1 else model.state_dict(),
                        best_path
                    )

                # Save checkpoint
                save_path = os.path.join(opt.Train.Checkpoint.checkpoint_dir, f'epoch_{epoch}.pth')
                torch.save(
                    model.module.state_dict() if args.device_num > 1 else model.state_dict(),
                    save_path
                )

                model.train()

            # Debug visualization
            if args.debug is True:
                debout = debug_tile(out)
                debug_path = os.path.join(
                    opt.Train.Checkpoint.checkpoint_dir,
                    'debug',
                    f'epoch_{epoch}.png'
                )
                cv2.imwrite(debug_path, debout)


if __name__ == '__main__':
    args = parse_args()
    # Honor --config from CLI; fall back to BACFR_Enhanced_v3_3.yaml when the
    # caller didn't pass one (parse_args' default points at a stale path).
    config = args.config
    if not os.path.isfile(config):
        config = 'configs/BACFR_Enhanced_v3_3.yaml'
    print(f'[Train_patch] using config: {config}')
    opt = load_config(config)
    train(opt, args)

# Single GPU
# CUDA_VISIBLE_DEVICES=1 python run/Train_patch.py --verbose --debug

# Multi-GPU (e.g., 2 GPUs)
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     run/Train_patch_v3.py --verbose --debug
