import torch
import os
import argparse
import tqdm
import sys

import torch.nn.functional as F
import numpy as np

from PIL import Image
from torch.nn import modules

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.utils import *
from utils.dataloader import *
# Explicit imports used in this script (helps linters/static analysis)
from utils.utils import parse_args, load_config, to_cuda
import torch.utils.data as data
from lib import *

def iou_score(pred_mask, gt_mask, threshold=0.5, eps=1e-7):
		"""Compute IoU between a predicted score map and a ground-truth mask.

		- pred_mask: numpy array (H,W) with float scores (assumed in [0,1] after sigmoid)
		- gt_mask: torch tensor or numpy array (H,W) with binary values (0/1) or 0-255
		- threshold: float threshold to binarize pred_mask
		Returns: IoU float
		"""
		# Convert GT to numpy
		
		gt_np = np.array(gt_mask)
		gt_bin = np.squeeze(gt_np)

		pred_bin = (pred_mask >= threshold).astype(np.uint8)

		inter = np.logical_and(pred_bin, gt_bin).sum()
		union = np.logical_or(pred_bin, gt_bin).sum()
		return float(inter) / (float(union) + eps)

def test(opt, args):
    #dataset
	val_dataset = eval(opt.Test.Dataset.type)(root=opt.Test.Dataset.root, transform_list=opt.Test.Dataset.transform_list,type = 'val')
	print(len(val_dataset))
	val_loader = data.DataLoader(dataset=val_dataset,
											batch_size=1,
											num_workers=opt.Test.Dataloader.num_workers,
											pin_memory=opt.Test.Dataloader.pin_memory)

    
    #model
	model = eval(opt.Model.name)(channels=opt.Model.channels,
								output_stride=opt.Model.output_stride,
								pretrained=opt.Model.pretrained)

	


	# iterate checkpoints and evaluate
	for i in range(10, 81, 10):
		ckpt_path = os.path.join(opt.Test.Checkpoint.checkpoint_dir, str(i) + '.pth')
		print(f"Evaluating checkpoint: {ckpt_path}")
		if not os.path.exists(ckpt_path):
			print(f"Checkpoint not found: {ckpt_path}, skipping")
			continue
		model.load_state_dict(torch.load(ckpt_path), strict=True)
		model.cuda()
		model.eval()

		metrics = []

		for sample in val_loader:
			gt = sample.get('gt', None).squeeze()
			
			sample['gt'] = None
			sample = to_cuda(sample)
			out = model(sample)
			pred = out['pred']
			pred = pred.data.cpu()
			pred = torch.sigmoid(pred)
			pred = pred.numpy().squeeze()
			
			metrics.append(iou_score(pred, gt))
		print(len(metrics))
		print(f"Checkpoint {ckpt_path} - Mean IoU: {np.mean(metrics):.4f}")
if __name__ == "__main__":
	args = parse_args()
	opt = load_config(args.config)
	test(opt, args)