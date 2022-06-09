import numpy as np
from util.s3dis import S3DIS
from util.stpls import STPLS
import torch
from util.data_util import collate_fn
import gc
gc.collect()
dataset = STPLS('train', data_root = '/ssd/STPLS3D/block_50/', test_area = 0, voxel_size = 0.4)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_fn)
print(dataset)
min_samples = 0
for i, (coord, feat, target, offset) in enumerate(train_loader): 
	if i == 0:
		min_samples = coord.shape[0]
	else:
		if coord.shape[0] < min_samples:
			min_samples = coord.shape[0]
	# print(coord.shape)
print(min_samples)