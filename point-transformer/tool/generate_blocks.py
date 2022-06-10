import os
import numpy as np
from util.ply import read_ply, write_ply
import argparse

parser = argparse.ArgumentParser(description = "Generate sub blocks from a large scale point cloud")
parser.add_argument('--data_path', '-d', dest = 'data_path', type = str, required = True)
args = parser.parse_args()

train_sequeces = ['Synthetic_v1', 'Synthetic_v2', 'Synthetic_v3', 'RealWorldData']
def ply2array(ply_path):
	cloud = read_ply(ply_path)
	cloud = np.vstack((cloud['x'], cloud['y'], cloud['z'],cloud['red'], cloud['green'], cloud['blue'],cloud['class'])).T
	return cloud

def splitPointCloud(cloud, size = 100, num_w = 0, num_d = 0):
	limitMax = np.amax(cloud[:, 0:3], axis=0)
	limitMin = np.amin(cloud[:, 0:3], axis=0)

	if num_w != 0 and num_d != 0:
		stride_x = (limitMax[0] - limitMin[0]) / num_w
		stride_y = (limitMax[1] - limitMin[1]) / num_d
		cells = [(limitMin[0] + x * stride_x, limitMin[1] + y * stride_y) for x in range(num_w) for y in range(num_d)]
	else:
		num_w = int(np.ceil((limitMax[0] - limitMin[0]) / size))
		num_d = int(np.ceil((limitMax[1] - limitMin[1]) / size))
		stride_x = size
		stride_y = size 
		cells = [(limitMin[0] + x * stride_x, limitMin[1] + y * stride_y) for x in range(num_w) for y in range(num_d)]
	blocks = []
	for (x, y) in cells:
		xcond = (cloud[:, 0] <= x + stride_x) & (cloud[:, 0] >= x)
		ycond = (cloud[:, 1] <= y + stride_y) & (cloud[:, 1] >= y)
		cond  = xcond & ycond
		block = cloud[cond, :]
		if block.shape[0] != 0:
			blocks.append(block)
	return blocks

os.mkdir(args.data_path + 'block/')

for sq in train_sequeces:
	for i in os.listdir(args.data_path + sq):
		if not os.path.isdir(args.data_path + 'block/' + sq):
			os.mkdir(args.data_path + 'block/' + sq)
		pc = ply2array(args.data_path + sq + '/' + i)
		blocks = splitPointCloud(pc, num_w = 4, num_d = 4)
		for c in range(len(blocks)):
			xyz = blocks[c][:, :3].astype(np.float32)
			colors = blocks[c][:, 3:6].astype(np.uint8)
			labels = blocks[c][:, 6].astype(np.uint8)
			save_path = args.data_path + 'block/' + sq + '/' + i.split('.')[0] + '_' + str(c) + '.ply'
			write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
