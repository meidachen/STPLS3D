import os
import numpy as np
from util.ply import read_ply, write_ply

train_sequeces = ['Synthetic_v1', 'Synthetic_v2', 'Synthetic_v3', 'RealWorldData']
# cvalid_sequences = ['OCCC_points', 'RA_points', 'USC_points', 'WMSC_points']
def ply2array(ply_path):
	cloud = read_ply(ply_path)
	cloud = np.vstack((cloud['x'], cloud['y'], cloud['z'],cloud['red'], cloud['green'], cloud['blue'],cloud['class'])).T
	# cloud = changeSemLabels(cloud)
	return cloud

def splitPointCloud(cloud, num_w = 4, num_d = 4):
	limitMax = np.amax(cloud[:, 0:3], axis=0)
	limitMin = np.amin(cloud[:, 0:3], axis=0)
	# width = int(np.ceil((limitMax[0] - size) / stride)) + 1
	# depth = int(np.ceil((limitMax[1] - size) / stride)) + 1
	stride_x = (limitMax[0] - limitMin[0]) / num_w
	stride_y = (limitMax[1] - limitMin[1]) / num_d
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
	
# os.mkdir('/ssd/STPLS3D/block/')
# for i in os.listdir('/ssd/STPLS3D/Synthetic_v1/'):
# 	if not os.path.isdir('/ssd/STPLS3D/block/Synthetic_v1/'):
# 		os.mkdir('/ssd/STPLS3D/block/Synthetic_v1/')
# 	pc = ply2array('/ssd/STPLS3D/Synthetic_v1/' + i)
# 	# print(pc.shape)
# 	# input()
# 	blocks = splitPointCloud(pc)
# 	for c in range(len(blocks)):
# 		xyz = blocks[c][:, :3].astype(np.float32)
# 		print(xyz.shape)
# 		colors = blocks[c][:, 3:6].astype(np.uint8)
# 		labels = blocks[c][:, 6].astype(np.uint8)
# 		save_path = '/ssd/STPLS3D/block/Synthetic_v1/' + i.split('.')[0] + '_' + str(c) + '.ply'
# 		write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

# for i in os.listdir('/ssd/STPLS3D/Synthetic_v2/'):
# 	if not os.path.isdir('/ssd/STPLS3D/block/Synthetic_v2/'):
# 		os.mkdir('/ssd/STPLS3D/block/Synthetic_v2/')
# 	pc = ply2array('/ssd/STPLS3D/Synthetic_v2/' + i)
# 	# print(pc.shape)
# 	# input()
# 	blocks = splitPointCloud(pc)
# 	for c in range(len(blocks)):
# 		xyz = blocks[c][:, :3].astype(np.float32)
# 		print(xyz.shape)
# 		colors = blocks[c][:, 3:6].astype(np.uint8)
# 		labels = blocks[c][:, 6].astype(np.uint8)
# 		save_path = '/ssd/STPLS3D/block/Synthetic_v2/' + i.split('.')[0] + '_' + str(c) + '.ply'
# 		write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

for i in os.listdir('/ssd/STPLS3D/Synthetic_v3/'):
	if not os.path.isdir('/ssd/STPLS3D/block/Synthetic_v3/'):
		os.mkdir('/ssd/STPLS3D/block/Synthetic_v3/')
	pc = ply2array('/ssd/STPLS3D/Synthetic_v3/' + i)
	# print(pc.shape)
	# input()
	blocks = splitPointCloud(pc)
	for c in range(len(blocks)):
		xyz = blocks[c][:, :3].astype(np.float32)
		print(xyz.shape)
		colors = blocks[c][:, 3:6].astype(np.uint8)
		labels = blocks[c][:, 6].astype(np.uint8)
		save_path = '/ssd/STPLS3D/block/Synthetic_v3/' + i.split('.')[0] + '_' + str(c) + '.ply'
		write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

# for i in os.listdir('/ssd/STPLS3D/RealWorldData/'):
# 	if not os.path.isdir('/ssd/STPLS3D/block/RealWorldData/'):
# 		os.mkdir('/ssd/STPLS3D/block/RealWorldData/')
# 	pc = ply2array('/ssd/STPLS3D/RealWorldData/' + i)
# 	# print(pc.shape)
# 	# input()
# 	blocks = splitPointCloud(pc)
# 	for c in range(len(blocks)):
# 		xyz = blocks[c][:, :3].astype(np.float32)
# 		print(xyz.shape)
# 		colors = blocks[c][:, 3:6].astype(np.uint8)
# 		labels = blocks[c][:, 6].astype(np.uint8)
# 		save_path = '/ssd/STPLS3D/block/RealWorldData/' + i.split('.')[0] + '_' + str(c) + '.ply'
# 		write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])