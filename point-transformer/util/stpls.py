import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset
from util.data_util import sa_create
from util.data_util import data_prepare
from util.ply import write_ply,read_ply
import os

train_sequeces = ['Synthetic_v1', 'Synthetic_v2', 'Synthetic_v3', 'RealWorldData']
cvalid_sequences = ['OCCC_points', 'RA_points', 'USC_points', 'WMSC_points']

def changeSemLabels(cloud):

	cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 2) &  (cloud[:, 6:7] <= 4), 2, cloud[:, 6:7])
	cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 5) &  (cloud[:, 6:7] <= 6), 3, cloud[:, 6:7])
	cloud[:, 6:7] = np.where((cloud[:, 6:7] == 8), 3, cloud[:, 6:7])
	cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 11) &  (cloud[:, 6:7] <= 12), 4, cloud[:, 6:7])
	cloud[:, 6:7] = np.where((cloud[:, 6:7] == 14), 5, cloud[:, 6:7])

	cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 7) &  (cloud[:, 6:7] <= 10), 1, cloud[:, 6:7])
	cloud[:, 6:7] = np.where((cloud[:, 6:7] == 13), 1, cloud[:, 6:7])
	cloud[:, 6:7] = np.where((cloud[:, 6:7] >= 15) &  (cloud[:, 6:7] <= 16), 0, cloud[:, 6:7])
	cloud[:, 6:7] = np.where((cloud[:, 6:7] == 17), 1, cloud[:, 6:7])
	cloud[:, 6:7] = np.where((cloud[:, 6:7] >17), 0, cloud[:, 6:7])
	return cloud

def ply2array(ply_path):
	cloud = read_ply(ply_path)
	cloud = np.vstack((cloud['x'], cloud['y'], cloud['z'],cloud['red'], cloud['green'], cloud['blue'],cloud['class'])).T
	cloud = changeSemLabels(cloud)
	return cloud

class STPLS(Dataset):
	def __init__(self, split='train', data_root='trainval', test_area=0, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
		super().__init__()
		self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
		data_list = []
		for sq in train_sequeces:
			data_list += os.listdir(data_root + sq)
		if split == 'train':
			self.data_list = [item for item in data_list if not '{}_'.format(cvalid_sequences[test_area]) in item]
		else:
			self.data_list = [item for item in data_list if '{}_'.format(cvalid_sequences[test_area]) in item]
		self.data_list = [item[:-4] for item in self.data_list]
		self.data_list = sorted(self.data_list)
		# for item in self.data_list:
		# 	if not os.path.exists("/dev/shm/{}".format(item)):
		# 		if "OCCC_points" in item or "RA_points" in item or "USC_points" in item or "WMSC_points" in item:
		# 			data_path = os.path.join(data_root, 'RealWorldData',item + '.ply')
		# 		elif "v2" in item:
		# 			data_path = os.path.join(data_root, 'Synthetic_v2',item + '.ply')
		# 		elif "v3" in item:
		# 			data_path = os.path.join(data_root, 'Synthetic_v3',item + '.ply')
		# 		else:
		# 			data_path = os.path.join(data_root, 'Synthetic_v1',item + '.ply')
		# 		data =ply2array(data_path)  # xyzrgbl, N*7
		# 		sa_create("shm://{}".format(item), data)
		self.data_root = data_root
		self.data_idx = np.arange(len(self.data_list))
		print("Totally {} samples in {} set.".format(len(self.data_idx), split))
	def __getitem__(self, idx):
		data_idx = self.data_idx[idx % len(self.data_idx)]
		# data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
		data_path = self.data_list[data_idx]
		if "OCCC_points" in data_path or "RA_points" in data_path or "USC_points" in data_path or "WMSC_points" in data_path:
			data_path = os.path.join(self.data_root, 'RealWorldData',data_path + '.ply')
		elif "v2" in data_path:
			data_path = os.path.join(self.data_root, 'Synthetic_v2',data_path + '.ply')
		elif "v3" in data_path:
			data_path = os.path.join(self.data_root, 'Synthetic_v3',data_path + '.ply')
		else:
			data_path = os.path.join(self.data_root, 'Synthetic_v1',data_path + '.ply') 

		data = ply2array(data_path)
		coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
		coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
		
		return coord, feat, label
	def __len__(self):
		return len(self.data_idx) * self.loop