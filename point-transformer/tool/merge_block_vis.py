import numpy as np
import os
from util.ply import read_ply, write_ply
import argparse

color_map = {0: [153, 151, 148], 1: [242, 207, 10], 2: [10, 242, 21], 3:[242, 10, 21], 4:[10, 41, 242], 5: [242, 150, 205]}

parser = argparse.ArgumentParser(description = "Merge blocks to form the large scale original point cloud with label and corresponding label color")
parser.add_argument('--data_path', '-d', dest = 'data_path', type = str, required = True)
parser.add_argument('--predictions', '-p', dest = 'predictions', type = str, required = True)
parser.add_argument('--save_path', '-s', dest = 'save_path', type = str, required = True)
args = parser.parse_args()

def ply2array(ply_path):
	cloud = read_ply(ply_path)
	cloud = np.vstack((cloud['x'], cloud['y'], cloud['z'],cloud['red'], cloud['green'], cloud['blue'],cloud['class'])).T
	return cloud

def main():
	pred_path = args.predictions
	pc_path = args.data_path
	if not os.path.isdir(args.save_path):
		os.mkdir(args.save_path)
	pred_list = os.listdir(pred_path)
	pc_list = os.listdir(pc_path)

	pred_list = [item for item in pred_list if 'pred' in item]
	
	pc_list = os.listdir(pc_path)
	pc_list = sorted(pc_list)
	pred_list = sorted(pred_list)

	maxkey = 0
	for key, data in color_map.items():
		if key > maxkey:
			maxkey = key
	remap_lut = np.zeros((maxkey + 100, 3), dtype=np.int32)
	for key, data in color_map.items():
		try:
			remap_lut[key] = data
		except IndexError:
			print("Wrong key ", key)

	for i in range(len(pc_list)):
		pc = ply2array(pc_path + pc_list[i])

		pred = np.load(pred_path + pred_list[i])
		pred_color = remap_lut[pred]
		save_pc = np.concatenate([pred_color, pred.reshape((-1, 1))], axis = 1)
		save_pc = np.concatenate([pc[:, :3], save_pc], axis = 1)
		if i == 0:
			whole_pc = save_pc
		else:
			whole_pc = np.concatenate([whole_pc, save_pc], axis = 0)
	field_names = ['x', 'y', 'z', 'red', 'green', 'blue', 'pred_label']
	write_ply(args.save_path + pc_list[i][:-4] + '_all.ply', [whole_pc[:, :3], whole_pc[:, 3:6].astype(np.uint8), whole_pc[:, -1].astype(np.int32)], field_names)

if __name__ == "__main__":
	main()