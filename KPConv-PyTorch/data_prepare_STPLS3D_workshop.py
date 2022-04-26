from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
import os, glob, pickle
from utils.ply import write_ply,read_ply
from datasets.common import grid_subsampling


def createFolder(folderPath):
    if not os.path.isdir(folderPath):
        os.makedirs(folderPath)

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
    cloud[:, 6:7] = np.where((cloud[:, 6:7] == 20), 1, cloud[:, 6:7])
    cloud[:, 6:7] = np.where((cloud[:, 6:7] >17), 0, cloud[:, 6:7])


    return cloud

def prepareData(dataOriginal_fpath, save_path):


    if dataOriginal_fpath[-4:] == '.ply':
        cloud = read_ply(dataOriginal_fpath)
        cloud = np.vstack((cloud['x'], cloud['y'], cloud['z'],cloud['red'], cloud['green'], cloud['blue'],cloud['class'])).T
    if dataOriginal_fpath[-4:] == '.txt':
        cloud = pd.read_csv(dataOriginal_fpath, delimiter=',', header=[0]).values
    limitMin = np.amin(cloud[:, 0:3], axis=0)
    cloud[:, 0:3] -= limitMin
    cloud = changeSemLabels(cloud)
    values, counts = np.unique(cloud[:, 6], return_counts=True)
    print (values)
    print (counts)
    for i,value in enumerate(values):
        labelCount[int(value)] += counts[i]

    xyz = cloud[:, :3].astype(np.float32)
    colors = cloud[:, 3:6].astype(np.uint8)
    labels = cloud[:, 6].astype(np.uint8)
    write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    # save sub_cloud and KDTree file
    sub_xyz, sub_colors, sub_labels = grid_subsampling(xyz, colors, labels, sub_grid_size)
    sub_colors = sub_colors / 255.0
    sub_ply_file = os.path.join(sub_pc_dir, os.path.basename(save_path))

    write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    search_tree = KDTree(sub_xyz)

    kd_tree_file = os.path.join(sub_pc_dir, os.path.basename(save_path).replace('.ply','_KDTree.pkl'))

    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)

    proj_save =  os.path.join(sub_pc_dir, os.path.basename(save_path).replace('.ply','_proj.pkl'))

    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels], f)


if __name__ == '__main__':

    dir_list = ['RealWorldData',
                'Synthetic_v1',
                'Synthetic_v2',
                'Synthetic_v3']

    sub_grid_size = 0.5

    for data_folder in dir_list:

        print()
        print('    0-' + '-'*len(data_folder) + '-0')
        print('    |', data_folder, '|')
        print('    0-' + '-'*len(data_folder) + '-0')
        print()

        dataOriginal_dir = 'E:\ECCV_workshop\SemanticSegmentation/tmp/' + data_folder
        dataTraining_dir = 'E:\ECCV_workshop\SemanticSegmentation/tmp'
        createFolder(dataTraining_dir)
        original_ply_dir = os.path.join(dataTraining_dir,'original_ply')
        createFolder(original_ply_dir)
        sub_pc_dir = os.path.join(dataTraining_dir, 'input_{:.3f}'.format(sub_grid_size))
        createFolder(sub_pc_dir)
        labelCount = [0,0,0,0,0,0]
        dataOriginal_flist = glob.glob(dataOriginal_dir + '/*.txt')

        for dataOriginal_fpath in dataOriginal_flist:
            print()
            print(dataOriginal_fpath)
            save_path = os.path.join(original_ply_dir,os.path.basename(dataOriginal_fpath).replace('.txt','.ply'))
            prepareData(dataOriginal_fpath, save_path)

        print (labelCount)

        print()
        print('*'*30)
        print()

    # labelCount = [0,0,0,0,0,0]
    # dataOriginal_flist = glob.glob(dataOriginal_dir + r'\*.txt')
    # for dataOriginal_fpath in dataOriginal_flist:
    #     cloud = pd.read_csv(dataOriginal_fpath, delimiter=',', header=[0]).values
    #     cloud = changeSemLabels(cloud)
    #     values, counts = np.unique(cloud[:, 6], return_counts=True)
    #     print (values)
    #     print (counts)
    #     for i,value in enumerate(values):
    #         labelCount[int(value)] += counts[i]
    # print (labelCount)
