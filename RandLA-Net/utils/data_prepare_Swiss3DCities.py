from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
import os, glob, pickle
from helper_ply import write_ply
from helper_ply import read_ply
from helper_tool import DataProcessing as DP



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
    cloud[:, 6:7] = np.where((cloud[:, 6:7] >17), 0, cloud[:, 6:7])


    return cloud

def convert_pc2ply(cloud, save_path):


    limitMin = np.amin(cloud[:, 0:3], axis=0)
    cloud[:, 0:3] -= limitMin
    cloud = changeSemLabels(cloud)


    xyz = cloud[:, :3].astype(np.float32)
    colors = cloud[:, 3:6].astype(np.uint8)
    labels = cloud[:, 6].astype(np.uint8)
    write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])


    # save sub_cloud and KDTree file
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
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
    dataOriginal_dir = r'E:\STPLS3D\Swiss3DCities'
    dataTraining_dir = r'E:\STPLS3D\Swiss3DCities_prepared'
    createFolder(dataTraining_dir)

    sub_grid_size = 0.3
    sub_pc_dir = os.path.join(dataTraining_dir, 'input_{:.3f}'.format(sub_grid_size))
    createFolder(sub_pc_dir)
    original_ply_dir = os.path.join(dataTraining_dir,'original_ply')
    createFolder(original_ply_dir)

    subFolders = [x[0] for x in os.walk(dataOriginal_dir)][1:]
    for eachDataDir in subFolders:
        originalPlyFilePaths = glob.glob(eachDataDir + r'\*.ply')
        outFileName = os.path.basename(eachDataDir)
        originalData = np.empty([1, 7])
        for originalPlyFilePath in originalPlyFilePaths:
            print (originalPlyFilePath)

            data = read_ply(originalPlyFilePath, triangular_mesh=False)
            data = np.array(data.tolist())
            if (len(data[0])>6):
                print ("This data has normal")
                data = data[:,:6]
            if ('1_terrain' in originalPlyFilePath):
                lb = 0
            elif ('2_construction' in originalPlyFilePath):
                lb = 1
            elif ('3_urbanasset' in originalPlyFilePath):
                lb = 1
            elif ('4_vegetation' in originalPlyFilePath):
                lb = 2
            elif ('5_vehicle' in originalPlyFilePath):
                lb = 5

            lbArray = np.full((len(data),1),lb)
            data = np.concatenate((data, lbArray), axis=1)
            originalData = np.concatenate((originalData, data), axis=0)
        originalData = np.delete(originalData, (0), axis=0)

        save_path = os.path.join(original_ply_dir,'%s.ply'%outFileName)
        convert_pc2ply(originalData, save_path)
