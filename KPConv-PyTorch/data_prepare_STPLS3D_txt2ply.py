from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
import os, glob, pickle
from utils.ply import write_ply
from datasets.common import grid_subsampling


def createFolder(folderPath):
    if not os.path.isdir(folderPath):
        os.makedirs(folderPath)

def convert_txt2ply(dataOriginal_fpath, save_path):

    cloud = pd.read_csv(dataOriginal_fpath, delimiter=',', header=[0]).values
    instance = True if len(cloud[0])>7 else False

    xyz = cloud[:, :3].astype(np.float32)
    colors = cloud[:, 3:6].astype(np.uint8)
    semLabels = cloud[:, 6].astype(np.uint8)
    if instance:
        insLabels = cloud[:, 7].astype(np.uint8)
        write_ply(save_path, (xyz, colors, semLabels, insLabels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'instance'])
    else:
        write_ply(save_path, (xyz, colors, semLabels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])


if __name__ == '__main__':

    dir_list = ['RealWorldData',
                'Synthetic_v1',
                'Synthetic_v2',
                'Synthetic_v3']

    for data_folder in dir_list:

        print()
        print('    0-' + '-'*len(data_folder) + '-0')
        print('    |', data_folder, '|')
        print('    0-' + '-'*len(data_folder) + '-0')
        print()

        dataOriginal_dir = 'E:/ECCV_workshop/SemanticSegmentation/STPLS3D_raw_txt/' + data_folder
        dataTraining_dir = 'E:/ECCV_workshop/SemanticSegmentation/STPLS3D_raw_ply/' + data_folder
        createFolder(dataTraining_dir)
        dataOriginal_flist = glob.glob(dataOriginal_dir + '/*.txt')

        for dataOriginal_fpath in dataOriginal_flist:
            print()
            print(dataOriginal_fpath)
            save_path = os.path.join(dataTraining_dir,os.path.basename(dataOriginal_fpath).replace('.txt','.ply'))
            convert_txt2ply(dataOriginal_fpath, save_path)

        print()
        print('*'*30)
        print()
