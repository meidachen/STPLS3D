import os
from sklearn.metrics import classification_report
import numpy as np
import sklearn
from sklearn.metrics import jaccard_score

from helper_ply import write_ply
from helper_ply import read_ply
import glob

def createFolder(folderPath):
    if not os.path.isdir(folderPath):
        os.makedirs(folderPath)

def compute_accuracyOneFile_txt(data):

    y_true_list = data[:,-1]
    y_pred_list = data[:,-2]


    classificationReport = classification_report(y_true_list, y_pred_list, digits=5)

    acc = sklearn.metrics.accuracy_score(y_true_list, y_pred_list)
    IOU = jaccard_score(y_true_list, y_pred_list, average=None)
    print (classificationReport)
    print ("accuracy,%f" %acc)
    print ("IOU: ",IOU)
    IOU = np.array(IOU)
    print ("mean IOU: %f" %np.average(IOU))
    return classificationReport, acc, IOU, np.average(IOU)


import sys

if __name__ == '__main__':

    prefictedPlyDir = r'E:\STPLS3D\SCF-Net\test\Log_2021-11-05_22-44-41\val_preds'
    prefictedPlyFilePaths = glob.glob(prefictedPlyDir + r'\*.ply')

    logFilePath = os.path.join(prefictedPlyDir, 'log.txt')
    sys.stdout = open(logFilePath,'w')

    allmiou = 0
    allAcc = 0
    allIou = np.array([0.0,0.0,0.0,0.0])
    count = 0
    for prefictedPlyFilePath in prefictedPlyFilePaths:
        if ('lbCorrected.ply' not in prefictedPlyFilePath):
            print (prefictedPlyFilePath)
            save_path = os.path.join(os.path.dirname(prefictedPlyFilePath),os.path.basename(prefictedPlyFilePath).replace('.ply','lbCorrected.ply'))

            if (os.path.exists(save_path)):
                data = read_ply(save_path, triangular_mesh=False)
                data = np.array(data.tolist())
            else:
                data = read_ply(prefictedPlyFilePath, triangular_mesh=False)
                data = np.array(data.tolist())

                data[:,6] = np.where((data[:,6] == 4), 1, data[:,6])
                data[:,6] = np.where((data[:,6] == 5), 1, data[:,6])
                write_ply(save_path, (data[:,:3], data[:,3:6], data[:,6], data[:,7]), ['x', 'y', 'z', 'red', 'green', 'blue', 'pre', 'class'])


            classificationReport, acc, IOU, mIOU = compute_accuracyOneFile_txt(data)
            allmiou += mIOU
            allAcc += acc
            allIou += IOU
            count += 1.0
    print ()
    print ("Average accuracy,%f" %(allAcc/count))
    print ("Average IOU: ",allIou/count)
    print ("Average mIOU: %f" %(allmiou/count))

