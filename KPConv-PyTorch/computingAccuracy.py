import argparse
import os
from sklearn.metrics import classification_report
import numpy as np
import sklearn
from sklearn.metrics import jaccard_score
import subprocess
import json
import os
import pandas as pd
from utils import ply

'''
Sample Example -
python computingAccuracy.py --GT_directory ./ --pdal_file_directory ./ --GT_name "MUTC_03_GT_leveled.txt" --pdal_file_name some.txt

Output -

GT File Directory: ./
PDAL pdal_file Directory: ./
Name of GT file: MUTC_03_GT_leveled.txt
Name of PDAL file: some.txt

sklearn computed recall precision
'''
def createFolder(folderPath):
    if not os.path.isdir(folderPath):
        os.makedirs(folderPath)


def KNNLabelTransfer(src, pred, dest, k=8, PDAL_PATH=r'C:\Users\mechen\Desktop\STPLS+1.2\STPLSPlusCore\External\PDAL'):
    print ("KNNLabelTransfer")
    pipeline = {
                "pipeline":[
                    src,
                    {
                    "type" : "filters.neighborclassifier",
                    "domain" : "Classification[1:1]",
                    "k" : k,
                    "candidate" : pred
                    },
                    dest
                ]
    }
    p = subprocess.run([PDAL_PATH,'pipeline','-s'],input=json.dumps(pipeline).encode())


def createPCwithLabelONE(PCPath, PCWithLabelONEPath):
    PC = open(PCPath, "r")
    PCwithLabelONE = open(PCWithLabelONEPath, "w")
    PCwithLabelONE.write("X,Y,Z,Red,Green,Blue,Classification\n")
    for line in PC.readlines():
        point = line.rstrip('\n').split(',')
        PCwithLabelONE.write('%s,%s,%s,%s,%s,%s,1\n' % (point[0], point[1], point[2], point[3], point[4], point[5]))
    PCwithLabelONE.close()
    PC.close()


def compute_accuracyTwoFiles(GT_file, predictPDAL_file):
    LBDict = {}
    with open(predictPDAL_file , 'r') as f:
        for line in f.readlines()[1:]:
            point = line.rstrip('\n').split(',')
            for i in range(0, len(point)):
                point[i] = round(float(point[i]),3)
            LBDict[str(point[0])+str(point[1])+str(point[2])] = [int(point[-1])]

    with open(GT_file, 'r') as f:
        for line in f.readlines()[1:]:
            point = line.rstrip('\n').split(',')
            for i in range(0, len(point)):
                point[i] = round(float(point[i]),3)
            key = str(point[0])+str(point[1])+str(point[2])
            if (key in LBDict):
                LBDict[str(point[0])+str(point[1])+str(point[2])].append(int(point[-1]))

    for k,v in list(LBDict.items()):
        if (len(v) != 2):
            del LBDict[k]

    LBList = np.array(list(LBDict.values()))
    LBList = np.transpose(LBList)
    y_true_list = LBList[1,:]
    y_pred_list = LBList[0,:]
    # classificationReport = classification_report(y_true_list, y_pred_list, digits=5, output_dict=True)
    classificationReport = classification_report(y_true_list, y_pred_list, digits=5)

    acc = sklearn.metrics.accuracy_score(y_true_list, y_pred_list)
    IOU = jaccard_score(y_true_list, y_pred_list, average=None)
    print (classificationReport)
    print ("accuracy,%f" %acc)
    print ("IOU: ",IOU)
    IOU = np.array(IOU)
    print ("mean IOU: %f" %np.average(IOU))
    return classificationReport, acc


def compute_accuracyOneFile_txt(GT_file):
    if (GT_file.split('.')[-1] == 'txt'):
        pc = pd.read_csv(GT_file, delimiter=',', header=[0]).values
        y_true_list = pc[:,-1]
        y_pred_list = pc[:,-2]
    elif (GT_file.split('.')[-1] == 'ply'):
        data = ply.read_ply(filename, triangular_mesh=False)
        data = np.array(data.tolist())
        y_true_list = data[:,-1]
        y_pred_list = data[:,-2]
    else:
        print ("Can not process file type: %s" % GT_file.split('.')[-1])
        return

    classificationReport = classification_report(y_true_list, y_pred_list, digits=5)

    acc = sklearn.metrics.accuracy_score(y_true_list, y_pred_list)
    IOU = jaccard_score(y_true_list, y_pred_list, average=None)
    print (classificationReport)
    print ("accuracy,%f" %acc)
    print ("IOU: ",IOU)
    IOU = np.array(IOU)
    print ("mean IOU: %f" %np.average(IOU))
    return classificationReport, acc, IOU, np.average(IOU)


if __name__ == '__main__':
    # GT_pointCloud =r'D:\syntheticVSreal_experiements\KPConv-PyTorch\test\USC_GT_300.txt'
    # result_pointCloud = r'D:\syntheticVSreal_experiements\KPConv-PyTorch\test\USC_GT_120.txt'

    # outPutDir = r'C:\Users\mechen\Desktop\test\outAcc'
    # # outPutAcc = os.path.join(outPutDir,r'acc.csv')
    # createFolder(outPutDir)
    # PCWithLabelONEPath = os.path.join(outPutDir,'cloudONE.txt')
    # createPCwithLabelONE(GT_pointCloud, PCWithLabelONEPath)
    # outPutPCPath = os.path.join(outPutDir, 'PointNetResult.txt')
    #
    # KNNLabelTransfer(PCWithLabelONEPath, result_pointCloud, outPutPCPath)



    # filename = r'E:\ResidentialArea_GT.ply'
    # classificationReport, acc, IOU, mIOU = compute_accuracyOneFile_txt(filename)




    import glob
    predDir = r'D:\syntheticVSreal_experiements\KPConv-PyTorch\results\Log_2021-09-27_05-24-43'
    filenames = glob.glob(predDir + r'\*\*.ply')
    resultDir = {'mIOUs':[],'IOUs':[],'accs':[],'filePaths':[]}


    for filename in filenames:
        if ('val_preds_' in filename ):
            print (filename)
            classificationReport, acc, IOU, mIOU = compute_accuracyOneFile_txt(filename)
            epoch = os.path.basename(os.path.dirname(filename))
            newName = os.path.basename(filename).split('.')[0] + '_' + str(epoch) + '___mIOU-%f'%mIOU + "---"
            for item in IOU:
                newName += str(item) + ','
            newName += '--acc-%f'%acc +'.ply'
            newName = os.path.join(os.path.dirname(filename),newName)

            resultDir['mIOUs'].append(mIOU)
            resultDir['accs'].append(acc)
            resultDir['IOUs'].append(IOU)
            resultDir['filePaths'].append(filename)

    print (resultDir)

    mIOUs = [x for _, x in sorted(zip(resultDir['mIOUs'], resultDir['mIOUs']))]
    print ("mIOUs: ")
    print (mIOUs)
    accs = [x for _, x in sorted(zip(resultDir['mIOUs'], resultDir['accs']))]
    print ("accs: ")
    print (accs)
    IOUs = [x for _, x in sorted(zip(resultDir['mIOUs'], resultDir['IOUs']))]
    print ("IOUs: ")
    print (IOUs)
    filePaths = [x for _, x in sorted(zip(resultDir['mIOUs'], resultDir['filePaths']))]
    print ("filePaths: ")
    print (filePaths)
