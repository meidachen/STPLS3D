import numpy as np
import glob
import pandas as pd
import json
import os

coordShiftPath = r'test/coordShift.json'
coordShift = json.load(open(coordShiftPath))
coordDir = r'test/coords_offsets'
semanticDir = r'test/semantic'
insMaskDir = r'test/predicted_masks'
outPath = r'test/out.txt'
outFile = open(outPath,'w')

coordFilePaths = glob.glob(coordDir + '/*.npy')
insLabel = 1
for coordFilePath in coordFilePaths:
    fileName = os.path.basename(coordFilePath).strip('.npy')
    print (fileName)
    xyz = np.load(coordFilePath)
    semantic = np.load(os.path.join(semanticDir,fileName+'.npy'))
    insMaskFilePathList = sorted(glob.glob(insMaskDir + '/%s*.txt' %fileName))

    ins = np.zeros(len(xyz))
    for insMaskPath in insMaskFilePathList:
        insMask = pd.read_csv(insMaskPath, delimiter=',', header=None).values
        insMask = np.squeeze(insMask, axis=1)
        ins[insMask==1] = insLabel
        insLabel+=1

    xyz[:,:3] += np.array([float(value) for value in coordShift[fileName]])
    xyz[:,:3] += np.array([float(value) for value in coordShift['globalShift']])
    for i in range(len(xyz)):
        outFile.write("%f,%f,%f,%d,%d\n" %(xyz[i][0],xyz[i][1],xyz[i][2],semantic[i],ins[i]))

outFile.close()

    # for i in range(len(xyz)):
    #     outFile.write("%f,%f,%f,%d,%d,%d,%d\n" %(xyz[i][0],xyz[i][1],xyz[i][2],xyz[i][3],xyz[i][4],xyz[i][5],semantic[i]))

# outFile.close()


