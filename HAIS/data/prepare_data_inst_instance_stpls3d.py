import glob, numpy as np, torch
import pandas as pd
import os
import json
import random
import math

def splitPointCloud(cloud, size=50.0, stride=50):
    limitMax = np.amax(cloud[:, 0:3], axis=0)
    width = int(np.ceil((limitMax[0] - size) / stride)) + 1
    depth = int(np.ceil((limitMax[1] - size) / stride)) + 1
    cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]
    blocks = []
    for (x, y) in cells:
        xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
        ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
        cond  = xcond & ycond
        block = cloud[cond, :]
        blocks.append(block)
    return blocks

def getFiles(files,fileSplit):
    res = []
    for filePath in files:
        name = os.path.basename(filePath)
        num = name[:2] if name[:2].isdigit() else name[:1]
        if int(num) in fileSplit:
            res.append(filePath)
    return res

def dataAug(file,semanticKeep):
    points = pd.read_csv(file, header = None).values
    angle = random.randint(1, 359)
    angleRadians = math.radians(angle)
    rotationMatrix = np.array([[math.cos(angleRadians), -math.sin(angleRadians),0],[math.sin(angleRadians),math.cos(angleRadians), 0],[0,0,1]])
    points[:,:3] = points[:,:3].dot(rotationMatrix)
    pointsKept = points[np.in1d(points[:,6], semanticKeep)]
    return pointsKept

def preparePthFiles(files, split, outPutFolder, AugTimes=0):
    ### save the coordinates so that we can merge the data to a single scene after segmentation for visualization
    outJsonPath = os.path.join(outPutFolder, 'coordShift.json')
    coordShift = {}
    ### used to increase z range if it is smaller than this, over come the issue where spconv may crash for voxlization.
    zThreshold = 6

    # Map relevant classes to {1,...,14}, and ignored classes to -100
    remapper = np.ones(150) * (-100)
    for i, x in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
        remapper[x] = i
    # Map instance to -100 based on selected semantic (change a semantic to -100 if you want to ignore it for instance)
    remapper_disableInstanceBySemantic = np.ones(150) * (-100)
    for i, x in enumerate([-100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
        remapper_disableInstanceBySemantic[x] = i

    ### only augment data for these classes
    semanticKeep = [0, 2, 3, 7, 8, 9, 12, 13]

    counter = 0
    for file in files:

        for AugTime in range(AugTimes+1):
            if AugTime == 0:
                points = pd.read_csv(file, header=None).values
            else:
                points = dataAug(file,semanticKeep)
            name = os.path.basename(file).strip('.txt')+'_%d'%AugTime

            if split != 'test':
                coordShift['globalShift'] = list(points[:, :3].min(0))
            points[:, :3] = points[:, :3] - points[:, :3].min(0)

            blocks = splitPointCloud(points, size=50, stride=50)
            for blockNum, block in enumerate(blocks):
                if (len(block) > 10000):
                    outFilePath = os.path.join(outPutFolder, name + str(blockNum) + '_inst_nostuff.pth')
                    if (block[:, 2].max(0) - block[:, 2].min(0) < zThreshold):
                        block = np.append(block, [[block[:, 0].mean(0), block[:, 1].mean(0),
                                                   block[:, 2].max(0) + (
                                                               zThreshold - (block[:, 2].max(0) - block[:, 2].min(0))),
                                                   block[:, 3].mean(0), block[:, 4].mean(0), block[:, 5].mean(0),
                                                   -100, -100]], axis=0)
                        print("range z is smaller than threshold ")
                        print(name + str(blockNum) + '_inst_nostuff')
                    if split != 'test':
                        outFileName = name + str(blockNum) + '_inst_nostuff'
                        coordShift[outFileName] = list(block[:, :3].mean(0))
                    coords = np.ascontiguousarray(block[:, :3] - block[:, :3].mean(0))

                    # coords = block[:, :3]
                    colors = np.ascontiguousarray(block[:, 3:6]) / 127.5 - 1

                    coords = np.float32(coords)
                    colors = np.float32(colors)
                    if split != 'test':
                        sem_labels = np.ascontiguousarray(block[:, -2])
                        sem_labels = sem_labels.astype(np.int32)
                        sem_labels = remapper[np.array(sem_labels)]

                        instance_labels = np.ascontiguousarray(block[:, -1])
                        instance_labels = instance_labels.astype(np.float32)

                        disableInstanceBySemantic_labels = np.ascontiguousarray(block[:, -2])
                        disableInstanceBySemantic_labels = disableInstanceBySemantic_labels.astype(np.int32)
                        disableInstanceBySemantic_labels = remapper_disableInstanceBySemantic[
                            np.array(disableInstanceBySemantic_labels)]
                        instance_labels = np.where(disableInstanceBySemantic_labels == -100, -100, instance_labels)

                        # map instance from 0.
                        # [1:] because there are -100
                        uniqueInstances = (np.unique(instance_labels))[1:].astype(np.int32)
                        remapper_instance = np.ones(50000) * (-100)
                        for i, j in enumerate(uniqueInstances):
                            remapper_instance[j] = i

                        instance_labels = remapper_instance[instance_labels.astype(np.int32)]

                        uniqueSemantics = (np.unique(sem_labels))[1:].astype(np.int32)

                        if split == 'train' and (
                                len(uniqueInstances) < 10 or (len(uniqueSemantics) >= (len(uniqueInstances) - 2))):
                            print("unique insance: %d" % len(uniqueInstances))
                            print("unique semantic: %d" % len(uniqueSemantics))
                            print()
                            counter += 1
                        else:
                            torch.save((coords, colors, sem_labels, instance_labels), outFilePath)
                            # outFilePthPath = outFilePath[:-4]+'.npy'
                            # data = np.concatenate((coords, colors, np.expand_dims(sem_labels, axis=1), np.expand_dims(instance_labels, axis=1)), axis=1)
                            # np.save(outFilePthPath,data)
                            ### save text file for each pth file
                            # outFilePath = os.path.join(outPutFolder,name+str(blockNum)+'.txt')
                            # outFile = open(outFilePath,'w')
                            # for i in range(len(coords)):
                            #     outFile.write("%f,%f,%f,%f,%f,%f,%d,%d\n" %(coords[i][0],coords[i][1],coords[i][2],
                            #                                                 colors[i][0],colors[i][1],colors[i][2],
                            #                                                 sem_labels[i],instance_labels[i]))
                    else:
                        torch.save((coords, colors), outFilePath)
                        # outFilePthPath = outFilePath[:-4]+'.npy'
                        # data = np.concatenate((coords, colors), axis=1)
                        # np.save(outFilePthPath,data)
                        # save text file for each pth file
                        # outFileTxtPath = outFilePath[:-4]+'.txt'
                        # outFile = open(outFileTxtPath,'w')
                        # for i in range(len(coords)):
                        #     outFile.write("%f,%f,%f,%f,%f,%f\n" %(coords[i][0],coords[i][1],coords[i][2],
                        #                                                 colors[i][0],colors[i][1],colors[i][2]))
    print("Total skipped file :%d" % counter)
    json.dump(coordShift, open(outJsonPath, 'w'))

def prepareInstGt(valOutDir, val_gtFolder,semantic_label_idxs):
    valFilesPth = sorted(glob.glob('{}/*_inst_nostuff.pth'.format(valOutDir)))
    blocks = [torch.load(i) for i in valFilesPth]

    for i in range(len(blocks)):
        xyz, rgb, label, instance_label = blocks[i]  # label 0~19 -100;  instance_label 0~instance_num-1 -100
        scene_name = os.path.basename(valFilesPth[i]).strip('.pth')
        print('{}/{} {}'.format(i + 1, len(blocks), scene_name))

        instance_label_new = np.zeros(instance_label.shape,
                                      dtype=np.int32)  # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)

        instance_num = int(instance_label.max()) + 1
        for inst_id in range(instance_num):
            instance_mask = np.where(instance_label == inst_id)[0]
            sem_id = int(label[instance_mask[0]])
            if (sem_id == -100): sem_id = 0
            semantic_label = semantic_label_idxs[sem_id]
            instance_label_new[instance_mask] = semantic_label * 1000 + inst_id + 1

        np.savetxt(os.path.join(val_gtFolder, scene_name + '.txt'), instance_label_new, fmt='%d')

if __name__ == '__main__':

    data_folder = os.path.join(os.path.dirname(os.getcwd()),'dataset/Synthetic_v3_InstanceSegmentation')
    filesOri = sorted(glob.glob(data_folder + '/*.txt'))

    trainSplit = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
    trainFiles = getFiles(filesOri,trainSplit)
    split = 'train'
    trainOutDir = os.path.join(data_folder,split)
    os.makedirs(trainOutDir,exist_ok=True)
    preparePthFiles(trainFiles, split, trainOutDir, AugTimes=6)

    valSplit = [5, 10, 15, 20, 25]
    split = 'val'
    valFiles = getFiles(filesOri, valSplit)
    valOutDir = os.path.join(data_folder,split)
    os.makedirs(valOutDir,exist_ok=True)
    preparePthFiles(valFiles, split, valOutDir)

    semantic_label_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    semantic_label_names = ['ground', 'Building', 'LowVegetation', 'MediumVegetation', 'HighVegetation', 'Vehicle',
                            'Truck', 'Aircraft', 'MilitaryVehicle', 'Bike', 'Motorcycle', 'LightPole', 'StreetSgin',
                            'Clutter', 'Fence']
    val_gtFolder = os.path.join(data_folder,'val_gt')
    os.makedirs(val_gtFolder,exist_ok=True)
    prepareInstGt(valOutDir, val_gtFolder, semantic_label_idxs)

    testSplit = [26,27,28]
    split = 'test'
    testFiles = getFiles(filesOri, testSplit)
    testOutDir = os.path.join(data_folder,split)
    os.makedirs(testOutDir,exist_ok=True)
    if len(testFiles)>0:
        preparePthFiles(testFiles, split, testOutDir)
