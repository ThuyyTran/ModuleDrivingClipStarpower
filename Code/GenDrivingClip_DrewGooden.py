import pandas as pd 
from moviepy.editor import VideoFileClip, concatenate_videoclips
import subprocess
import json
from tqdm import tqdm
import argparse
import os
import stat
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
import ast 
import timeit
import ast
# import Levenshtein as lev
import time
startTime = time.time()
with open('/home/ubuntu/DrivingClipsModule/DataBase/DrewGooden_DataPose_15d_newdata/StartListVideo.json') as json_file:
    dataStart = json.load(json_file)
with open('/home/ubuntu/DrivingClipsModule/DataBase/DrewGooden_DataPose_15d_newdata/ErrorVideoList.txt') as f:
    ListErroFile = [line.rstrip() for line in f]
Threshold_MaxYaw  = 9
Threshold_MaxPitch = 9
Threshold_MaxRoll  = 9
Threshold_MinYaw   = -9
Threshold_MinPitch = -9
Threshold_MinRoll =-9
def filterClips(emotion_in,duration_in,featureClips,idChoosen):
    """
        Choose video base on emotion
    """
    if idChoosen == 0 :
        startList = dataStart[emotion_in]
    else:
        startList = []
    listClips = {}
    for keyFilename in featureClips.keys():
        emotion = str(featureClips[keyFilename]['Foldername']).split('/')[0]
        duration = float(featureClips[keyFilename]['Duration'])
        MaxYaw = float(featureClips[keyFilename]['YawMax'])
        MinYaw = float(featureClips[keyFilename]['YawMin'])
        MaxRoll = float(featureClips[keyFilename]['RollMax'])
        MinRoll = float(featureClips[keyFilename]['RollMin'])
        MaxPitch = float(featureClips[keyFilename]['PitchMax'])
        MinPitch = float(featureClips[keyFilename]['PitchMin'])
        startx_min = featureClips[keyFilename]['Start_xmin']
        starty_min = featureClips[keyFilename]['Start_ymin']
        startx_max = featureClips[keyFilename]['Start_xmax']
        starty_max = featureClips[keyFilename]['Start_ymax']
        endx_min = featureClips[keyFilename]['End_xmin']
        endy_min = featureClips[keyFilename]['End_ymin']
        endx_max = featureClips[keyFilename]['End_xmax']
        endy_max = featureClips[keyFilename]['End_ymax']
        startYaw = featureClips[keyFilename]['YawStart']
        startRoll = featureClips[keyFilename]['RollStart']
        startPitch = featureClips[keyFilename]['PitchStart']
        endYaw = featureClips[keyFilename]['YawEnd']
        endRoll = featureClips[keyFilename]['RollEnd']
        endPitch = featureClips[keyFilename]['PitchEnd']
        distance1 = np.sqrt(np.power((endx_min-startx_min),2)+np.power((endy_min-starty_min),2))
        distance2 = np.sqrt(np.power((endx_max-startx_max),2)+np.power((endy_max-starty_max),2))
        conditions = (MaxYaw <= Threshold_MaxYaw and MinYaw>=Threshold_MinYaw and   
                MaxPitch<= Threshold_MaxPitch and MinPitch>= Threshold_MinPitch and 
                MaxRoll  <= Threshold_MaxRoll and MinRoll>= Threshold_MinRoll)
        conditionBasePose = (startYaw <=10 and startPitch<=10 and
                            startRoll<=10 and endYaw<=10 and 
                            endPitch<=10 and endRoll<=10)
        # conditionBasePose = True
        # if emotion_in == emotion and duration <= duration_in:
        #     print(keyFilename)
        #     print(conditions)
        #     print(keyFilename in startList)
        #     print('==================')
        if idChoosen == 0:
            if (distance1 <= 50 and distance2<=80 and emotion_in == emotion  and duration <= duration_in and 
                conditions and conditionBasePose and
                featureClips[keyFilename]['NumberNeighborhood']>0 and ( (startYaw<=8 and startPitch<=8 and startRoll<=8))
                )or (distance1 <= 50 and distance2<=80 and emotion == 'NeutralNonspeaking'  and duration <= duration_in and 
                conditions and conditionBasePose and
                featureClips[keyFilename]['NumberNeighborhood']>0  and ( (startYaw<=8 and startPitch<=8 and startRoll<=8))):
                listClips[keyFilename] = featureClips[keyFilename]
        else:
            if (distance1 <= 50 and distance2<=80 and emotion_in == emotion  and duration <= duration_in and 
                conditions and conditionBasePose and
                featureClips[keyFilename]['NumberNeighborhood']>0
                ):
                listClips[keyFilename] = featureClips[keyFilename]
    data_sorted = dict(sorted(listClips.items(), key=lambda x: (x[0],-x[1]['Duration'], -x[1]['NumberNeighborhood'])))
    # data_sorted = dict(sorted(listClips.items(), key=lambda x: (-x[1]['Duration'], -x[1]['NumberNeighborhood'])))
    return data_sorted
def convertPandasToDict(data):
    result = {}
    for i in range(len(data)):
        if pd.isna(data['Duration'][i]):
            continue
        result[str(data['Filename'][i])] = {
            'Foldername':data['Foldername'][i],
            'Start_ymin':data['Start_ymin'][i],
            'Start_ymax':data['Start_ymax'][i],
            'Start_xmin':data['Start_xmin'][i],
            'Start_xmax':data['Start_xmax'][i],
            'End_ymin':data['End_ymin'][i],
            'End_ymax':data['End_ymax'][i],
            'End_xmin':data['End_xmin'][i],
            'End_xmax':data['End_xmax'][i],
            'YawStart':data['YawStart'][i],
            'RollStart':data['RollStart'][i],
            'PitchStart':data['PitchStart'][i],
            'YawEnd':data['YawEnd'][i],
            'RollEnd':data['RollEnd'][i],
            'PitchEnd':data['PitchEnd'][i],
            'YawMax':data['YawMax'][i],
            'YawMin':data['YawMin'][i],
            'RollMax':data['RollMax'][i],
            'RollMin':data['RollMin'][i],
            'PitchMax':data['PitchMax'][i],
            'PitchMin':data['PitchMin'][i],
            'Duration':int(data['Duration'][i]),
            'BBox':data['BBox'][i],
            'NumberNeighborhood':int(data['NumberNeighborhood'][i]),
            'FileNameNeighborhood':data['FileNameNeighborhood'][i],
            'NumberNeighborhoodAngry':data['NumberNeighborhoodAngry'][i] ,
            'NumberNeighborhoodNeutral':data['NumberNeighborhoodNeutral'][i],
            'NumberNeighborhoodSad':data['NumberNeighborhoodSad'][i],
            'NumberNeighborhoodHappy':data['NumberNeighborhoodHappy'][i],
            'NumberNeighborhoodNeutralNonspeaking':data['NumberNeighborhoodNeutralNonSpeaking'][i],
            'NumberNeighborhoodAngry1s':data['NumberNeighborhoodAngry1s'][i] ,
            'NumberNeighborhoodNeutral1s':data['NumberNeighborhoodNeutral1s'][i],
            'NumberNeighborhoodSad1s':data['NumberNeighborhoodSad1s'][i],
            'NumberNeighborhoodHappy1s':data['NumberNeighborhoodHappy1s'][i],
            'NumberNeighborhoodNeutralNonspeaking1s':data['NumberNeighborhoodNeutralNonSpeaking1s'][i],
            'Emotion':str(data['Foldername'][i]).split('/')[0]
        }
    return result
def dfs(graph, node, path, current_Duration, target_Duration, deadlock, last_node, uniqueList):
    current_Duration += graph[node]['Duration']
    path.append(node)
    MaxYaw = float(graph[node]['YawMax'])
    MinYaw = float(graph[node]['YawMin'])
    MaxRoll = float(graph[node]['RollMax'])
    MinRoll = float(graph[node]['RollMin'])
    MaxPitch = float(graph[node]['PitchMax'])
    MinPitch = float(graph[node]['PitchMin'])
    startYaw = float(graph[node]['YawStart'])
    startRoll = float(graph[node]['RollStart'])
    startPitch = float(graph[node]['PitchStart'])
    endYaw = float(graph[node]['YawEnd'])
    endRoll = float(graph[node]['RollEnd'])
    endPitch = float(graph[node]['PitchEnd'])
    if MaxYaw > Threshold_MaxYaw or MinYaw<Threshold_MinYaw or MaxPitch>Threshold_MaxPitch or MinPitch<Threshold_MinPitch or MaxRoll>Threshold_MaxRoll or MinRoll<Threshold_MinRoll :
        return None
    # if startYaw >10 and startPitch>10 and  startRoll>10 and endYaw>10 and endPitch>10 and endRoll>10:
    #     return None
    if node in uniqueList or node in ListErroFile :
        return None
    else: 
        uniqueList.append(node)
    if current_Duration > target_Duration:
        return None
    # if current_Duration == target_Duration or (current_Duration >= target_Duration-1 and current_Duration <= target_Duration):
    if current_Duration == target_Duration:
    # if current_Duration == target_Duration :
        if node not in deadlock:
            return path
        else:
            return None
    neighborStage = ast.literal_eval(graph[node]['FileNameNeighborhood'])
    for neighbor in neighborStage:
        result_path = dfs(graph, neighbor, path[:], current_Duration, target_Duration, deadlock, node,uniqueList)
        if result_path is not None:
            return result_path
    return None

def ProcessWithBackTrack(dataTimeStamp,featureClips):
    dictDataTemp = {}
    listUnique = []
    backupListUnique = []
    backupData = {'BestId':-1}
    idChosen = 0
    for i in range(len(dataTimeStamp)):
        dictDataTemp[i] = {'Visited':False,'Deadlock':[]}
    checkPose = False
    while idChosen < len(dataTimeStamp):
        if idChosen == 0:
            checkPose = False
        if idChosen<0:
            break
        emotion = dataTimeStamp[idChosen][1]
        targetDuration = dataTimeStamp[idChosen][0]
        if targetDuration<1:
            roundTargetDuration = 1
        else:
            roundTargetDuration = np.round(targetDuration)
        if targetDuration==0:
            idChosen+=1
            continue
        if idChosen+1==len(dataTimeStamp):
            nextEmotion = None
            nextDuration = 0
        else:
            nextEmotion = dataTimeStamp[idChosen+1][1]
            nextDuration = dataTimeStamp[idChosen+1][0]
        #Deadlock node: Can't move to any other node
        if len(dictDataTemp[idChosen]['Deadlock'])==0:
            deadlockList = []
            if nextEmotion!=None:
                for filenamekey in featureClips.keys():
                    if np.round(nextDuration) == 1:
                        if featureClips[filenamekey]['Emotion'] ==  emotion and featureClips[filenamekey]['NumberNeighborhood'+nextEmotion+'1s']==0:
                            deadlockList.append(filenamekey)
                    else:
                        if featureClips[filenamekey]['Emotion'] ==  emotion and featureClips[filenamekey]['NumberNeighborhood'+nextEmotion]==0:
                            deadlockList.append(filenamekey)
            data_sorted = filterClips(emotion,roundTargetDuration,featureClips,idChosen)
        else:
            deadlockList = dictDataTemp[idChosen]['Deadlock']
        tmpPath = []
        if checkPose == False:
            for i in range(len(data_sorted)):
                tmpPath = dfs(featureClips,list(data_sorted.keys())[i],[], 0, roundTargetDuration, deadlockList, '',[])
                if tmpPath:
                    yawEnd, pitchEnd, rollEnd = featureClips[tmpPath[-1]]['YawEnd'],featureClips[tmpPath[-1]]['PitchEnd'],featureClips[tmpPath[-1]]['RollEnd']
                    dictDataTemp[idChosen]['YawEnd'] = yawEnd
                    dictDataTemp[idChosen]['PitchEnd'] = pitchEnd
                    dictDataTemp[idChosen]['RollEnd'] = rollEnd
                    dictDataTemp[idChosen]['IdSelected'] = i
                    dictDataTemp[idChosen]['ListSelected'] = tmpPath
                    dictDataTemp[idChosen]['Duration'] = targetDuration
                    dictDataTemp[idChosen]['Deadlock'] = deadlockList
                    dictDataTemp[idChosen]['DeadlockNext'] = []
                    break
            if tmpPath == None:
                print('No solutions')
                break
            else:
                dictDataTemp[idChosen]['DeadlockNext'] = []
                if idChosen> backupData['BestId']:
                    # backupData[idChosen] = {'ListSelected':tmpPath}
                    backupData['BestId'] = idChosen
                    backupListUnique = listUnique.copy()
                    backupData[idChosen] = dict(dictDataTemp[idChosen])
                if idChosen == 0:
                    checkPose = True
        else:
            datanow = {}
            for key in data_sorted.keys():
                if np.abs(data_sorted[key]['YawStart']-dictDataTemp[idChosen-1]['YawEnd'])<=10 and np.abs(data_sorted[key]['PitchStart']-dictDataTemp[idChosen-1]['PitchEnd'])<=10 and np.abs(data_sorted[key]['RollStart']-dictDataTemp[idChosen-1]['RollEnd'])<=10 and key not in dictDataTemp[idChosen-1]['DeadlockNext']:
                    datanow[key] = data_sorted[key]
            # tmpsort =  sorted(datanow.keys(), key=lambda x: lev.distance(x, dictDataTemp[0]['ListSelected'][-1]))
            # sortDict = {}
            # for key in datanow:
            #     sortDict[key] = datanow[key]
            # datanow = sortDict
            for i in range(len(datanow)):
                listUniqueFlatten = []
                for idex in range(len(listUnique)):
                    for value in listUnique[idex]:
                        listUniqueFlatten.append(value)
                tmpPath = dfs(featureClips,list(datanow.keys())[i],[], 0, roundTargetDuration, deadlockList, '',listUniqueFlatten)
                if tmpPath:
                    yawEnd, pitchEnd, rollEnd = featureClips[tmpPath[-1]]['YawEnd'],featureClips[tmpPath[-1]]['PitchEnd'],featureClips[tmpPath[-1]]['RollEnd']
                    dictDataTemp[idChosen]['YawEnd'] = yawEnd
                    dictDataTemp[idChosen]['PitchEnd'] = pitchEnd
                    dictDataTemp[idChosen]['RollEnd'] = rollEnd
                    dictDataTemp[idChosen]['IdSelected'] = i
                    dictDataTemp[idChosen]['ListSelected'] = tmpPath
                    dictDataTemp[idChosen]['Duration'] = targetDuration
                    dictDataTemp[idChosen]['Deadlock'] = deadlockList
                    if 'DeadlockNext' not in list(dictDataTemp[idChosen].keys()):
                        dictDataTemp[idChosen]['DeadlockNext'] = []
                    break
                else:
                    if 'DeadlockNext' not in list(dictDataTemp[idChosen].keys()):
                        dictDataTemp[idChosen]['DeadlockNext'] = [list(datanow.keys())[i]]
                    else:
                        dictDataTemp[idChosen]['DeadlockNext'].append(list(datanow.keys())[i])
        if tmpPath:
            listUnique.append(tmpPath)
            if idChosen>backupData['BestId']:
                backupData['BestId'] = idChosen
                backupListUnique = listUnique.copy()
                backupData[idChosen] = dict(dictDataTemp[idChosen])
            idChosen+=1
        else:
            if 'DeadlockNext' not in list(dictDataTemp[idChosen].keys()):
                dictDataTemp[idChosen]['DeadlockNext'] = list(datanow.keys())
            else:
                try:
                    dictDataTemp[idChosen]['DeadlockNext']+=list(datanow.keys())
                except:
                    dictDataTemp[idChosen]['DeadlockNext']+=list(data_sorted.keys())
            dictDataTemp[idChosen]['Deadlock'] = []
            dictDataTemp[idChosen]['ListSelected'] = []
            if len(listUnique)>0:
                listUnique.pop()
            if idChosen-1>=0:
                dictDataTemp[idChosen-1]['Deadlock'].append(dictDataTemp[idChosen-1]['ListSelected'][-1])
            idChosen-=1
    if len(backupData.keys())-1 <len(dataTimeStamp):
        idChosen = len(backupData.keys())-1
        dictDataTemp = {}
        for i in range(idChosen,len(dataTimeStamp)):
            dictDataTemp[i] = {'Visited':False,'Deadlock':[]}
        checkPose = False
        while idChosen < len(dataTimeStamp):
            if idChosen == len(backupData.keys())-1:
                checkPose = False
            if idChosen<len(backupData.keys())-1:
                print('NO SOLUTION')
                break
            emotion = dataTimeStamp[idChosen][1]
            targetDuration = dataTimeStamp[idChosen][0]
            if targetDuration<1:
                roundTargetDuration = 1
            else:
                roundTargetDuration = np.round(targetDuration)
            if targetDuration==0:
                idChosen+=1
                continue
            if idChosen+1==len(dataTimeStamp):
                nextEmotion = None
                nextDuration = 0
            else:
                nextEmotion = dataTimeStamp[idChosen+1][1]
                nextDuration = dataTimeStamp[idChosen+1][0]
            #Deadlock node: Can't move to any other node
            if len(dictDataTemp[idChosen]['Deadlock'])==0:
                deadlockList = []
                if nextEmotion!=None:
                    for filenamekey in featureClips.keys():
                        if np.round(nextDuration) == 1:
                            if featureClips[filenamekey]['Emotion'] ==  emotion and featureClips[filenamekey]['NumberNeighborhood'+nextEmotion+'1s']==0:
                                deadlockList.append(filenamekey)
                        else:
                            if featureClips[filenamekey]['Emotion'] ==  emotion and featureClips[filenamekey]['NumberNeighborhood'+nextEmotion]==0:
                                deadlockList.append(filenamekey)
                data_sorted = filterClips(emotion,roundTargetDuration,featureClips,idChosen)
            else:
                deadlockList = dictDataTemp[idChosen]['Deadlock']
            tmpPath = []
            if checkPose == False:
                for i in range(len(data_sorted)):
                    tmpPath = dfs(featureClips,list(data_sorted.keys())[i],[], 0, roundTargetDuration, deadlockList, '',[])
                    if tmpPath:
                        yawEnd, pitchEnd, rollEnd = featureClips[tmpPath[-1]]['YawEnd'],featureClips[tmpPath[-1]]['PitchEnd'],featureClips[tmpPath[-1]]['RollEnd']
                        dictDataTemp[idChosen]['YawEnd'] = yawEnd
                        dictDataTemp[idChosen]['PitchEnd'] = pitchEnd
                        dictDataTemp[idChosen]['RollEnd'] = rollEnd
                        dictDataTemp[idChosen]['IdSelected'] = i
                        dictDataTemp[idChosen]['ListSelected'] = tmpPath
                        dictDataTemp[idChosen]['Duration'] = targetDuration
                        dictDataTemp[idChosen]['Deadlock'] = deadlockList
                        break
                if tmpPath == None:
                    print('No solutions')
                    break
                else:
                    if idChosen == len(backupData.keys())-1:
                        checkPose = True
            else:
                datanow = {}
                for key in data_sorted.keys():
                    if np.abs(data_sorted[key]['YawStart']-dictDataTemp[idChosen-1]['YawEnd'])<=10 and np.abs(data_sorted[key]['PitchStart']-dictDataTemp[idChosen-1]['PitchEnd'])<=10 and np.abs(data_sorted[key]['RollStart']-dictDataTemp[idChosen-1]['RollEnd'])<=10:
                        datanow[key] = data_sorted[key]
                for i in range(len(datanow)):
                    listUniqueFlatten = []
                    for idex in range(len(backupListUnique)):
                        for value in backupListUnique[idex]:
                            listUniqueFlatten.append(value)
                    tmpPath = dfs(featureClips,list(datanow.keys())[i],[], 0, roundTargetDuration, deadlockList, '',listUniqueFlatten)
                    if tmpPath:
                        yawEnd, pitchEnd, rollEnd = featureClips[tmpPath[-1]]['YawEnd'],featureClips[tmpPath[-1]]['PitchEnd'],featureClips[tmpPath[-1]]['RollEnd']
                        dictDataTemp[idChosen]['YawEnd'] = yawEnd
                        dictDataTemp[idChosen]['PitchEnd'] = pitchEnd
                        dictDataTemp[idChosen]['RollEnd'] = rollEnd
                        dictDataTemp[idChosen]['IdSelected'] = i
                        dictDataTemp[idChosen]['ListSelected'] = tmpPath
                        dictDataTemp[idChosen]['Duration'] = targetDuration
                        dictDataTemp[idChosen]['Deadlock'] = deadlockList
                        break
            if tmpPath:
                idChosen+=1
                backupListUnique.append(tmpPath)
            else:
                dictDataTemp[idChosen]['Deadlock'] = []
                dictDataTemp[idChosen]['ListSelected'] = []
                if len(backupListUnique)>0:
                    backupListUnique.pop()
                if idChosen-1>=len(backupData.keys())-1:
                    dictDataTemp[idChosen-1]['Deadlock'].append(dictDataTemp[idChosen-1]['ListSelected'][-1])
                idChosen-=1
        if idChosen == len(dataTimeStamp):
            for key in dictDataTemp.keys():
                backupData[key] = dictDataTemp[key]
    results = []
    try:
        for key in backupData.keys():
            tmp = []
            if key == 'BestId':
                continue
            for value in backupData[key]['ListSelected']:
                tmp.append(value)
            results.append(tmp)
    except:
        print('Not enough video to combine')
        exit()
    return results
def combineVideo(referenceClips,DetailData,featureClips,path_dataset,path_audio,savepath,config,ref_poses,mode):
    finalVideoList = []
    listVideoCombine = []
    dataJson = {}
    # dataJson = []
    # mapName2Video = {}
    #get id video pose
    # for key in ref_poses:
    #     mapName2Video[os.path.basename(ref_poses[key]['path'])] = key
    if len(referenceClips)!=0:
        for i in range(len(referenceClips)):
            dataJson[str(i)] = {} 
            newName = list(DetailData.keys())[i]
            expDuration = DetailData[list(DetailData.keys())[i]]['Duration']
            dataJson[str(i)]['Emotion'] = DetailData[newName]['Emotion'] 
            dataJson[str(i)]['StartFrame'] = int(DetailData[newName]['Start']*30)
            dataJson[str(i)]['StartTime'] = int(DetailData[newName]['Start']*1000)
            dataJson[str(i)]['EndFrame'] = int(DetailData[newName]['Duration']*30) + int(DetailData[newName]['Start']*30)
            dataJson[str(i)]['EndTime'] = DetailData[newName]['Duration']*1000 + DetailData[newName]['Start']*1000
            dataJson[str(i)]['ExpectedFrames'] = int(DetailData[newName]['Duration']*30)
            dataJson[str(i)]['ExpectedDuration'] = DetailData[newName]['Duration']*1000
            dataJson[str(i)]['Mode'] = mode
            dataJson[str(i)]['Clips'] = {}
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter('/home/ubuntu/DrivingClipsModule/Temp/TempVideo/'+newName+'.mp4', fourcc, 30, (config['size'], config['size']))
            endx_min_bf = featureClips[os.path.basename(referenceClips[0][0])]['End_xmin']
            endy_min_bf = featureClips[os.path.basename(referenceClips[0][0])]['End_ymin']
            endx_max_bf = featureClips[os.path.basename(referenceClips[0][0])]['End_xmax']
            endy_max_bf = featureClips[os.path.basename(referenceClips[0][0])]['End_ymax']
            paddx_ori = int((config['size']-(endx_max_bf-endx_min_bf))/2)
            paddy_ori = int((config['size']-(endy_max_bf-endy_min_bf))/2)
            idxVideo = 0
            finalVideoList.append('/home/ubuntu/DrivingClipsModule/Temp/TempVideoAudio/'+newName+'.mp4')
            # tmp = []
            realFrame = 0
            for nameVideo in referenceClips[i]:
                pathVideo = os.path.join(path_dataset,featureClips[nameVideo]['Foldername'],nameVideo)
                # print(pathVideo)
                basename = os.path.basename(pathVideo)
                if not os.path.exists(pathVideo):
                    pathVideo = os.path.join(path_dataset,featureClips[nameVideo]['Foldername'].split('/')[0]+'/'+'2s',nameVideo)
                    if not os.path.exists(pathVideo):
                        pathVideo = os.path.join(path_dataset,featureClips[nameVideo]['Foldername'].split('/')[0]+'/'+'3s',nameVideo)
                if not os.path.exists(pathVideo):
                    print(os.path.join(path_dataset,featureClips[nameVideo]['Foldername'],nameVideo))
                    exit()
                # dataJson[str(i)]['Clips'][basename] = {'Path':os.path.join(featureClips[nameVideo]['Foldername'],nameVideo),'Duration':featureClips[basename]['DurationOriginal']*1000}
                endx_min = featureClips[basename]['End_xmin']
                endy_min = featureClips[basename]['End_ymin']
                endx_max = featureClips[basename]['End_xmax']
                endy_max = featureClips[basename]['End_ymax']
                video = cv2.VideoCapture(pathVideo)
                dataJson[str(i)]['Clips'][basename] = {'Path':os.path.join(featureClips[nameVideo]['Foldername'],nameVideo),'Duration':featureClips[basename]['Duration']*1000,'Frames':int(video.get(cv2.CAP_PROP_FRAME_COUNT))}
                numFrame = 0
                paddx = int(((endx_max-endx_min)*paddx_ori)/(endx_max_bf-endx_min_bf))
                paddy = int(((endy_max-endy_min)*paddy_ori)/(endy_max_bf-endy_min_bf))
                # index = 0
                while True:
                    ret,frame = video.read()
                    if ret == False:
                        break
                    startX = np.max([int(endx_min)-paddx,0])
                    startY = np.max([int(endy_min-paddy),0])
                    endX = np.min([frame.shape[1],int(endx_max)+paddx])
                    endY = np.min([frame.shape[0],int(endy_max+(paddy))])
                    cropFrame = frame[startY:endY,startX:endX]
                    # try:
                    #     cropFrame = frame[int(ast.literal_eval(featureClips[basename]['BBox'].split(';')[index])[1])-50:int(ast.literal_eval(featureClips[basename]['BBox'].split(';')[index])[3])+50,int(ast.literal_eval(featureClips[basename]['BBox'].split(';')[index])[0])-50:int(ast.literal_eval(featureClips[basename]['BBox'].split(';')[index])[2])+50]
                    # except:
                    #     cropFrame = frame[int(ast.literal_eval(featureClips[basename]['BBox'].split(';')[len(featureClips[basename]['BBox'].split(';'))-1])[1])-50:int(ast.literal_eval(featureClips[basename]['BBox'].split(';')[len(featureClips[basename]['BBox'].split(';'))-1])[3])+50,int(ast.literal_eval(featureClips[basename]['BBox'].split(';')[len(featureClips[basename]['BBox'].split(';'))-1])[0])-50:int(ast.literal_eval(featureClips[basename]['BBox'].split(';')[len(featureClips[basename]['BBox'].split(';'))-1])[2])+50]
                    height_ori, width_ori, _ = cropFrame.shape
                    if height_ori!=config['size'] or width_ori!=config['size']:
                        cropFrame = cv2.resize(cropFrame,(config['size'],config['size']))
                    numFrame+=1
                    # index+=1
                    out.write(cropFrame)
                idxVideo+=1
            out.release()
            # if len(tmp)==1:
            #     dataJson.append({'id':tmp[0]['Id'],'frames':tmp[0]['TotalFrame'],'expected_frames':int(DetailData[newName]['Duration']*30)})
            # elif len(tmp) > 1:
            #     stretchFrameEachClip = int((DetailData[newName]['Duration']*30-realFrame)/len(tmp))
                # for j in range(len(tmp)):
                #     if j == len(tmp)-1:
                #         # print(int(int(DetailData[newName]['Duration']*30)-stretchFrameEachClip*(len(tmp)-1)),stretchFrameEachClip*(len(tmp)-1),(DetailData[newName]['Duration']*30))
                #         # exit()
                #         dataJson.append({'id':tmp[j]['Id'],'frames':tmp[j]['TotalFrame'],'expected_frames':int(tmp[j]['TotalFrame'])+int((DetailData[newName]['Duration']*30-realFrame)-stretchFrameEachClip*(len(tmp)-1))})
                #     else:
                #         dataJson.append({'id':tmp[j]['Id'],'frames':tmp[j]['TotalFrame'],'expected_frames':int(tmp[j]['TotalFrame']+stretchFrameEachClip)})
            clips = VideoFileClip('/home/ubuntu/DrivingClipsModule/Temp/TempVideo/'+newName+'.mp4',verbose=False)
            dataJson[str(i)]['OriginalDuration'] = clips.duration*1000
            dataJson[str(i)]['Stretch_Time'] = np.abs(dataJson[str(i)]['ExpectedDuration'] - dataJson[str(i)]['OriginalDuration'])
            dataJson[str(i)]['OriginalFrameCount'] = sum(video['Frames'] for video in dataJson[str(i)]['Clips'].values())
            dataJson[str(i)]['Stretch_Frames'] = int(np.abs(dataJson[str(i)]['ExpectedFrames'] - dataJson[str(i)]['OriginalFrameCount']))
            new_clip = clips.speedx(final_duration=expDuration)
            listVideoCombine.append(new_clip)
            new_clip.write_videofile('/home/ubuntu/DrivingClipsModule/Temp/TempVideoSpeed/'+newName+'.mp4',logger=None)
            if newName.startswith('NonSpeaking'):
                new_clip.write_videofile('/home/ubuntu/DrivingClipsModule/Temp/TempVideoAudio/'+newName+'.mp4',logger=None)
            else:
                from videoprops import get_audio_properties
                fileClip = os.path.join('/home/ubuntu/DrivingClipsModule/Temp/TempVideoSpeed',newName+'.mp4')
                fileAudio = os.path.join(path_audio)
                props = get_audio_properties(fileAudio)
                # if props['codec_name'] == 'pcm_s16le':
                new_clip.write_videofile('/home/ubuntu/DrivingClipsModule/Temp/TempVideoAudio/'+newName+'.mp4',logger=None)
                # else:
                #     fileResult = os.path.join('/home/ubuntu/DrivingClipsModule/Temp/TempVideoAudio',newName+'.mp4')
                #     os.system("ffmpeg -y -i "+fileClip+" -i "+fileAudio+" -c copy "+fileResult)
        clips = [VideoFileClip(file) for file in finalVideoList]
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(savepath)
    else:
        print('NoneVideo')
    return dataJson
def getDataFromJson(dataJson):
    listTimeStamp = {}
    listTimeStamp[str(dataJson['audio_path']).split('/')[-1].split('.')[0]] = {'Start': (float(dataJson['timestamp'])-float(dataJson['merged_timestamp']))/1000,'Duration':(float(dataJson['end_time'])-float(dataJson['timestamp']))/1000,'Emotion':dataJson['emotion']}
    for i in range(len(dataJson['silences'])):
        dataChosen = dataJson['silences'][i]
        if float(dataChosen['duration'])/1000 >=0.1:
            listTimeStamp['NonSpeaking_'+str(i)] = {'Start':(float(dataChosen['timestamp'])-float(dataJson['merged_timestamp']))/1000,'Duration':float(dataChosen['duration'])/1000,'Emotion':'NeutralNonspeaking'}
        else:
            # print('====')
            # print(float(dataChosen['duration'])/1000 >=100)
            # print( float(dataChosen['duration'])/1000)
            # exit()
            listTimeStamp[str(dataJson['audio_path']).split('/')[-1].split('.')[0]]['Duration'] = listTimeStamp[str(dataJson['audio_path']).split('/')[-1].split('.')[0]]['Duration'] + float(dataChosen['duration'])/1000
    sorted_dict = dict(sorted(listTimeStamp.items(), key=lambda x: x[1]['Start']))
    dataTimeStamp = []
    for key in sorted_dict.keys():
        dataTimeStamp.append((sorted_dict[key]['Duration'],sorted_dict[key]['Emotion']))
    return sorted_dict,dataTimeStamp
def main():
    parser = argparse.ArgumentParser(description='Extract information from a sentence and save it to a file.')
    parser.add_argument('--data', type=str, help='Json data')
    args = parser.parse_args()
    data_arg = args.data.replace("\\/", "/")
    data_arg = data_arg.replace("Dark Doc", "Neutral")
    dataJson = ast.literal_eval(data_arg)
    DetailData,dataTimeStamp = getDataFromJson(dataJson)
    # print(DetailData)
    # print(dataTimeStamp)
    # exit()
    # DetailData= {"NonSpeaking_0": {"Start": 0.0, "Duration": 3.517, "Emotion": "NeutralNonspeaking"}, "5062_1": {"Start": 3.517, "Duration": 10.95, "Emotion": "Neutral"}}
    # dataTimeStamp = [[3.517, "NeutralNonspeaking"], [10.95, "Neutral"]]
    # DetailData = {"NonSpeaking_0": {"Start": 0.0, "Duration": 4.464, "Emotion": "NeutralNonspeaking"}, "5110_1": {"Start": 4.464, "Duration": 23.7, "Emotion": "Angry"}}
    # dataTimeStamp= [[4.464, "NeutralNonspeaking"], [23.7, "Angry"]]
    # f = open(os.path.join('/home/ubuntu/DrivingClipsModule/Config',dataJson['influencer']+'Config.json'))
    if dataJson['influencer'] == 'DrDisrespect':
        mode = 'Sunglasses'
    else:
        mode = 'Normal'
    f = open(os.path.join('/home/ubuntu/DrivingClipsModule/Config','DrewGoodenConfig.json'))
    config = json.load(f)
    # ref_poses = json.load(open(os.path.join('/home/ubuntu/DrivingClipsModule/DataBase/Feature/',dataJson['influencer']+'_ref_poses.json')))
    data = pd.read_csv(config['feature_path'])
    featureClips = convertPandasToDict(data)
    path_dataset = config['dataset_path']
    path_audio = dataJson['audio_path']
    savepath = dataJson['save_path']
    pathTempVideo = '/home/ubuntu/DrivingClipsModule/Temp/TempVideo'
    pathTempVideoSpeed = '/home/ubuntu/DrivingClipsModule/Temp/TempVideoSpeed'
    pathTempVideoAudio = '/home/ubuntu/DrivingClipsModule/Temp/TempVideoAudio'
    if not os.path.exists(pathTempVideo):
        os.makedirs(pathTempVideo, mode=0o777)
        os.chmod(pathTempVideo, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    if not os.path.exists(pathTempVideoSpeed):
        os.makedirs(pathTempVideoSpeed, mode=0o777)
        os.chmod(pathTempVideoSpeed, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    if not os.path.exists(pathTempVideoAudio):
        os.makedirs(pathTempVideoAudio,mode=0o777)
        os.chmod(pathTempVideoAudio, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    referenceClips = ProcessWithBackTrack(dataTimeStamp,featureClips)
    jsonData = combineVideo(referenceClips,DetailData,featureClips,path_dataset,path_audio,savepath,config,'',mode)
    out_file = open(dataJson['json_save_path'], "w")
    jsonData['DetailData'] = DetailData
    jsonData['dataTimeStamp'] = dataTimeStamp
    json.dump(jsonData, out_file,indent=4)
    out_file.close()
    # out_file1 = open('/home/ubuntu/DrivingClipsModule/Temp/Temp.json', "w")
    # json.dump(jsonData, out_file1,indent=4)
    # out_file1.close()
    #Remove temp folder
    os.system('rm -r '+pathTempVideo)
    os.system('rm -r '+pathTempVideoAudio)
    os.system('rm -r '+pathTempVideoSpeed)
    print(time.time()- startTime)
if __name__ == '__main__':
    main()
