import pandas as pd 
from moviepy.editor import VideoFileClip, concatenate_videoclips
import subprocess
import json
import random
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
from videoprops import get_audio_properties
# import Levenshtein as lev
import time
import random
startTime = time.time()
Threshold_MaxYaw  = 15
Threshold_MaxPitch = 15
Threshold_MaxRoll  = 15
Threshold_MinYaw   = -15
Threshold_MinPitch = -15
Threshold_MinRoll =-15
StartPoseAngle = 10
def main():
    parser = argparse.ArgumentParser(description='Extract information from a sentence and save it to a file.')
    parser.add_argument('--data', type=str, help='Json data')
    args = parser.parse_args()
    data_arg = args.data.replace("\\/", "/")
    data_arg = data_arg.replace("Dark Doc", "Neutral")
    dataJson = ast.literal_eval(data_arg)
    DetailData,dataTimeStamp = getDataFromJson(dataJson)
    # DetailData = {'NonSpeaking_0': {'Start': 0.0, 'Duration': 17.207, 'Emotion': 'NeutralNonspeaking'}, '5867_3': {'Start': 17.207, 'Duration': 13.6, 'Emotion': 'Neutral'}}
    # dataTimeStamp =[(17.207, 'NeutralNonspeaking'), (13.6, 'Neutral')]
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
    f = open(os.path.join('/media/anlab/data-2tb/ANLAB_THUY/ModuleDrivingClipStarpower/Config','SethMeyersConfig_NewAlgorithm.json'))
    config = json.load(f)
    global dataStart
    global ListErroFile
    global BaseVideoList
    global ListErrorStart
    if config['start_list_path'] !='':
        with open(config['start_list_path']) as json_file:
            dataStart = json.load(json_file)
    else:
        dataStart = []
    if config['list_error_path'] !='':
        with open(config['list_error_path']) as f:
            ListErroFile = [line.rstrip() for line in f]
    else:
        ListErroFile = []
    if config['base_list_path'] !='':
        with open(config['base_list_path']) as f:
            BaseVideoList = [line.rstrip() for line in f]
    else:
        BaseVideoList = []
    if config['base_list_path'] !='':
        with open(config['base_list_path']) as f:
            ListErrorStart = [line.rstrip() for line in f]
    else:
        ListErrorStart = []
    # ref_poses = json.load(open(os.path.join('/home/ubuntu/DrivingClipsModule/DataBase/Feature/',dataJson['influencer']+'_ref_poses.json')))
    data = pd.read_csv(config['feature_path'])
    featureClips = convertPandasToDict(data)
    path_dataset = config['dataset_path']
    path_audio = dataJson['audio_path']
    savepath = dataJson['save_path']
    pathTempVideo = '/media/anlab/data-2tb/ANLAB_THUY/ModuleDrivingClipStarpower/Temp/TempVideo'
    pathTempVideoSpeed = '/media/anlab/data-2tb/ANLAB_THUY/ModuleDrivingClipStarpower/Temp/TempVideoSpeed'
    pathTempVideoAudio = '/media/anlab/data-2tb/ANLAB_THUY/ModuleDrivingClipStarpower/Temp/TempVideoAudio'
    if not os.path.exists(pathTempVideo):
        os.makedirs(pathTempVideo, mode=0o777)
        os.chmod(pathTempVideo, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    if not os.path.exists(pathTempVideoSpeed):
        os.makedirs(pathTempVideoSpeed, mode=0o777)
        os.chmod(pathTempVideoSpeed, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    if not os.path.exists(pathTempVideoAudio):
        os.makedirs(pathTempVideoAudio,mode=0o777)
        os.chmod(pathTempVideoAudio, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    referenceClips,cutDurationOption = ProcessWithBackTrack(dataTimeStamp,featureClips)
    # print(referenceClips)
    # print(cutDurationOption)
    jsonData = combineVideo(referenceClips,cutDurationOption,DetailData,featureClips,path_dataset,path_audio,savepath,config,'',mode)
    out_file = open(dataJson['json_save_path'], "w")
    jsonData['DetailData'] = DetailData
    jsonData['dataTimeStamp'] = dataTimeStamp
    json.dump(jsonData, out_file,indent=4)
    out_file.close()
    #Remove temp folder
    os.system('rm -r '+pathTempVideo)
    os.system('rm -r '+pathTempVideoAudio)
    os.system('rm -r '+pathTempVideoSpeed)
    print(time.time()- startTime)
def randomly_arrange_segments(data):
    # Separate into segments
    segment_less_than_2s = {k: v for k, v in data.items() if v["Duration"] < 2}
    segment_less_than_3s = {k: v for k, v in data.items() if 2 <= v["Duration"] < 3}
    segment_greater_than_3s = {k: v for k, v in data.items() if v["Duration"] >= 3}

    # Shuffle keys within each segment
    keys_less_than_2s = list(segment_less_than_2s.keys())
    random.shuffle(keys_less_than_2s)
    sorted_segment_less_than_2s = {k: segment_less_than_2s[k] for k in keys_less_than_2s}

    keys_less_than_3s = list(segment_less_than_3s.keys())
    random.shuffle(keys_less_than_3s)
    sorted_segment_less_than_3s = {k: segment_less_than_3s[k] for k in keys_less_than_3s}

    keys_greater_than_3s = list(segment_greater_than_3s.keys())
    random.shuffle(keys_greater_than_3s)
    sorted_segment_greater_than_3s = {k: segment_greater_than_3s[k] for k in keys_greater_than_3s}
    # Combine into a single dictionary
    return {**sorted_segment_greater_than_3s,**sorted_segment_less_than_3s,**sorted_segment_less_than_2s}

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
        # distance1 = np.sqrt(np.power((endx_min-startx_min),2)+np.power((endy_min-starty_min),2))
        # distance2 = np.sqrt(np.power((endx_max-startx_max),2)+np.power((endy_max-starty_max),2))
        conditions = (MaxYaw <= Threshold_MaxYaw and MinYaw>=Threshold_MinYaw and   
                MaxPitch<= Threshold_MaxPitch and MinPitch>= Threshold_MinPitch and 
                MaxRoll  <= Threshold_MaxRoll and MinRoll>= Threshold_MinRoll)
        conditionBasePose = (startYaw <=StartPoseAngle and startPitch<=StartPoseAngle and
                            startRoll<=StartPoseAngle and endYaw<=StartPoseAngle and 
                            endPitch<=StartPoseAngle and endRoll<=StartPoseAngle and
                            startYaw >=-StartPoseAngle and startPitch>=-StartPoseAngle and
                            startRoll>=-StartPoseAngle and endYaw>=-StartPoseAngle and 
                            endPitch>=-StartPoseAngle and endRoll>=-StartPoseAngle)
        if idChoosen == 0:
            if (emotion_in == emotion  and 
                conditions and conditionBasePose and
                featureClips[keyFilename]['NumberNeighborhood']>0 and ( (startYaw<=10 and startPitch<=10 and startRoll<=10))
                 and keyFilename not in ListErroFile and (keyFilename in startList or keyFilename in BaseVideoList)) or (
                emotion == 'NeutralNonspeaking'  and 
                conditions  and
                featureClips[keyFilename]['NumberNeighborhood']>0  and keyFilename not in ListErroFile):
                listClips[keyFilename] = featureClips[keyFilename]
        else:
            if (emotion_in == emotion  and 
                conditions and conditionBasePose and
                featureClips[keyFilename]['NumberNeighborhood']>0 and keyFilename not in ListErroFile
                ):
                listClips[keyFilename] = featureClips[keyFilename]
    data_sorted = dict(sorted(listClips.items(), key=lambda x: (x[0],-x[1]['Duration'], -x[1]['NumberNeighborhood'])))
    data_sorted_random = randomly_arrange_segments(data_sorted)
    # data_sorted = dict(sorted(listClips.items(), key=lambda x: (-x[1]['Duration'], -x[1]['NumberNeighborhood'])))
    return data_sorted_random
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
            'Duration':data['Duration'][i],
            'BBox':data['BBox'][i],
            'NumberNeighborhood':int(data['NumberNeighborhood'][i]),
            'FileNameNeighborhood':data['FileNameNeighborhood'][i],
            'FileNameNeighborhoodCut1s':data['FileNameNeighborhoodCut1s'][i],
            'FileNameNeighborhoodCut2s':data['FileNameNeighborhoodCut2s'][i],
            'FileNameNeighborhoodCut3s':data['FileNameNeighborhoodCut3s'][i],
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
        return None,None
    if startYaw >StartPoseAngle and startPitch>StartPoseAngle and  startRoll>StartPoseAngle and endYaw>StartPoseAngle and endPitch>StartPoseAngle and endRoll>StartPoseAngle and startYaw <-StartPoseAngle and startPitch<-StartPoseAngle and  startRoll<-StartPoseAngle and endYaw<-StartPoseAngle and endPitch<-StartPoseAngle and endRoll<-StartPoseAngle:
        return None,None
    if node in uniqueList or node in ListErroFile :
        return None,None
    else: 
        uniqueList.append(node)
    if current_Duration > target_Duration and node in BaseVideoList:
        return path, (target_Duration - current_Duration,'CUT')
    elif current_Duration > target_Duration and node not in BaseVideoList:
        return None,None
    if current_Duration == target_Duration or (target_Duration- current_Duration <=0.1):
        if node not in deadlock:
            return path,(current_Duration,'STRETCH')
        else:
            return None,None
    neighborStage = ast.literal_eval(graph[node]['FileNameNeighborhood'])
    neighborStageCut1s = ast.literal_eval(graph[node]['FileNameNeighborhoodCut1s'])
    neighborStageCut2s = ast.literal_eval(graph[node]['FileNameNeighborhoodCut2s'])
    neighborStageCut3s = ast.literal_eval(graph[node]['FileNameNeighborhoodCut3s'])
    random.shuffle(neighborStage)
    random.shuffle(neighborStageCut1s)
    random.shuffle(neighborStageCut2s)
    random.shuffle(neighborStageCut3s)
    if current_Duration >= target_Duration-3 and len(neighborStageCut3s)>0:
        if current_Duration <= target_Duration and current_Duration >= target_Duration-3:
            if node not in deadlock:
                while len(neighborStageCut3s)>0:
                    idcut = random.randint(0,len(neighborStageCut3s)-1)
                    cutfile = neighborStageCut3s[idcut]
                    if cutfile not in ListErroFile:
                        path.append(cutfile)
                        return path, (target_Duration - current_Duration,'CUT')
                    else:
                        neighborStageCut3s.pop(idcut)
                        continue
            else:
                return None,None
    elif current_Duration >= target_Duration-2 and len(neighborStageCut2s)>0:
        if current_Duration <= target_Duration and current_Duration >= target_Duration-2:
            if node not in deadlock:
                while len(neighborStageCut2s)>0:
                    idcut = random.randint(0,len(neighborStageCut2s)-1)
                    cutfile = neighborStageCut2s[idcut]
                    if cutfile not in ListErroFile:
                        path.append(cutfile)
                        return path, (target_Duration - current_Duration,'CUT')
                    else:
                        neighborStageCut2s.pop(idcut)
                        continue
            else:
                return None,None
    elif current_Duration >= target_Duration-1 and len(neighborStageCut1s)>0:
        if current_Duration <= target_Duration and current_Duration >= target_Duration-1:
            if node not in deadlock:
                while len(neighborStageCut1s)>0:
                    idcut = random.randint(0,len(neighborStageCut1s)-1)
                    cutfile = neighborStageCut1s[idcut]
                    if cutfile not in ListErroFile:
                        path.append(cutfile)
                        return path, (target_Duration - current_Duration,'CUT')
                    else:
                        neighborStageCut1s.pop(idcut)
                        continue
            else:
                return None,None
    
    for neighbor in neighborStage:
        result_path, cutduration = dfs(graph, neighbor, path[:], current_Duration, target_Duration, deadlock, node,uniqueList)
        if result_path is not None:
            return result_path,cutduration
    return None,None
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
        print(idChosen)
        if idChosen == 0:
            checkPose = False
        if idChosen<0:
            break
        emotion = dataTimeStamp[idChosen][1]
        targetDuration = dataTimeStamp[idChosen][0]
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
                    if featureClips[filenamekey]['Emotion'] ==  emotion and len(ast.literal_eval(featureClips[filenamekey]['FileNameNeighborhoodCut1s']))==0 and len(ast.literal_eval(featureClips[filenamekey]['FileNameNeighborhoodCut2s']))==0 and len(ast.literal_eval(featureClips[filenamekey]['FileNameNeighborhoodCut3s']))==0:
                        deadlockList.append(filenamekey)
            data_sorted = filterClips(emotion,targetDuration,featureClips,idChosen)
        else:
            deadlockList = dictDataTemp[idChosen]['Deadlock']
        tmpPath = []
        if checkPose == False:
            for i in range(len(data_sorted)):
                tmpPath,cutDuration = dfs(featureClips,list(data_sorted.keys())[i],[], 0, targetDuration, deadlockList, '',[])
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
                    dictDataTemp[idChosen]['CutDuration'] = cutDuration
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
            # datanow = {}
            # for key in data_sorted.keys():
            #     # if np.abs(data_sorted[key]['YawStart']-dictDataTemp[idChosen-1]['YawEnd'])<=10 and np.abs(data_sorted[key]['PitchStart']-dictDataTemp[idChosen-1]['PitchEnd'])<=10 and np.abs(data_sorted[key]['RollStart']-dictDataTemp[idChosen-1]['RollEnd'])<=10 and key not in dictDataTemp[idChosen-1]['DeadlockNext']:
            #     if np.abs(data_sorted[key]['YawStart']-dictDataTemp[idChosen-1]['YawEnd'])<=10 and np.abs(data_sorted[key]['PitchStart']-dictDataTemp[idChosen-1]['PitchEnd'])<=10 and np.abs(data_sorted[key]['RollStart']-dictDataTemp[idChosen-1]['RollEnd'])<=10 and key not in dictDataTemp[idChosen-1]['DeadlockNext']:
            #         datanow[key] = data_sorted[key]
            for i in range(len(data_sorted)):
                listUniqueFlatten = []
                for idex in range(len(listUnique)):
                    for value in listUnique[idex]:
                        listUniqueFlatten.append(value)
                tmpPath,cutDuration = dfs(featureClips,list(data_sorted.keys())[i],[], 0, targetDuration, deadlockList, '',listUniqueFlatten)
                if tmpPath:
                    yawEnd, pitchEnd, rollEnd = featureClips[tmpPath[-1]]['YawEnd'],featureClips[tmpPath[-1]]['PitchEnd'],featureClips[tmpPath[-1]]['RollEnd']
                    dictDataTemp[idChosen]['YawEnd'] = yawEnd
                    dictDataTemp[idChosen]['PitchEnd'] = pitchEnd
                    dictDataTemp[idChosen]['RollEnd'] = rollEnd
                    dictDataTemp[idChosen]['IdSelected'] = i
                    dictDataTemp[idChosen]['ListSelected'] = tmpPath
                    dictDataTemp[idChosen]['Duration'] = targetDuration
                    dictDataTemp[idChosen]['Deadlock'] = deadlockList
                    dictDataTemp[idChosen]['CutDuration'] = cutDuration
                    if 'DeadlockNext' not in list(dictDataTemp[idChosen].keys()):
                        dictDataTemp[idChosen]['DeadlockNext'] = []
                    break
                else:
                    if 'DeadlockNext' not in list(dictDataTemp[idChosen].keys()):
                        dictDataTemp[idChosen]['DeadlockNext'] = [list(data_sorted.keys())[i]]
                    else:
                        dictDataTemp[idChosen]['DeadlockNext'].append(list(data_sorted.keys())[i])
        if tmpPath:
            listUnique.append(tmpPath)
            if idChosen>backupData['BestId']:
                backupData['BestId'] = idChosen
                backupListUnique = listUnique.copy()
                backupData[idChosen] = dict(dictDataTemp[idChosen])
            idChosen+=1
        else:
            if 'DeadlockNext' not in list(dictDataTemp[idChosen].keys()):
                dictDataTemp[idChosen]['DeadlockNext'] = list(data_sorted.keys())
            else:
                try:
                    dictDataTemp[idChosen]['DeadlockNext']+=list(data_sorted.keys())
                except:
                    dictDataTemp[idChosen]['DeadlockNext']+=list(data_sorted.keys())
            dictDataTemp[idChosen]['Deadlock'] = []
            dictDataTemp[idChosen]['ListSelected'] = []
            if len(listUnique)>0:
                listUnique.pop()
            if idChosen-1>=0:
                dictDataTemp[idChosen-1]['Deadlock'].append(dictDataTemp[idChosen-1]['ListSelected'][-1])
            idChosen-=1
    results = []
    durationCut = []
    try:
        for key in backupData.keys():
            tmp = []
            if key == 'BestId':
                continue
            for value in backupData[key]['ListSelected']:
                tmp.append(value)
            durationCut.append(backupData[key]['CutDuration'])
            results.append(tmp)
    except:
        print('Not enough video to combine')
        exit()
    return results,durationCut
def combineVideo(referenceClips,cutDurationOption,DetailData,featureClips,path_dataset,path_audio,savepath,config,ref_poses,mode):
    finalVideoList = []
    listVideoCombine = []
    dataJson = {}
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
            out = cv2.VideoWriter('/media/anlab/data-2tb/ANLAB_THUY/ModuleDrivingClipStarpower/Temp/TempVideo/'+newName+'.mp4', fourcc, 30, (config['size'], config['size']))
            endx_min_bf = featureClips[os.path.basename(referenceClips[0][0])]['End_xmin']
            endy_min_bf = featureClips[os.path.basename(referenceClips[0][0])]['End_ymin']
            endx_max_bf = featureClips[os.path.basename(referenceClips[0][0])]['End_xmax']
            endy_max_bf = featureClips[os.path.basename(referenceClips[0][0])]['End_ymax']
            paddx_ori = int((config['size']-(endx_max_bf-endx_min_bf))/2)
            paddy_ori = int((config['size']-(endy_max_bf-endy_min_bf))/2)
            idxVideo = 0
            finalVideoList.append('/media/anlab/data-2tb/ANLAB_THUY/ModuleDrivingClipStarpower/Temp/TempVideoAudio/'+newName+'.mp4')
            # tmp = []
            realFrame = 0
            totaltime = 0
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
                frames = video.get(cv2.CAP_PROP_FRAME_COUNT) 
                fps = video.get(cv2.CAP_PROP_FPS) 
                seconds = frames / fps
                totaltime+=seconds
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
            clips = VideoFileClip('/media/anlab/data-2tb/ANLAB_THUY/ModuleDrivingClipStarpower/Temp/TempVideo/'+newName+'.mp4',verbose=False)
            dataJson[str(i)]['OriginalDuration'] = clips.duration*1000
            # dataJson[str(i)]['Stretch_Time'] = np.abs(dataJson[str(i)]['ExpectedDuration'] - dataJson[str(i)]['OriginalDuration'])
            dataJson[str(i)]['OriginalFrameCount'] = sum(video['Frames'] for video in dataJson[str(i)]['Clips'].values())
            # dataJson[str(i)]['Stretch_Frames'] = int(np.abs(dataJson[str(i)]['ExpectedFrames'] - dataJson[str(i)]['OriginalFrameCount']))
            if cutDurationOption[i][1] == 'CUT':
                os.system('ffmpeg -y -i '+' /media/anlab/data-2tb/ANLAB_THUY/ModuleDrivingClipStarpower/Temp/TempVideo/'+newName+'.mp4 ' + '-s 0 -t '+str(expDuration)+' -crf 17 -c:v copy -c:a copy '+'/media/anlab/data-2tb/ANLAB_THUY/ModuleDrivingClipStarpower/Temp/TempVideoSpeed/'+newName+'.mp4')
                dataJson[str(i)]['ModeCut'] = 'CUT'
                # dataJson[str(i)]['Stretch_Time'] = expDuration*1000
            else:
                dataJson[str(i)]['ModeCut'] = 'STRETCH'
                # dataJson[str(i)]['Stretch_Time'] = np.abs(dataJson[str(i)]['ExpectedDuration'] - dataJson[str(i)]['OriginalDuration'])
                new_clip = clips.speedx(final_duration=expDuration)
                new_clip.write_videofile('/media/anlab/data-2tb/ANLAB_THUY/ModuleDrivingClipStarpower/Temp/TempVideoSpeed/'+newName+'.mp4',logger=None)
            new_clip = VideoFileClip('/media/anlab/data-2tb/ANLAB_THUY/ModuleDrivingClipStarpower/Temp/TempVideoSpeed/'+newName+'.mp4',verbose=False)
            # exit()
            # new_clip.write_videofile('/media/anlab/data-2tb/ANLAB_THUY/ModuleDrivingClipStarpower/Temp/TempVideoSpeed/'+newName+'.mp4',logger=None)
            if newName.startswith('NonSpeaking'):
                new_clip.write_videofile('/media/anlab/data-2tb/ANLAB_THUY/ModuleDrivingClipStarpower/Temp/TempVideoAudio/'+newName+'.mp4',logger=None)
            else:
                fileClip = os.path.join('/media/anlab/data-2tb/ANLAB_THUY/ModuleDrivingClipStarpower/Temp/TempVideoSpeed',newName+'.mp4')
                fileAudio = os.path.join(path_audio)
                props = get_audio_properties(fileAudio)
                if props['codec_name'] == 'pcm_s16le':
                    new_clip.write_videofile('/media/anlab/data-2tb/ANLAB_THUY/ModuleDrivingClipStarpower/Temp/TempVideoAudio/'+newName+'.mp4',logger=None)
                else:
                    fileResult = os.path.join('/media/anlab/data-2tb/ANLAB_THUY/ModuleDrivingClipStarpower/Temp/TempVideoAudio',newName+'.mp4')
                    os.system("ffmpeg -y -i "+fileClip+" -i "+fileAudio+" -c copy "+fileResult)
        clips = [VideoFileClip(file) for file in finalVideoList]
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(savepath,fps=30)
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
if __name__ == '__main__':
    main()
