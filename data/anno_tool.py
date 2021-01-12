# import pandas as pd
# from mne.io import concatenate_raws, read_raw_edf

dic = {'?':5, 'W':0, '1':1, '2':2, '3':3, '4':3, 'R':4}
dic2 = {'5':4, '0':0, '1':1, '2':2, '3':3, '4':3, '9':0}


def saf(path = "D:\\SLEEP\\MASS\\SS1/01-01-0053.saf"):
    anno = []
    with open(path,'rb') as f:
        lines = f.readlines()
        string = lines[0].decode()

        i = string.find('Sleep stage')
        while i!=-1:
            string = string[i+12:]
            anno.append (string[0])
            i = string.find('Sleep stage')
    
    anno = [dic[a] for a in anno]
    return anno

def txt(path = "D:\\SLEEP\\MASS\\Base_annotations\\SS1/01-01-0001 Base_annotations.txt"):
    with open(path, 'r') as f:
        lines = f.readlines()
    assert lines[0].strip() == 'Onset,Duration,Annotation'
    anno = []
    startTime = -1
    for line in lines[1:]:
        line = line.strip()
        if line != []:
            onset, duration, ann = line.split(',')
            if startTime < 0:
                startTime = float(onset)
            anno.append(dic[ann[12]])
    return anno, startTime

def xml(path = "D:/SLEEP/shhs/polysomnography/annotations-events-profusion/shhs1/shhs1-200101-profusion.xml"):
    import xml.etree.ElementTree as ET
    anno = []
    tree = ET.ElementTree(file= path)
    root = tree.getroot()
    sleepStages = root.find('SleepStages')
    stages = sleepStages.findall('SleepStage')
    for stage in stages:
        s = stage.text
        anno.append(dic2[s])
    return anno 

def saf2txt_mass(path = "D:\\SLEEP\\MASS\\SS3/01-03-0043.saf"):
    anno = []
    with open(path,'rb') as f:
        lines = f.readlines()
        string = lines[0].decode()

        i = string.find('Sleep stage')
        while i!=-1:
            # got a sleep stage
            j = i
            while j>0 and string[j]!='\x00': j-=1
            stageInfo = string[j+1:i+13]
            s = stageInfo[:stageInfo.find('\x15')] + ',' + stageInfo[stageInfo.find('\x15')+1 : stageInfo.find('\x14')] + ',' + stageInfo[stageInfo.find('\x14')+1:]
            with open('saf2txt.txt', 'a') as ff:
                ff.write(s + '\n')
            string = string[i+13:]
            i = string.find('Sleep stage')
    

if __name__ == "__main__":
    saf2txt_mass()
    pass
        
    
