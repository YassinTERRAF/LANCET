#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhu Wenjing
# Date: 2022-03-07
# E-mail: zhuwenjing02@duxiaoman.com

import glob
import json
import os

# Updated LABEL dictionary to merge 'hap' and 'exc' into '07'
LABEL = {
    'neu': '01',  # 'neutral'
    'fru': '02',  # 'calm'
    'hap': '07',  # 'happy' merged into 'excited'
    'sad': '04',  # 'sad'
    'ang': '05',  # 'angry'
    'fea': '06',  # 'fearful'
    'exc': '07',  # 'excited'
    'sur': '08',  # 'surprised'
    'xxx': '09'   # 'other'
}

IEMOCAP = ''

PATH_TXT = glob.glob(IEMOCAP + "/*/dialog/EmoEvaluation/S*.txt")
PATH_WAV = glob.glob(IEMOCAP + "/*/sentences/wav/*/S*.wav")

PAIR = {}

def getPair():
    for path in PATH_TXT:
        with open(path, 'r') as f:
            fr = f.read().split("\t")
            for i in range(len(fr)):
                if fr[i] in LABEL:
                    PAIR[fr[i - 1]] = LABEL[fr[i]]  # Map directly using updated LABEL

def rename():
    for i in PATH_WAV:
        for j in PAIR:
            if os.path.basename(i)[:-4] == j:
                k = j.split('_')
                if len(k) == 3:
                    name = os.path.dirname(i) + '/' + k[0] + '-' + k[1] + '-' + PAIR[j] + '-01-' + k[2] + '.wav'
                    os.rename(src=i, dst=name)
                elif len(k) == 4:
                    name = os.path.dirname(i) + '/' + k[0] + '-' + k[1] + '-' + PAIR[j] + '-01-' + k[2] + '_' + k[3] + '.wav'
                    os.rename(src=i, dst=name)

if __name__ == '__main__':
    pairPath = IEMOCAP + "/pair.json"
    if os.path.exists(pairPath):
        with open(pairPath, 'r') as f:
            PAIR = json.load(f)
    else:
        getPair()
        with open(pairPath, 'w') as f:
            json.dump(obj=PAIR, fp=f)
    print('Starting...')
    rename()