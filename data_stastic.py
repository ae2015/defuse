#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   data_stastic.py
@Time    :   2024/10/06 11:20:18
@Author  :   Zhiyuan Peng
@Version :   1.0
@Contact :   zpeng@scu.edu
@License :   (C)Copyright 2020-2021, Zhiyuan Peng/SCU
@Desc    :   None
'''
import os
import pandas as pd
cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data/experiments/llmq-gpt-4o-mini/llmr-gpt-3.5/docp-dt-z-1/20-toy-concise_ori_refine_13to18_example_3iter_6facts/sport')
qrc = pd.read_csv(os.path.join(data_dir, 'qrc_out.csv'))

non_confused = []
confused =[]
for index, row in qrc.iterrows():
    if row['is_confusing'] == 'no':
        non_confused.append(len(row['question'].split()))
    else:
        confused.append(len(row['question'].split()))

non_average = sum(non_confused) / len(non_confused)
confused_average = sum(confused) / len(confused)
print(f'Non-confused average length: {non_average}, Max: {max(non_confused)}, Min: {min(non_confused)}')
print(f'Confused average length: {confused_average}, Max: {max(confused)}, Min: {min(confused)}')
