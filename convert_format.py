#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   convert_format.py
@Time    :   2024/09/26 23:09:23
@Author  :   Zhiyuan PEng
@Version :   1.0
@Contact :   zpeng@scu.edu
@License :   (C)Copyright 2020-2021, zpeng@scu
@Desc    :   write all the news under the same category into a single file
'''

import os
import json
import pandas as pd
from os.path import join
cwd = os.getcwd()
base_dir = join(cwd, 'data', 'raw', 'News1k2024-300')
output_dir = join(cwd, 'data', 'processed', 'News1k2024-300')

def convert_format(base_dir, output_dir, sample_size):
    for root, dirs, files in os.walk(base_dir):
        df = pd.DataFrame(columns=['doc_id', 'source', 'document'])
        topic = os.path.basename(root)
        files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        i = 0
        for file in files:
            i += 1
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
            doc_id = file
            url = data.get('url', '')
            title = data.get('title', '')
            content = data.get('content', '')
            document = f"{title} {content}"
            df.loc[len(df)] = [doc_id, url, document]
            if i == sample_size:
                break
        if i == 0:
            continue
        output_folder = join(output_dir, f"{topic}/{i}")
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, "docs_in.csv")
        df.to_csv(output_file, index=False)

            

if __name__ == "__main__":
    convert_format(base_dir, output_dir, sample_size=-1)
