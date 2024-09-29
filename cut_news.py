#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   process_news.py
@Time    :   2024/09/26 09:54:06
@Author  :   Zhiyuan Peng
@Version :   1.0
@Contact :   zpeng@scu.edu
@License :   (C)Copyright 2020-2021, zpeng@scu
@Desc    :   read news dataset and cut the content to 300 words
'''
import os
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from os.path import join
# Ensure the necessary NLTK data files are downloaded
nltk.download('punkt')
cwd = os.getcwd()
base_dir = join(cwd, 'data', 'raw', 'News1k2024')
output_dir = join(cwd, 'data', 'processed', 'News1k2024_300')

def process_json_file(file_path, output_dir):
    with open(file_path, 'r') as f:
        data = json.load(f)
    content = data.get('content', '')
    token_count = data.get('token_count', 0)
    if token_count < 300:
        new_content_str = content
        token_count = token_count
    else:
        sentences = sent_tokenize(content)
        new_content = []
        token_count = 0
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            token_count += len(tokens)
            new_content.append(sentence)
            if token_count > 300:
                break
        new_content_str = ' '.join(new_content)
    data['content'] = new_content_str
    data['token_count'] = token_count
    del data['summary']
    output_file_path = os.path.join(output_dir, os.path.basename(file_path))
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def process_folders(base_dir):
    parent_dir = os.path.dirname(base_dir)
    for root, dirs, files in os.walk(base_dir):
        topic = os.path.basename(root)
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                output_dir = os.path.join(parent_dir, 'News1k2024-300', topic)
                os.makedirs(output_dir, exist_ok=True)
                process_json_file(file_path, output_dir)

if __name__ == "__main__":
    process_folders(base_dir)