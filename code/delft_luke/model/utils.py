#!/usr/bin/python
# -*- coding: utf-8 -*-
import json

def load_data(data_file):
    data = list()
    with open(data_file) as f:
        for line in f.readlines():
            ex = json.loads(line)
            data.append(ex)
    return data

def dump_data_add(data_file,result):
    # with open(data_file, mode='a+', encoding='utf-8') as feedsjson:
    with open(data_file, "a+", encoding='utf-8') as f:
        line = json.dumps(result, ensure_ascii=False)
        f.write(line+'\n')
        f.close()
#
# def dump_data_write(data_file,result):
#     # with open(data_file, mode='a+', encoding='utf-8') as feedsjson:
#     with open(data_file, "w", encoding='utf-8') as f:
#         for item in result:
#             line = json.dumps(item, ensure_ascii=False)
#             f.write(line+'\n')
#         f.close()

