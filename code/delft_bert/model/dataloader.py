#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import urllib
import requests
import time
from transformers import pipeline
from utils import load_data, dump_data

def init_model():
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli") #device=0 for gpu

    #sequence_to_classify ="Cancer is a group of diseases involving abnormal cell growth with the potential to invade or spread to other parts of the body."
    candidate_labels = ['Person', 'Location', 'Organization', 'Product',"Art","Event","Building",'Others']
    per_labels=['doctor','actor','engineer','architect','monarch','artist','musician','athlete','politician','author','religious leader','coach','director','soldier','terrorist']
    org_labels=['terrorist organization',"airline","government agency","company","government","educational institution","political party","fraternity sorority","educational department","sports league", "military","sports team","news agency"]
    loc_labels=["body of water","city","island","country","mountain", "county","glacier","province","astral body", "railway","cemetery","road","park","bridge"]
    product_labels=["camera","engine","mobile phone","airplane","computer","car","software","ship","game","spacecraft","instrument","train","weapon"]
    art_labels=["written work","film","newspaper","play","music"]
    event_labels=["military conflict","attack","natural disaster","election","sports event","protest","terrorist attack"]
    building_labels=["airport","dam","hospital","hotel","library","power station","restaurant","sports facility", "theater"]
    other_labels=["time","color","award","educational degree","title","law","ethnicity","language","religion","god","chemical thing","biology thing","medical treatment","disease","symptom","drug","body part","living thing","animal","food","website","broadcast network","broadcast program","currency","stock exchange","algorithm","programming language","transit system","transit line"]
    labels_list=per_labels+org_labels+loc_labels+product_labels+art_labels+event_labels+building_labels+other_labels
    return classifier,labels_list

def named_entity_classifier(classifier,labels_list,sequence_to_classify,multi_class=False):
    similarity_dict=load_data("label_similarity.json")[0]
    threshold=0.9
    time_start=time.time()
    res=classifier(sequence_to_classify, labels_list,multi_class=multi_class)
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
    #print(res)
    # print(res["labels"][0])
    if multi_class: # multiple labels
        num=0
        for score in res["scores"]:
            if score>threshold:
               num+=1
        i=0
        j=0
        labels=[]
        if num<5:
            num=5
        #adjust_num=num
        for i in range(num):
            #print(res["labels"][:num])
            #print(similarity_dict,type(similarity_dict))
            print(similarity_dict[res["labels"][i]])
            if similarity_dict[res["labels"][i]] in res["labels"][:num]:
                labels.append(res["labels"][i])
            else:
                labels.append(res["labels"][i])
                labels.append(similarity_dict[res["labels"][i]])

                #adjust_num+=1
        # print(adjust_num)
        # while j<adjust_num:
        #     labels.append(res["labels"][j])
        #     j+=1
        print(len(labels))
        print(labels)
        return labels #return list
    else: # single label
        return res["labels"][0] #return str




def label_classification(file_name):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    dataset = load_data(file_name)
    classifier,labels_list=init_model()
    modified=[]
    # for each question line
    for question in dataset:
        for sub_ques_et in question['q_et']:
            # 0. question labels
            ques = sub_ques_et["text"]
            # print(type(sub_ques_et)) #dictionary
            sub_ques_et["label"]= named_entity_classifier(classifier,labels_list,ques,multi_class=True)
            # 1. question entity labels
            sub_question = sub_ques_et['entity']
            for entity in sub_question:
                # dict: 1'et'  2 first  sentence
                # print(entity['et'],entity['first_sent'])
                # print(type(entity)) #dictionary
                entity["label"]=named_entity_classifier(classifier,labels_list,entity['first_sent'])
                # label_classifier(entity['first_sent'])
                # break
        # 2. candidate entity labels
        # 2.1 positive candidate entity
        sub_cand_pos_et = question['pos_et']  # dictionary
        sub_cand_pos_et["label"]=named_entity_classifier(classifier,labels_list,sub_cand_pos_et['first_sent'],multi_class=False)
        # print(sub_cand_pos_et['et'],sub_cand_pos_et['first_sent'])
        # 2.2 negative candidate entities
        sub_cand_neg_ets = question['neg_ets']
        for sub_cand_neg_et in sub_cand_neg_ets:
            sub_cand_neg_et["label"]=named_entity_classifier(classifier,labels_list,sub_cand_neg_et['first_sent'])
        print(question)
        print("#######################################")
        modified.append(question)
    dump_data(file_name+"_labeled",modified)
    # data=data[1]
    # print((data[1]["pos_et"]))
    # id
    # text
    # q_et       #et []
    # first_sent
    ##score
    ##id
    # pos_et     #et
    # evidence
    #
    # neg_ets    #et []
    # first_sent
    # evidence


if __name__ == '__main__':
    #print("你好")
    file_name="../data/trivia_test_graph.json"
    dataset = load_data(file_name)
    print(dataset)
    label_classification(file_name)
    #print("noob")

# def label_classifier(sentence):
#     sentence=nlp("Cancer is a group of diseases involving abnormal cell growth with the potential to invade or spread to other parts of the body.")
#     sentence=nlp(' '.join([str(t) for t in sentence if not t.is_stop]))
#     label="disease"
#     label=nlp(label)
#     print(label, "<->", sentence, label.similarity(sentence))
#     # search_doc = nlp("This was very strange argument between american and british person")
#     # main_doc = nlp("He was from Japan, but a true English gentleman in my eyes, and another one of the reasons as to why I liked going to school.")
#     #
#     # search_doc_no_stop_words = nlp(' '.join([str(t) for t in search_doc if not t.is_stop]))
#     # main_doc_no_stop_words = nlp(' '.join([str(t) for t in main_doc if not t.is_stop]))
#     #
#     # print(search_doc_no_stop_words.similarity(main_doc_no_stop_words))