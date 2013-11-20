# -*- coding: utf-8 -*-
total_senti = [ln.decode('utf-8').strip() for ln in open('dictionary/total_senti.txt').readlines()]

valid_senti = [ln.decode('utf-8').strip().split('\t')[0] for ln in open('dictionary/sentiment_dict.txt').readlines() if not ln.startswith('#') and len(ln) > 0]

all_senti = set()

for ln in total_senti:
    entity,instances,classes = ln.split('\t')
    instances = instances.split('|')
    issenti = False
    if entity in valid_senti:
        issenti = True
    for instance in instances:
        if instance in valid_senti:
            issenti = True
            break
    if issenti:
        print ln.encode('utf-8')
