#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'congzicun'
import redis
import nltk


def trans_kv():
    lns = [ln.decode('utf-8').strip() for ln in open('knowl-base.txt').readlines() if
           not ln.startswith('#') and len(ln) > 1]

    degree = set()
    sentiment = set()

    for ln in lns:
        en, inst, clss = ln.split('\t')
        if u'程度' in clss:
            degree.add(en)
        if u'正面' in clss or u'负面' in clss:
            sentiment.add(en)
            instances = inst.split('|')
            for i in instances:
                if len(i.strip()) > 0:
                    sentiment.add(i)

    combinekw = {k1 + k2: k2 for k1 in degree for k2 in sentiment}
    combinekw.update({k1 + u'的': k1 for k1 in sentiment})

    r = redis.StrictRedis(host='192.168.1.123', port=6379)
    file_reader = open('global_cooccur.txt')
    kw_dist = nltk.FreqDist()
    key = 'global:mdl'
    for ln in file_reader:
        ln = ln.decode('utf-8').strip()
        pair, num = ln.split('\t')
        k1, k2 = pair.split('$')
        k1 = combinekw.get(k1, k1)
        kw_dist.inc(k1)
        pair = k1 + '$' + k2
        r.hset(key, float(num), pair)
        print (pair + '$' + str(num)).encode('utf-8')

    for k, v in kw_dist.items():
        r.hset(key, float(v), k)
