#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
__author__ = 'congzicun'
import jieba
import kw_util
import nltk

FILE_NAME = 'xiaomi/xiaomi_train_data.txt'
lns = [ln.decode('utf-8') for ln in open(FILE_NAME).readlines()]
stop_dic = [ln.strip().decode('utf-8') for ln in open('dictionary/stopwords.txt').readlines()]
jieba.load_userdict('dictionary/new_words.txt')
def get_top_kw():
    kw_dist = nltk.FreqDist()
    for ln in lns:
        ln = kw_util.tweet_filter(kw_util.punc_replace(ln))
        kws = list(jieba.posseg.cut(ln))
        for kw in kws:
            if kw.word not in stop_dic and len(kw.word) > 1:
                kw_dist.inc(kw.word.lower()+'$'+kw.flag)
    for kw,count in kw_dist.items():
        if len(kw) > 1:
            print kw.encode('utf-8'),count
if __name__ == '__main__':
    get_top_kw()

