#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'congzicun'
import jieba
from lsh import LSHCache


def lsh():
    lns = [ln.decode('utf-8') for ln in open('clean_data').readlines()]
    cache = LSHCache()
    docs = []
    for ln in lns:
        word_dic = []
        for wd in list(jieba.cut(ln)):
            # if len(wd) > 1:
            word_dic.append(wd)
        docs.append(' '.join(word_dic))
    dups = {}

    for i, doc in enumerate(docs):
        dups[i] = cache.insert(doc.split(), i)
    for i, duplist in dups.items():
        if duplist:
            print 'orig [%d]: %s' % (i, docs[i])
            for dup in duplist:
                print'\tdup : [%d] %s' % (dup, docs[dup])
        else:
            print 'no dups found for doc [%d] : %s' % (i, docs[i])

if  __name__ == '__main__':
    lsh()

