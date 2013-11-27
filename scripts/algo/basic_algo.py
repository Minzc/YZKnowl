#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import operator

import jieba
import nltk

from scripts.util import MyLib, file_loader, kw_util
import jieba

__author__ = 'congzicun'


def tf_idf(lns, idf_value):
    kw_frq = nltk.FreqDist()
    # TF
    for ln in lns:
        kws = list(jieba.posseg.cut(ln))
        for kw in kws:
            if 'n' in kw.flag:
                kw_frq.inc(kw.word)
    #IDF
    tf_idf = {}
    for k, v in kw_frq.items():
        tf_idf[k] = v * idf_value.get(k, 15.5312024064)
    return sorted(tf_idf.iteritems(), key=operator.itemgetter(1), reverse=True)


def pmi(lns, dic):
    sentences = []

    for ln in lns:
        # TODO: 可以调节粒度
        sentences.extend(re.split(ur'[!.?…~;"#,]', kw_util.punc_replace(ln)))

    co_occur = nltk.FreqDist()
    kw_dis = nltk.FreqDist()
    total_tweets = len(sentences)

    for sentence in sentences:
        kws = MyLib.seg(sentence, dic)
        for kwone in kws:
            kw_dis.inc(kwone)
            for kwtwo in kws:
                if kwone == kwtwo:
                    continue
                co_occur.inc(kwone + '$' + kwtwo)
    pmi = {}
    for kwpair, value in co_occur.items():
        kwone, kwtwo = kwpair.split('$')
        pmi[kwpair] = float(total_tweets * value) / float(kw_dis.get(kwone) * kw_dis.get(kwtwo))
    return pmi

