#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import operator

import jieba
import nltk

from scripts.util import MyLib, kw_util

__author__ = 'congzicun'


def tf_idf(lns, idf_value, dic):
    kw_frq = nltk.FreqDist()
    # TF
    for ln in lns:
        kws = MyLib.seg(ln, dic)
        for kw in kws:
            # if 'n' in kw.flag or kw.flag == 'v':
            # kw_frq.inc(kw.word)
         kw_frq.inc(kw)
    #IDF
    tf_idf = {}
    for k, v in kw_frq.items():
        tf_idf[k] = v * idf_value.get(k, 15.5312024064)
    return sorted(tf_idf.iteritems(), key=operator.itemgetter(1), reverse=True)


def pmi(lns, dic, kb):
    sentences = []

    for ln in lns:
        # TODO: 可以调节粒度
        sentences.extend(re.split(ur'[!.?…~;"#:—,]', kw_util.punc_replace(ln)))

    co_occur = nltk.FreqDist()
    kw_dis = nltk.FreqDist()

    for sentence in sentences:
        tokenlst = MyLib.filter_sentiment(MyLib.seg_token(sentence, dic, kb), kb)
        for tokenone in tokenlst:
            kw_dis.inc(tokenone.keyword)
            for tokentwo in tokenlst:
                if tokenone == tokentwo:
                    continue
                co_occur.inc(tokenone.keyword + '$' + tokentwo.keyword)
    pmi = {}
    for kwpair, value in co_occur.items():
        kwone, kwtwo = kwpair.split('$')
        pmi[kwpair] = float(value + 1) / float(kw_dis.get(kwone) * kw_dis.get(kwtwo))
    return pmi

