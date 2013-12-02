#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
from scripts.algo import basic_algo
from scripts.util import kw_util, file_loader

__author__ = 'congzicun'


def extr_kw(lns, kb, obj_name, dic):
    idf = file_loader.load_idf()
    phrases = set()
    for ln in lns:
        ln = kw_util.tweet_filter(re.sub('#(.+?)#', ' ', ln))
        phrases |= set(ln.split(' '))

    tf_idf_srtd = basic_algo.tf_idf(phrases, idf, dic)
    added_ftr_num = min(len(tf_idf_srtd), 20)
    print 'LOCAL FEATURE:'
    counter = 0
    for k, v in tf_idf_srtd:
        if k not in kb.instances and k not in kb.sentiments and k not in kb.stop_dic \
                and len(k) > 1 and not k.encode('utf-8').isalnum() and k not in obj_name:
            counter += 1
            if counter == added_ftr_num:
                break
            print k.encode('utf-8'), v
            kb.instances[k] = k
            kb.features[k] = kb.FEATURE
    return {k: v for k, v in tf_idf_srtd}


def cal_pmi(lns, dic, kb):
    return basic_algo.pmi(lns, dic, kb)