#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
__author__ = 'congzicun'
import jieba
import kw_util
import MyLib
import nltk
import operator
import sys
import re
import FreqBase
STOP_DIC = 'stopwords.txt'
lns = [ln for ln in open('testCodeCrrct.txt').readlines()]




def test_load_knw_base():
    class_entity,synonym,sent_dic = FreqBase.load_knw_base()
    print class_entity['物超所值'][0]
    print '物超所值' in sent_dic

def test_gen_model_get_kws_knwbase():
    class_entity,synonym,sent_dic = FreqBase.load_knw_base()
    for ln in lns:
        print ln
        kws = FreqBase.gen_model_get_kws_knwbase(ln,synonym,'伊利谷粒多')
        for kw in kws:
            print kw.token.word,kw.wrdpos


if __name__ == '__main__':
#    test_gen_model_get_kws_knwbase()
    test_load_knw_base()
