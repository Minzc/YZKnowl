#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from scripts.algo import FreqBase
from scripts.util import kw_util

__author__ = 'congzicun'
import re

STOP_DIC = 'stopwords.txt'
lns = [ln.decode('utf-8').lower() for ln in open('testCodeCrrct.txt').readlines()]


def test_gen_model():
    FreqBase.DEBUG = True
    FreqBase.gen_model('testCodeCrrct.txt')


def test_load_knw_base():
    class_entity,synonym,sent_dic = FreqBase.load_knw_base()
#    print class_entity['物超所值'][0]
    print u'冰冰' in synonym.keys()
    print len('好'.decode('utf-8'))


def test_gen_model_get_kws_knwbase():
    class_entity,synonym,sent_dic = FreqBase.load_knw_base()
    for ln in lns:
        print ln
        kws,obj_poss = FreqBase.seg_ln(ln,synonym,u'伊利谷粒多')
        for kw in kws:
            print kw.token.word,kw.wrd_strt_pos,kw.wrd_end_pos,kw.sntnc
            for kw_2 in kws:
                if kw != kw_2:
                    print kw_2.token.word
                    print FreqBase.decide_dis_feature_type(kw,kw_2)
def test_load_model():
    FreqBase.load_mdl('model.txt')


def test_classify():
    FreqBase.DEBUG = True
    FreqBase.OBJ_NAME = u'伊利谷粒多'
    FreqBase.class_new('testCodeCrrct.txt', u'恒大', 'model.txt')


def format_know_base():
    lns = [ln.decode('utf-8').strip().lower()
           for ln in open('untitled.txt').readlines() if not ln.startswith('#')]
    for ln in lns:
        ln_seg = ln.split('\t')
        print ln_seg[0].strip().encode('utf-8')+'\t'+ln_seg[1].strip().encode('utf-8')+'\t'+ln_seg[2].strip().encode('utf-8')


def testPuncReplace():
    ln = 'good！a！b。c，d？e（f）)k)g。。h～g；h“j”i'
    ln =  kw_util.punc_replace(ln.decode('utf-8'))
    print re.sub(r'\(.*?\)','',ln)
    print re.split(ur'[!.?…;"]',ln)

def testTweetFilter():
    for ln in lns:
        print kw_util.tweet_filter(ln)

if __name__ == '__main__':
#    test_gen_model_get_kws_knwbase()
#    test_gen_model()
#    test_load_knw_base()
#    test_load_model()
    test_classify()
#    testPuncReplace()
#    testTweetFilter()
