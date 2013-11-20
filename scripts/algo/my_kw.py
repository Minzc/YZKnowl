#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from scripts.util import kw_util

__author__ = 'congzicun'

import sys
import jieba
import nltk
import math
import operator


def gen_segmnt(infile, usrdic = ""):
    if usrdic != "":
        jieba.load_userdict(usrdic)
    lines = [ ln.strip().decode('utf-8') for ln in open(infile).readlines()]
    for ln in lines:
        print ln.encode('utf-8')
        print "seg:\t",
        for w in jieba.posseg.cut(kw_util.tweet_filter(ln)):
            print w.word.encode('utf-8') +'/'+w.flag,
        print
        print "tag:\t",
        for w in jieba.analyse.extract_tags(kw_util.tweet_filter(ln)):
            print w.encode('utf-8'),
        print

def extract_kw_pair(kw_lst, obj_name='伊利谷粒多'):
    len_kw_lst = len(kw_lst)
    rst_lst = {}
    for i in range(len_kw_lst):
        if kw_lst[i].word.encode('utf-8') == obj_name and i + 1 < len_kw_lst and kw_lst[i+1].flag == 'v':
            if i + 2 < len_kw_lst:
                if kw_lst[i+2].flag == 'a' or kw_lst[i+2].flag == 'n' or kw_lst[i+2].flag == 'v':
                    rst_str = '(' + kw_lst[i+1].word + ',' + kw_lst[i+2].word+')'
                    rst_lst[rst_str] = 2
            rst_lst['(' + kw_lst[i+1].word + ')'] = 1
            return  rst_lst
        elif kw_lst[i].word.encode('utf-8') == obj_name and i + 1 < len_kw_lst and kw_lst[i+1].flag == 'a':
            if i + 2 < len_kw_lst:
                if kw_lst[i+2].flag == 'a' or kw_lst[i+2].flag == 'v' or kw_lst[i+2].flag == 'n':
                    rst_str = '(' + kw_lst[i+1].word + ',' +  kw_lst[i+2].word + ')'
                    rst_lst[rst_str] = 2
            rst_lst['(' + kw_lst[i+1].word + ')'] = 1
            return rst_lst
        elif kw_lst[i].word.encode('utf-8') == obj_name and i + 1 < len_kw_lst and kw_lst[i+1].flag == 'n':
            if i + 2 < len_kw_lst:
                if kw_lst[i+2].flag == 'a' or kw_lst[i+2].flag == 'v':
                    rst_str = '(' + kw_lst[i+1].word + ',' + kw_lst[i+2].word + ')'
                    rst_lst[rst_str] = 2
            rst_lst['('+kw_lst[i+1].word+')'] = 1
        elif kw_lst[i].flag == 'v' and i + 1 < len_kw_lst and kw_lst[i+1].word.encode('utf-8') == obj_name:
            rst_lst['('+kw_lst[i].word+')'] = 1
    return rst_lst

def gen_combine_keywords(infile, usrdic = '', stopdic = 'stopwords.txt'):
    stop_dic = {}
    if usrdic != '':
        jieba.load_userdict(usrdic)
    if stopdic != '':
        print 'load stop keywords'.encode('utf-8')
        stop_dic = {ln.strip() for ln in open(stopdic).readlines()}
    lines = { ln.strip() for ln in open(infile).readlines() }
    kwFrqDist = nltk.FreqDist()
    for ln in lines:
#        print 'ln:\t',ln
        kws = list(jieba.posseg.cut(kw_util.tweet_filter(ln)))
        kws_lst = combine_kw(kws)
        final_rst = []
        for w in kws_lst:
            if w.flag == 'r' or w.flag == 'uj':
                continue
            elif w.word.encode('utf-8') in stop_dic:
                continue
            final_rst.append(w)

#        for i in range(len(kws_lst)):
#            print kws_lst[i].word.encode('utf-8')+'/'+kws_lst[i].flag.encode('utf-8'),
#        print
        rst_pair = extract_kw_pair(final_rst)
        if len(rst_pair) != 0:
            print 'ln:\t', ln
            for i in range(len(final_rst)):
                print final_rst[i].word.encode('utf-8')+'/'+final_rst[i].flag.encode('utf-8'),
            print
            print "ext:\t",
            for kw,score in rst_pair.items():
                print kw.encode('utf-8')+'/'+str(score),
            print '\n'
#        else:
#            print "tag:\t",
#            print ",".join(list(jieba.analyse.extract_tags(kw_util.tweet_filter(ln)))).encode('utf-8')


#        print "tag:\t",
#        for w in jieba.analyse.extract_tags(kw_util.tweet_filter(ln)):
#            print w.encode('utf-8'),
#        print
#        for w in kws_lst:
#            kwFrqDist.inc(w.word)

    for w,count in kwFrqDist.items():
        if len(w) > 1 and count > 2:
            print w.encode('utf-8')+','+str(count)


def add_kw_pair(kw1, kw2,kwpair_mrtx={}):
    if not kwpair_mrtx.has_key(kw1):
        kwpair_mrtx[kw1] = {}
    if not kwpair_mrtx.has_key(kw2):
        kwpair_mrtx[kw2] = {}
    kwpair_mrtx[kw1].setdefault(kw2,0)
    kwpair_mrtx[kw2].setdefault(kw1,0)
    kwpair_mrtx[kw2][kw1] += 1
    kwpair_mrtx[kw1][kw2] += 1
    return kwpair_mrtx

def gen_bias_test(infile, usrdic = ""):
    if usrdic != "":
        jieba.load_userdict(usrdic)
    lines = [ ln.strip().decode('utf-8') for ln in open(infile).readlines()]
    kwdist = nltk.FreqDist()
    kwpair_mtr = {}
    for ln in lines:
        sublns = kw_util.tweet_filter(ln).split(' ')
        for subln in sublns:
            kws = jieba.cut(subln)
            kws_arr = []
            for kw in kws:
                if kw != ' ':
                    kws_arr.append(kw)

            for i in range(len(kws_arr)):
                kwdist.inc(kws_arr[i])
                for j in range(i+1,len(kws_arr)):
                    kwpair_mtr = add_kw_pair(kws_arr[i],kws_arr[j],kwpair_mtr)

    kfdist = {}

    for kw,count in kwdist.items():
        kfscore = 0
        for fqkw,fqcount in kwdist.items()[:100]:
            if kw != fqkw:
                if(kwpair_mtr.has_key(kw)):
                    kfscore += math.pow(kwpair_mtr[kw].get(fqkw,0)-fqcount*count,2)/(fqcount*count)
        kfdist[kw] = kfscore
    sorted_x = sorted(kfdist.iteritems(), key=operator.itemgetter(1))
    for k,scr in sorted_x:
        print k,scr



def gen_belong(infile, usrdic = '',objname = '伊利谷粒多'):
    if usrdic != "":
        jieba.load_userdict(usrdic)
    lines = [ ln.strip().decode('utf-8') for ln in open(infile).readlines()]
    belong_dic = {}
    for ln in lines:
        kws = list(jieba.cut(kw_util.tweet_filter(ln)))
        for i in range(len(kws)):
            if kws[i].encode('utf-8') == objname and i+2 < len(kws) and kws[i+1].encode('utf-8') == '的':
                belong_dic[kws[i+2]] = 1
    for kw in belong_dic.keys():
        if len(kw) > 1:
            print kw.encode('utf-8')

def combine_kw(kw_w_lst, obj_name):
    combined_w = []
    kw_combined = False
    for i in range(len(kw_w_lst)-1):
        if kw_w_lst[i].word.encode('utf-8') == obj_name:
            kw_w_lst[i].flag = 'obj'

        if kw_combined:
            kw_combined = False
            continue
        word = kw_w_lst[i].word
        flag = kw_w_lst[i].flag

        if len(kw_w_lst[i].word) == 1:
            kw_combined = True
            if 'v' in kw_w_lst[i].flag and 'ul' in kw_w_lst[i+1].flag:
                word = kw_w_lst[i].word + kw_w_lst[i+1].word
                flag = 'a'
            elif 'a' in kw_w_lst[i].flag and 'ul' in kw_w_lst[i+1].flag:
                word = kw_w_lst[i].word + kw_w_lst[i+1].word
                flag = 'a'
            elif 'v' in kw_w_lst[i].flag and 'v' in kw_w_lst[i+1].flag:
                word = kw_w_lst[i].word + kw_w_lst[i+1].word
                flag = 'v'
            elif 'a' in kw_w_lst[i].flag and 'v' in kw_w_lst[i+1].flag:
                word = kw_w_lst[i].word + kw_w_lst[i+1].word
                flag = 'v'
            elif 'd' in kw_w_lst[i].flag and 'v' in kw_w_lst[i+1].flag:
                word = kw_w_lst[i].word + kw_w_lst[i+1].word
                flag = 'a'
            elif 'n' in kw_w_lst[i].flag and 'ul' in kw_w_lst[i+1].flag:
                word = kw_w_lst[i].word + kw_w_lst[i+1].word
                flag = 'a'
            elif 'z' in kw_w_lst[i].flag and 'v' in kw_w_lst[i+1].flag:
                word = kw_w_lst[i].word + kw_w_lst[i+1].word
                flag = 'a'
            elif 'd' in kw_w_lst[i].flag and 'n' in kw_w_lst[i+1].flag:
                word = kw_w_lst[i].word + kw_w_lst[i+1].word
                flag = 'a'
            else:
                kw_combined = False
        kw_w_lst[i].word = word
        kw_w_lst[i].flag = flag
        combined_w.append(kw_w_lst[i])

    if not kw_combined:
        combined_w.append(kw_w_lst[len(kw_w_lst)-1])
    return combined_w

def gen_obj_na(infile, usrdic = '',objname = "伊利谷粒多"):
    if usrdic != "":
        jieba.load_userdict(usrdic)
    lines = [ ln.strip().decode('utf-8') for ln in open(infile).readlines()]
    for ln in lines:
        kws = list(jieba.posseg.cut(kw_util.tweet_filter(ln)))
        find_obj = False
        kws_lst = combine_kw(kws)
        print 'line:\t' + ln.encode('utf-8') + '\n'

        print "tag:\t",
        for w in jieba.analyse.extract_tags(kw_util.tweet_filter(ln)):
            print w.encode('utf-8'),
        print

        print 'seg:\t',
        for w in kws_lst:
            print w.word.encode('utf-8')+'/'+w.flag.encode('utf-8'),
        print

        for w in kws_lst:
            if w.word.encode('utf-8') == objname:
                find_obj = True
            if find_obj:
                if 'n' in w.flag :
                    print w.word.encode('utf-8')+"/n",
                if 'a' in w.flag:
                    print w.word.encode('utf-8')+"/a",
        print



#        for i in range(len(kws)):
#            if kws[i].encode('utf-8') == objname:
#                find_obj = True
#            if find_obj:
#                if 'n' in kws_f[i] :
#                    print kws[i].encode('utf-8')+"/n",
#                if 'a' in kws_f[i]:
#                    print kws[i].encode('utf-8')+"/a",
#        print
#
#        print "tag:\t",
#        for w in jieba.analyse.extract_tags(kw_util.tweet_filter(ln)):
#            print w.encode('utf-8'),
#        print



if __name__=='__main__':
    if len(sys.argv) < 3:
        print """Usage: python my_kw.py <cmd> <input_file>
         <cmd> = [segmnt]
         segmnt <input_file> <usr_dic>
         bias <input_file> <usr_dic>
              """

    elif sys.argv[1] == "segmnt":
        if len(sys.argv) == 3:
            gen_segmnt(sys.argv[2])
        else:
            gen_segmnt(sys.argv[2],sys.argv[3])
    elif sys.argv[1] == "bias":
        if len(sys.argv) == 3:
            gen_bias_test(sys.argv[2])
        else:
            gen_bias_test(sys.argv[2],sys.argv[3])
    elif sys.argv[1] == 'cmbkw':
        if len(sys.argv) == 3:
            gen_combine_keywords(sys.argv[2])
        else:
            gen_combine_keywords(sys.argv[2],sys.argv[3])
    elif sys.argv[1] == 'objna':
        if len(sys.argv) == 3:
            gen_obj_na(sys.argv[2])
        else:
            gen_obj_na(sys.argv[2],sys.argv[3])
    elif sys.argv[1] == 'belong':
        gen_belong(sys.argv[2],sys.argv[3])
