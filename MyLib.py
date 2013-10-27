#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'congzicun'
import jieba
import nltk
import kw_util


def combine_kw(kw_w_lst, obj_name):
    combined_w = []
    kw_combined = False
    for i in range(len(kw_w_lst)-1,0,-1):
        if kw_w_lst[i].word.encode('utf-8') == obj_name:
            kw_w_lst[i].flag = 'obj'
        if kw_combined:
            kw_combined = False
            continue
        word = kw_w_lst[i].word
        flag = kw_w_lst[i].flag

        if len(kw_w_lst[i-1].word) == 1 and len(kw_w_lst[i].word) == 1:
            kw_combined = True
            if 'v' in kw_w_lst[i-1].flag and 'ul' in kw_w_lst[i].flag :
                word = kw_w_lst[i-1].word + kw_w_lst[i].word
                flag = 'a'
            elif 'a' in kw_w_lst[i-1].flag and 'ul' in kw_w_lst[i].flag:
                word = kw_w_lst[i-1].word + kw_w_lst[i].word
                flag = 'a'
            elif 'v' in kw_w_lst[i-1].flag and 'v' in kw_w_lst[i].flag:
                word = kw_w_lst[i-1].word + kw_w_lst[i].word
                flag = 'v'
            elif 'a' in kw_w_lst[i-1].flag and 'v' in kw_w_lst[i].flag:
                word = kw_w_lst[i-1].word + kw_w_lst[i].word
                flag = 'v'
            elif 'd' in kw_w_lst[i-1].flag and 'v' in kw_w_lst[i].flag :
                word = kw_w_lst[i-1].word + kw_w_lst[i].word
                flag = 'v'
            elif 'n' in kw_w_lst[i-1].flag and 'ul' in kw_w_lst[i].flag:
                word = kw_w_lst[i-1].word + kw_w_lst[i].word
                flag = 'a'
            elif 'z' in kw_w_lst[i-1].flag and 'v' in kw_w_lst[i].flag:
                word = kw_w_lst[i-1].word + kw_w_lst[i].word
                flag = 'a'
            elif 'd' in kw_w_lst[i-1].flag and 'n' in kw_w_lst[i].flag:
                word = kw_w_lst[i-1].word + kw_w_lst[i].word
                flag = 'a'
            elif 'd' in kw_w_lst[i-1].flag and 'a' in kw_w_lst[i].flag:
                word = kw_w_lst[i-1].word + kw_w_lst[i].word
                flag = 'a'
            elif 'a' in kw_w_lst[i-1].flag and 'q' in kw_w_lst[i].flag:
                word = kw_w_lst[i-1].word + kw_w_lst[i].word
                flag = 'a'
            else:
                kw_combined = False
        kw_w_lst[i].word = word
        kw_w_lst[i].flag = flag
        combined_w.append(kw_w_lst[i])

    if not kw_combined:
        if kw_w_lst[0].word.encode('utf-8') == obj_name:
            kw_w_lst[0].flag = 'obj'
        combined_w.append(kw_w_lst[0])
    return reversed(combined_w)

def seg_and_filter(ln,obj_name,stop_dic):
    kws =  [kw for kw in  combine_kw(list( jieba.posseg.cut(ln) ),obj_name)\
           if not(len(kw.word) == 1
                   or kw.word.encode('utf-8') in stop_dic
                   or 'f' in kw.flag
                   or 'q' in kw.flag
                   or 'm' in kw.flag
                   or 't' in kw.flag
                   or 'd' in kw.flag
                   or 'o' == kw.flag
                   or 'nz' == kw.flag
        )]
    for i in range(len(kws)):
        kws[i].word = kws[i].word.encode('utf-8')
    return kws


def create_and_init_frqdis(*argvs):
    tmp = nltk.FreqDist()
    for argv in argvs:
        tmp.inc(argv)
    return tmp

def print_seg(kws):
    for kw in kws:
        print kw.token.word+'/'+kw.token.flag+'/'+str(kw.sntnc),str(kw.wrd_strt_pos),str(kw.wrd_end_pos)
    print

