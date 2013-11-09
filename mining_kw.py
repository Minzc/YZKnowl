#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
__author__ = 'congzicun'
import jieba
import sys
import kw_util
import nltk
import re

stop_dic = [ln.strip().decode('utf-8') for ln in open('dictionary/stopwords.txt').readlines()]
jieba.load_userdict('dictionary/new_words.txt')


def clean_data(ln):
    ln_elements = re.split(ur'[!.?…~;",]', kw_util.punc_replace(ln))
    clean_ln = []
    for ln_ele in ln_elements:
        if len(ln_ele.strip()) != 0 and re.search(r':\s+http', ln_ele) is None:
            clean_ln.append(ln_ele)
    return ' '.join(clean_ln)


def get_top_kw(lns):
    kw_dist = nltk.FreqDist()
    kw_flag = {}
    phrases = set()
    for ln in lns:
        ln = kw_util.tweet_filter(clean_data(re.sub('#(.+?)#', ' ', ln)))
        phrases |= set(ln.split(' '))
    for phrase in phrases:
        kws = list(jieba.posseg.cut(phrase))
        for kw in kws:
            if kw.word not in stop_dic and len(kw.word) > 1 and kw.flag != 'eng' and\
                    (kw.flag == 'a' or 'n' in kw.flag):
                kw_dist.inc(kw.word.lower())
                kw_flag[kw.word] = kw.flag
#    for kw,count in kw_dist.items():
#        print kw.encode('utf-8'),count
    return kw_dist, phrases, kw_flag


def cal_chi_squar(kw, nw, total, kw_dist, kw_pair_dist):
    chi = 0
    for otherkw, count in kw_dist:
        debug = False
        if otherkw == kw or nw < 10:
            continue
        pg = count/total
        frq_wg = kw_pair_dist[kw + '$' + otherkw]
        if frq_wg == 0:
            continue
        if debug:
            print kw + '$' + otherkw
            print 'frq_wg', frq_wg
        nwpg = pg * nw
        if debug:
            print pg, count, nwpg, frq_wg
        chi += (frq_wg - nwpg) * (frq_wg - nwpg) / nwpg
    return chi


def find_kw(kw_dist, phrases):
    total_kw = sum(kw_dist.values())
    kw_pair_dist = nltk.FreqDist()
    for ln in phrases:
        ln = kw_util.tweet_filter(clean_data(re.sub('#(.+?)#', ' ', ln)))
        kw_poses = kw_util.backward_maxmatch(ln, kw_dist, 100, 2)
        kws = []
        for kw_pos in kw_poses:
            kws.append(ln[kw_pos[0]:kw_pos[1]])
        kw_pairs = [kw1 + '$' + kw2 for kw1 in kws for kw2 in kws if kw1 != kw2]
        for kw_pair in kw_pairs:
            kw_pair_dist.inc(kw_pair)
    kw_chi = nltk.FreqDist()
    for kw, count in kw_dist.items():
        kw_chi.inc(kw, cal_chi_squar(kw, count, total_kw, kw_dist.items()[:100], kw_pair_dist))
#    kw_chi.inc(u'先人',cal_chi_squar(u'先人',1,total_kw,kw_dist.items()[:100],kw_pair_dist))
#    for k,value in kw_chi.items():
#        print k,value
    return kw_chi


if __name__ == '__main__':
    global lns
    FILE_NAME = sys.argv[1]
    lns = [ln.decode('utf-8') for ln in open(FILE_NAME).readlines()]
    kw_dist, phrases, kw_flag = get_top_kw(lns)
    chi = find_kw(kw_dist, phrases)
    for k in chi.keys()[:100]:
        flag = '商品特征'
        if 'a' in kw_flag[k]:
            flag = '情感词'
        print k.encode('utf-8') + '\t\t' + flag
