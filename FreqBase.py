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

STOP_DIC = 'stopwords.txt'
LESS_THAN_THREE_WORDS = 1
MORE_THAN_THREE_WORDS = 2
LESS_THAN_ONE_SENT = 3
NO_CO_OCCURENCE = 4
DEBUG = False

class Token:
    def __init__(self,sntnc_in_twt,word_in_sntnc,seg_token):
        self.sntnc = sntnc_in_twt
        self.wrdpos = word_in_sntnc
        self.token = seg_token


def stat_sentiment(infile='train.txt', obj_name = '伊利谷粒多', usrdic_file = 'new_words.txt'):
    jieba.load_userdict(usrdic_file)
    stop_dic = {ln.strip() for ln in open(STOP_DIC).readlines()}
    lns = [ln.strip() for ln in open(infile).readlines()]
    kw_pair_dis = {}
    kw_dis = nltk.FreqDist()
    phrases_count = 0
    for ln in lns:
        # Split tweets by punctuation [. ? !]
        # Divide the tweet into sub_sentences which contains complete semantic
        sub_sents = filter(lambda x: x != '',re.split(r"[!.?]",punc_replace(ln.decode('utf-8'))))
        kws = []
        for sub_sent_index in range(len(sub_sents)):
            for phrases in kw_util.tweet_filter(sub_sents[sub_sent_index]).split(' '):
                if len(phrases) < 1:
                    continue
                phrases_count += 1
                # Filter word segments
                tmpkws = MyLib.seg_and_filter(phrases,obj_name,stop_dic)
                for kw_pos_in_phase in range(len(tmpkws)):
                    kws.append(Token(sub_sent_index,kw_pos_in_phase,tmpkws[kw_pos_in_phase]))
        # Start Statistic
        for i in range(len(kws)):
            kw_dis.inc(kws[i].token.word)
            kw_dis.inc(kws[i].token.flag)
            for j in range(len(kws)):
                if i == j:
                    continue
                type = NO_CO_OCCURENCE
                if kws[i].sntnc == kws[j].sntnc:
                    if abs(kws[i].wrdpos - kws[j].wrdpos) < 3:
                        type = LESS_THAN_THREE_WORDS
                    else:
                        type = MORE_THAN_THREE_WORDS
                elif abs(kws[i].sntnc - kws[j].sntnc) < 2:
                    type = LESS_THAN_ONE_SENT
                kw_pair_dis.setdefault(kws[i].token.word,{})
                if not kw_pair_dis[kws[i].token.word].has_key(kws[j].token.word):
                    kw_pair_dis[kws[i].token.word][kws[j].token.word] =\
                    MyLib.create_and_init_frqdis(LESS_THAN_THREE_WORDS,MORE_THAN_THREE_WORDS,LESS_THAN_ONE_SENT,NO_CO_OCCURENCE)
                if not kw_pair_dis[kws[i].token.word].has_key(kws[j].token.flag):
                    kw_pair_dis[kws[i].token.word][kws[j].token.flag] =\
                    MyLib.create_and_init_frqdis(LESS_THAN_THREE_WORDS,MORE_THAN_THREE_WORDS,LESS_THAN_ONE_SENT,NO_CO_OCCURENCE)
                kw_pair_dis[kws[i].token.word][kws[j].token.word].inc(type)
                kw_pair_dis[kws[i].token.word][kws[j].token.flag].inc(type)

    for kw_1, kw_1_pair_dis in kw_pair_dis.items():
        for kw_2,type_dis in kw_1_pair_dis.items():
            pair_count = sum(type_dis.values())
            for type, count in type_dis.items():
                if count > 1:
                    pair_count += (count - 1)
                print kw_1+'$'+kw_2+"$"+str(type)+'$'+str(count/pair_count)
            print kw_1+'$'+kw_2+'$'+str(pair_count/phrases_count)

    for kw,count in kw_dis.items():
        print kw+'$'+str((count+1)/sum(kw_dis.values()))


def load_mdl(infile = 'snti_mdl.txt'):
    lns = [ln.strip() for ln in open(infile).readlines()]
    snt_pair_dis = {}
    kw_pair_dis = {}
    kw_dis = {}
    for ln in lns:
        ln_segs = ln.split('$')
        if len(ln_segs) == 2:
            kw_dis[ln_segs[0]] = float(ln_segs[1])
        elif len(ln_segs) == 3:
            snt_pair_dis.setdefault(ln_segs[0],{})
            snt_pair_dis[ln_segs[0]][ln_segs[1]] = float(ln_segs[2])
        elif len(ln_segs) == 4:
            kw_pair_dis.setdefault(ln_segs[0],{})
            kw_pair_dis[ln_segs[0]].setdefault(ln_segs[1],{})
            kw_pair_dis[ln_segs[0]][ln_segs[1]][int(ln_segs[2])] = float(ln_segs[3])
    return kw_dis,snt_pair_dis,kw_pair_dis

def generate_segment_lst(ln,obj_name,stop_dic):
    # Split tweets by punctuation [. ? !]
    # Divide the tweet into sub_sentences
    sub_sents = filter(lambda x: x != '',re.split(r"[!.?]",punc_replace(ln.decode('utf-8'))))
    kws = []
    obj_poss = []
    indx = 0
    phrases_count = 0
    for sub_sent_index in range(len(sub_sents)):
        for phrases in kw_util.tweet_filter(sub_sents[sub_sent_index]).split(' '):
            if len(phrases) < 1:
                continue
            phrases_count += 1
            # Filter word segments, words whose length is 1 will be removed
            # Besides, time, location, quantity, degree words would be removed as well
            tmpkws = MyLib.seg_and_filter(phrases,obj_name,stop_dic)
            for kw_pos_in_phase in range(len(tmpkws)):
                kws.append(Token(sub_sent_index,kw_pos_in_phase,tmpkws[kw_pos_in_phase]))
                if tmpkws[kw_pos_in_phase].flag == 'obj':
                    obj_poss.append(indx)
                indx += 1
    return sub_sents,kws,obj_poss,phrases_count

def decide_dis_feature_type(obj_poss,feature,kws):
    # Distance Feature
    type = NO_CO_OCCURENCE
    if len(obj_poss) != 0:
        for obj_pos in obj_poss:
            if kws[obj_pos].sntnc == feature.sntnc:
                if abs(kws[obj_pos].wrdpos - feature.wrdpos) < 3:
                    type = LESS_THAN_THREE_WORDS
                else:
                    type = MORE_THAN_THREE_WORDS
            elif abs(kws[obj_pos].sntnc - feature.sntnc) < 2:
                type = LESS_THAN_ONE_SENT
    return type

def select_sentiment_word(kws,snt_dic,feature,kw_pair_dis,snt_pair,type):
    # For each keyword, go through all words in the sub-sentence to mine best sentiment keywords
    max_likelihood = -1
    best_fit_senti = ''
    for senti_kw in kws:
        if feature == senti_kw or senti_kw.token.flag == 'obj':
            continue
            # Only classify words in sentiment dictionary or adjective
        if senti_kw.token.word in snt_dic or senti_kw.token.flag == 'a':
            snt_lkhd = 1
            kw_pair_dis.setdefault(senti_kw.token.word,{})
            snt_pair.setdefault(senti_kw.token.word,{})

            # P(c=dic|kw-sentiment)
            if kw_pair_dis[senti_kw.token.word].has_key(feature.token.word):
                snt_lkhd *= kw_pair_dis[senti_kw.token.word][feature.token.word][type]
            else:
                snt_lkhd *= 0.25
            if DEBUG:
                print 'sentiment:',senti_kw.token.word,snt_lkhd,type
                # P(kw-sentiment)
            if snt_pair[senti_kw.token.word].has_key(feature.token.word):
                snt_lkhd = snt_lkhd*snt_pair[senti_kw.token.word][feature.token.word]
            else:
                snt_lkhd = snt_lkhd*0.00001
            if DEBUG:
                print 'Final Score:',snt_lkhd
            if snt_lkhd > max_likelihood:
                best_fit_senti = senti_kw.token.word
    return max_likelihood,best_fit_senti

def class_new(infile='test_data.txt',obj_name = '伊利谷粒多',model_name = 'model.txt',
              usrdic_file = 'new_words.txt',snt_file = 'sentiment_dict.txt'):
    # Load Model
    kw_dis,snt_pair,kw_pair_dis = load_mdl(model_name)

    # Initial Data
    jieba.load_userdict(usrdic_file)
    stop_dic = {ln.strip() for ln in open(STOP_DIC).readlines()}
    snt_dic = dict([ln.split('\t') for ln in open(snt_file).readlines() if not ln.startswith('#')])
    lns = [ln.strip() for ln in open(infile).readlines()]
    phrases_count = 0

    # Classify Test Data
    for ln in lns:
        # Generate word segments list
        sub_sents,kws,obj_poss,sub_phrases_count = generate_segment_lst(ln,obj_name,stop_dic)
        phrases_count += sub_phrases_count

        can_rst = {}
        if DEBUG:
            MyLib.print_seg(kws)
            print obj_poss
        for feature in kws:
            if feature.token.word == obj_name:
                continue
            # Cal the probability of the distance between the keyword and the object
            likelihood = 1

            # Distance Feature
            type = decide_dis_feature_type(obj_poss,feature,kws)

            # First compare <obj,keyword> pair
            # If the word pair didn't not exist in training data set, Compare <obj,flag> pair
            # if the <obj,flag> didn't exist in train data set, assign an equal value to each type
            # P(c=dis|obj-kw)
            if kw_pair_dis[obj_name].has_key(feature.token.word):
                likelihood *= kw_pair_dis[obj_name][feature.token.word][type]
            else:
                likelihood *= 0.25
            if DEBUG:
                print 'Feature:',feature.token.word,likelihood,type
            # For each keyword, go through all words in the sub-sentence to mine best sentiment keywords
            max_likelihood,best_fit_senti = select_sentiment_word(kws,snt_dic,feature,kw_pair_dis,snt_pair,type)

            # P(keyword)
            likelihood *= kw_dis.get(feature.token.word,0.000000000001)

            if DEBUG:
                print 'Feature final score:' + feature.token.word,likelihood,kw_dis.get(feature.token.word,0.000000000001)
            if max_likelihood != -1:
                can_rst[feature.token.word+'$'+best_fit_senti] = likelihood
            else:
                can_rst[feature.token.word] = likelihood
        can_rst = sorted(can_rst.items(),key=operator.itemgetter(1),reverse=True)
        print ln
        for k,v in can_rst:
                print '('+k+','+str(v)+')',
        print

# convert chines punctuation to english punctuation
# 1. period
# 2. exclaim
#
def punc_replace(ln):
    ln = re.sub(ur"。",'.',ln)
    ln = re.sub(ur'！','!',ln)
    ln = re.sub(ur'？','?',ln)
    ln = re.sub(ur'，',',',ln)
    return ln

def train_data_clean(infile):
    lns = [ln.strip() for ln in open(infile).readlines()]
    clean_lns = {}
    for ln in lns:
        tmp_ln = kw_util.tweet_filter(ln)
        if not clean_lns.has_key(tmp_ln):
            clean_lns[tmp_ln] = ln
    for ln in clean_lns.values():
        print ln

def seg_files(infile='test_data.txt', obj_name = '伊利谷粒多', usrdic_file = 'new_words.txt'):
    jieba.load_userdict(usrdic_file)
    stop_dic = {ln.strip() for ln in open(STOP_DIC).readlines()}
    lns = [ln.strip() for ln in open(infile).readlines()]

    for ln in lns:
        print ln
        kws = MyLib.seg_and_filter(kw_util.tweet_filter(ln.decode('utf-8')),obj_name,stop_dic)
        MyLib.print_seg(kws)
        print 'origin segment:'
        kws = list(jieba.posseg.cut(kw_util.tweet_filter(ln.decode('utf-8'))))
        for i in range(len(kws)):
            kws[i].word = kws[i].word.encode('utf-8')
        MyLib.print_seg(kws)
        print '\n'

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print """Usage: python FreqBase.py <cmd> <input_file> where
<input_file> line delimited text;
<cmd> = [gen_model|classify|gen_kw_frq|segment]
\tgen_model <input_file> <obj_name> [user_dic], generate model;
\tgen_kw_frq <input_file> <obj_name> [user_dic], generate keyword frequency statistics
\tsegment <infile> <obj_name> [user_dic], segment sentences
\tclassify <infile> <obj_name> <model_file> <kw_freq_file> [user_dic], classify test data

"""

    if sys.argv[1] == 'gen_model':
        stat_sentiment(sys.argv[2])
    elif sys.argv[1] == 'classify':
#        class_new('testCodeCrrct.txt')
        class_new(sys.argv[2])
    elif sys.argv[1] == 'segment':
        seg_files(sys.argv[2],sys.argv[3])
    elif sys.argv[1] == 'senti':
        stat_sentiment(sys.argv[2])
    elif sys.argv[1] == 'test':
        class_new()
    elif sys.argv[1] == 'clean':
        train_data_clean(sys.argv[2])


