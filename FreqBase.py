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
import math

STOP_DIC = 'stopwords.txt'
LESS_THAN_THREE_WORDS = 1
MORE_THAN_THREE_WORDS = 2
LESS_THAN_TWO_PHRASE = 3
MORE_THAN_TWO_PHRASE = 4
LESS_THAN_ONE_SENT = 5
MORE_THAN_ONE_SENT = 6

APLHI = 0.5
DEBUG = False
KNWB_PATH = 'knowl-base.txt'

class Token:
    def __init__(self,sntnc_in_twt,phrase,word_strt_pos,word_end_pos,seg_token):
        self.sntnc = sntnc_in_twt
        self.phrase = phrase
        self.wrd_strt_pos = word_strt_pos
        self.wrd_end_pos = word_end_pos
        self.token = seg_token
class Seg_token:
    def __init__(self,word,flag):
        self.word = word
        self.flag = flag

def load_knw_base():
    lns = [ln.decode('utf-8').strip().lower() for ln in open(KNWB_PATH).readlines()]
    # indx: entity
    # value: class
    entity_class = {}
    # indx: entity
    # value: instance
    synonym = {}
    # sentiment
    # indx: instance
    # value: entity
    sent_dic = set()
    for ln in lns:
        if ln.startswith('#') or len(ln) ==0:
            continue
        entity,instances,classes = ln.split('\t')
        if classes == u'食品':
            # TODO: remove noise
            continue
        for cls in classes.split('|'):
            if len(cls) != 0:
                if u'情感词' in cls:
                    sent_dic.add(entity)
                else:
                    entity_class.setdefault(entity,[])
                    entity_class[entity].append(cls)
        synonym[entity] = entity
        for instance in instances.split('|'):
            if len(instance) != 0:
                synonym[instance] = entity
    return entity_class,synonym,sent_dic


def gen_model(infile='train.txt', obj_name = u'伊利谷粒多'):
    entity_class,synonym,senti_dic = load_knw_base()
    lns = [ln.decode('utf-8').lower() for ln in open(infile).readlines()]
    kw_pair_wd_dis,kw_pair_snt_dis,kw_pair_phrs_dis = {},{},{}
    kw_distr = nltk.FreqDist()

    for ln in lns:
        kws,obj_poss = generate_segment_lst_know(ln,synonym,obj_name)
        # Start Statistic
        for kw1 in kws:
            for kw2 in kws:
                if kw1 == kw2:
                    continue
                kw_pair = kw1.token.word+'$'+kw2.token.word
                kw_distr.inc(kw_pair)
                wd_dis_type,snt_dis_type,phrase_dis_type =  decide_dis_feature_type(kw1,kw2)
                # word distance
                kw_pair_wd_dis.setdefault(kw_pair,MyLib.create_and_init_frqdis(MORE_THAN_THREE_WORDS,LESS_THAN_THREE_WORDS))
                kw_pair_wd_dis[kw_pair].inc(wd_dis_type)

                # sentence distance
                kw_pair_snt_dis.setdefault(kw_pair,MyLib.create_and_init_frqdis(LESS_THAN_ONE_SENT,MORE_THAN_ONE_SENT))
                kw_pair_snt_dis[kw_pair].inc(snt_dis_type)

                # phrase distance
                kw_pair_phrs_dis.setdefault(kw_pair,MyLib.create_and_init_frqdis(LESS_THAN_TWO_PHRASE,MORE_THAN_TWO_PHRASE))
                kw_pair_phrs_dis[kw_pair].inc(phrase_dis_type)

    print '#WD_PAIR_DISTR'
    for kw_pair,wd_dises in kw_pair_wd_dis.items():
        for type,value in wd_dises.items():
            print kw_pair.encode('utf-8')+'$'+str(type)+'$'+str(value)
    print '#SNT_PAIR_DISTR'
    for kw_pair,snt_dises in kw_pair_snt_dis.items():
        for type,value in snt_dises.items():
            print kw_pair.encode('utf-8')+'$'+str(type)+'$'+str(value)
    print '#PHRS_PAIR_DISTR'
    for kw_pair,phrs_dises in kw_pair_phrs_dis.items():
        for type,value in phrs_dises.items():
            print kw_pair.encode('utf-8')+'$'+str(type)+'$'+str(value)
    print '#WD_DIST'
    for kw,count in kw_distr.items():
        print kw.encode('utf-8')+'$'+str(count+1)


def load_mdl(infile = 'snti_mdl.txt'):
    lns = [ln.decode('utf-8').lower().strip() for ln in open(infile).readlines()]
    kw_pair_wd_dis,kw_pair_snt_dis,kw_pair_phrs_dis = {},{},{}
    pair_distr = nltk.FreqDist()
    feature_type = -1
    for ln in lns:
        if ln == '#WD_PAIR_DISTR'.lower():
            feature_type = 0
            continue
        elif ln == '#SNT_PAIR_DISTR'.lower():
            feature_type = 1
            continue
        elif ln == '#WD_DIST'.lower():
            feature_type = 2
            continue
        elif ln == '#PHRS_PAIR_DISTR'.lower():
            feature_type = 3
            continue

        if feature_type == 0:
            kw1,kw2,type,value = ln.strip().split('$')
            kw_pair_wd_dis.setdefault(kw1+'$'+kw2,nltk.FreqDist())
            kw_pair_wd_dis[kw1+'$'+kw2].inc(int(type),int(value))
        elif feature_type == 1:
            kw1,kw2,type,value = ln.strip().split('$')
            kw_pair_snt_dis.setdefault(kw1+'$'+kw2,nltk.FreqDist())
            kw_pair_snt_dis[kw1+'$'+kw2].inc(int(type),int(value))
        elif feature_type == 2:
            kw1,kw2,value = ln.strip().split('$')
            pair_distr[kw1+'$'+kw2] = int(value)
        elif feature_type == 3:
            kw1,kw2,type,value = ln.strip().split('$')
            kw_pair_phrs_dis.setdefault(kw1+'$'+kw2,nltk.FreqDist())
            kw_pair_phrs_dis[kw1+'$'+kw2].inc(int(type),int(value))
    return kw_pair_wd_dis,kw_pair_phrs_dis,kw_pair_snt_dis,pair_distr

def generate_segment_lst_know(ln,synonym,obj_name):
    know_dic = set(synonym.keys())
    know_dic.add(obj_name)
    sub_sents = filter(lambda x: x != '',re.split(ur"[!.?…~]",punc_replace(ln)))
    kws,obj_poss = [],[]
    pre_phrases_len = 0
    phrase_position = 0
    for sub_sent_index in range(len(sub_sents)):
        if DEBUG:
            print sub_sents[sub_sent_index]
        for phrases in kw_util.tweet_filter(sub_sents[sub_sent_index]).strip().split(' '):
            if len(phrases) < 1:
                continue
            kw_poses = kw_util.backward_maxmatch( phrases, know_dic, 100, 2 )
            for kw_pos in kw_poses:
                start,end,abs_word_start,abs_word_end = kw_pos[0],\
                                                        kw_pos[1],\
                                                        pre_phrases_len+kw_pos[0],\
                                                        pre_phrases_len+kw_pos[1]
                keyword = phrases[start:end]
                if phrases[start:end] == obj_name:
                    obj_token = Token(sub_sent_index,phrase_position,abs_word_start,abs_word_end,Seg_token(synonym[keyword],'obj'))
                    kws.append(obj_token)
                    obj_poss.append(obj_token)
                else:
                    kws.append(Token(sub_sent_index,phrase_position,abs_word_start,abs_word_end,Seg_token(synonym[keyword],'')))
            pre_phrases_len += len(phrases)
            phrase_position += 1
    return kws,obj_poss
def decide_dis_feature_type(kw1,kw2):
    # absolute word distance
    word_distance = kw1.wrd_strt_pos - kw2.wrd_end_pos
    if word_distance < 0:
        word_distance = kw2.wrd_strt_pos - kw1.wrd_end_pos
    # Distance Feature
    wd_dis_type,snt_dis_type,phrase_dis_type = MORE_THAN_THREE_WORDS,\
                                               MORE_THAN_ONE_SENT,\
                                               MORE_THAN_TWO_PHRASE
    if abs(kw1.sntnc-kw2.sntnc) < 2:
        snt_dis_type = LESS_THAN_ONE_SENT
    if (word_distance/2) < 3:
        wd_dis_type = LESS_THAN_THREE_WORDS
    if abs(kw1.phrase - kw2.phrase) < 2:
        phrase_dis_type = LESS_THAN_TWO_PHRASE
    return wd_dis_type,snt_dis_type,phrase_dis_type

def select_sentiment_word(kws,snt_dic,feature,kw_pair_wd_dis,kw_pair_phrs_dis,kw_pair_snt_dis,pair_distr,total_pair_occur):
    # For each keyword, go through all words in the sub-sentence to mine best sentiment keywords
    max_likelihood = -1
    best_fit_senti = ''
    prv_dis_type = MORE_THAN_ONE_SENT

    for kw in kws:
        if feature == kw or kw.token.flag == 'obj':
            continue
        # Only classify words in sentiment dictionary or adjective
        if kw.token.word in snt_dic:
            snt_lkhd = 1
            wd_dis_type,snt_dis_type,phrase_dis_type = decide_dis_feature_type(kw,feature)
            feature_senti_pair = feature.token.word + '$' + kw.token.word
            # P(c=dis|kw-sentiment)
            if kw_pair_wd_dis.has_key(feature_senti_pair):
                snt_lkhd *= kw_pair_wd_dis[feature_senti_pair][wd_dis_type]\
                              / sum(kw_pair_wd_dis[feature_senti_pair].values())
            else:
                snt_lkhd *= 0.5
            if DEBUG:
                print 'SENTIMENT WORD:',feature_senti_pair,snt_lkhd,wd_dis_type


            if kw_pair_snt_dis.has_key(feature_senti_pair):
                snt_lkhd *= kw_pair_snt_dis[feature_senti_pair][snt_dis_type]\
                              / sum(kw_pair_snt_dis[feature_senti_pair].values())
            else:
                snt_lkhd *= 0.5
            if DEBUG:
                print 'SENTIMENT SENTENCE:',feature_senti_pair,snt_lkhd,snt_dis_type

            if kw_pair_phrs_dis.has_key(feature_senti_pair):
                snt_lkhd *= kw_pair_phrs_dis[feature_senti_pair][phrase_dis_type] \
                            / sum(kw_pair_phrs_dis[feature_senti_pair].values())
            else:
                snt_lkhd *= 0.5
            if DEBUG:
                print 'SENTIMENT PHRASE:',feature_senti_pair,snt_lkhd,phrase_dis_type

            # P(kw-sentiment)
            snt_lkhd = snt_lkhd * pair_distr.get(feature_senti_pair,1) / total_pair_occur
            if DEBUG:
                print 'Sentiment Final Score:',feature_senti_pair,snt_lkhd,pair_distr.get(feature_senti_pair,1)

            if snt_lkhd > max_likelihood or \
               (snt_lkhd == max_likelihood and wd_dis_type < prv_dis_type):
                best_fit_senti = kw.token.word
                max_likelihood = snt_lkhd
    return best_fit_senti

def class_new(infile='test_data.txt',obj_name = u'伊利谷粒多',model_name = 'model.txt',):
    # Load Model
    kw_pair_wd_dis,kw_pair_phrs_dis,kw_pair_snt_dis,pair_distr = load_mdl(model_name)
    entity_class,synonym,sent_dic = load_knw_base()
    total_pair_occur = 0
    for pair,dist in kw_pair_snt_dis.items():
        total_pair_occur += sum(dist.values())

    # Initial Data
    lns = [ln.decode('utf-8').lower() for ln in open(infile).readlines()]
    total_pair_occur = sum(pair_distr.values())
    # Classify Test Data
    for ln in lns:
        # Generate word segments list
        kws,obj_poss = generate_segment_lst_know(ln,synonym,obj_name)

        can_rst = {}
        if DEBUG:
            MyLib.print_seg(kws)
            print obj_poss
        for feature in kws:
            if feature.token.word == obj_name:
                continue
            # Cal the probability of the distance between the keyword and the object
            max_likelihood = -1
            obj_feature_pair = obj_name + '$' + feature.token.word
            for obj_pos in obj_poss:
                likelihood = 1
                # Distance Feature
                wd_dis_type,snt_dis_type,phrase_dis_type = decide_dis_feature_type(obj_pos,feature)

                # First compare <obj,keyword> pair
                # If the word pair didn't not exist in training data set, Compare <obj,flag> pair
                # if the <obj,flag> didn't exist in train data set, assign an equal value to each type
                # P(c=dis|obj-kw)

                # Word Distance Diff
                if kw_pair_wd_dis.has_key(obj_feature_pair):
                    likelihood *= kw_pair_wd_dis[obj_feature_pair][wd_dis_type] \
                                  / sum(kw_pair_wd_dis[obj_feature_pair].values())
                else:
                    likelihood *= 0.5

                if DEBUG:
                    print 'Wb Dist Feature:',feature.token.word,likelihood,wd_dis_type

                # Sentence Distance Diff
                if kw_pair_snt_dis.has_key(obj_feature_pair):
                    likelihood *= kw_pair_snt_dis[obj_feature_pair][snt_dis_type] \
                                  / sum(kw_pair_snt_dis[obj_feature_pair].values())
                else:
                    likelihood *= 0.5
                if DEBUG:
                      print 'Snt Dist Feature:',feature.token.word,likelihood,snt_dis_type

                # Phrase Distance Diff
                if kw_pair_phrs_dis.has_key(obj_feature_pair):
                    likelihood *= kw_pair_phrs_dis[obj_feature_pair][phrase_dis_type]\
                                  / sum(kw_pair_phrs_dis[obj_feature_pair].values())
                else:
                    likelihood *= 0.5
                if DEBUG:
                    print 'Phrs Dist Feature:',feature.token.word,likelihood,phrase_dis_type

                if likelihood > max_likelihood:
                    max_likelihood = likelihood



            best_fit_senti = select_sentiment_word(kws,sent_dic,feature,kw_pair_wd_dis,kw_pair_phrs_dis,kw_pair_snt_dis,pair_distr,total_pair_occur)
            likelihood = max_likelihood * pair_distr.get(obj_feature_pair,1) / total_pair_occur
            if feature.token.word in entity_class.keys():
                likelihood += 1
            else:
                best_fit_senti = ''
            can_rst[feature.token.word+'$'+best_fit_senti] = likelihood
        can_rst = sorted(can_rst.items(),key=operator.itemgetter(1),reverse=True)
        print ln.encode('utf-8')
        for k,v in can_rst:
                print '('+k.encode('utf-8')+','+str(v).encode('utf-8')+')',
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
    ln = re.sub(ur'（','(',ln)
    ln = re.sub(ur'）',')',ln)
    ln = re.sub(u'\.\.',u'…',ln)
    ln = re.sub(u'～','~',ln)
    ln = re.sub(r'\(.*?\)','',ln)
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
        for kw in kws:
            print kw.word+'/'+kw.flag,
        print
        print 'origin segment:'
        kws = list(jieba.posseg.cut(kw_util.tweet_filter(ln.decode('utf-8'))))
        for i in range(len(kws)):
            kws[i].word = kws[i].word.encode('utf-8')
        for kw in kws:
            print kw.word+'/'+kw.flag
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
        gen_model(sys.argv[2])
    elif sys.argv[1] == 'classify':
#        class_new('testCodeCrrct.txt')
        class_new(sys.argv[2])
    elif sys.argv[1] == 'segment':
        seg_files(sys.argv[2],sys.argv[3])
    elif sys.argv[1] == 'senti':
        gen_model(sys.argv[2])
    elif sys.argv[1] == 'test':
        class_new()
    elif sys.argv[1] == 'clean':
        train_data_clean(sys.argv[2])


