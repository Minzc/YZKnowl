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
MORE_THAN_ONE_SENT = 4

APLHI = 0.5
DEBUG = False
KNWB_PATH = 'knowl-base.txt'

class Token:
    def __init__(self,sntnc_in_twt,word_in_sntnc,seg_token):
        self.sntnc = sntnc_in_twt
        self.wrdpos = word_in_sntnc
        self.token = seg_token
class Seg_token:
    def __init__(self,word,flag):
        self.word = word
        self.flag = flag

def load_knw_base():
    lns = [ln.strip() for ln in open(KNWB_PATH).readlines()]
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
        for cls in classes.split('|'):
            if len(cls) != 0:
                if '情感词' in cls:
                    sent_dic.add(entity)
                else:
                    entity_class.setdefault(entity,[])
                    entity_class[entity].append(cls)
        synonym[entity] = entity
        for instance in instances.split('|'):
            if len(instance) != 0:
                synonym[instance] = entity
    return entity_class,synonym,sent_dic

#def gen_model_get_kws(ln,obj_name,stop_dic):
#    sub_sents = filter(lambda x: x != '',re.split(r"[!.?,]",
#        punc_replace(ln.decode('utf-8'))))
#    kws = []
#    for sub_sent_index in range(len(sub_sents)):
#        for phrases in kw_util.tweet_filter(sub_sents[sub_sent_index]).strip().split(' '):
#            if len(phrases) < 1:
#                continue
#            # Filter word segments
#            tmpkws = MyLib.seg_and_filter(phrases,obj_name,stop_dic)
#            for kw_pos_in_phase in range(len(tmpkws)):
#                kws.append(Token(sub_sent_index,kw_pos_in_phase,tmpkws[kw_pos_in_phase]))
#    return kws

def gen_model_get_kws_knwbase(ln,synonym,obj_name):
    know_dic = set(synonym.values())
    know_dic.add(obj_name)
    sub_sents = filter(lambda x: x != '',re.split(r"[!.?,]",punc_replace(ln.decode('utf-8'))))
    kws = []
    pre_phrases_len = 0
    for sub_sent_index in range(len(sub_sents)):
        for phrase in kw_util.tweet_filter(sub_sents[sub_sent_index]).strip().split(' '):
            if len(phrase) < 1:
                continue
            phrase = phrase.encode('utf-8')
            kw_poses = kw_util.backward_maxmatch( phrase, know_dic, 100, 2 )
            for kw_pos in kw_poses:
                start = kw_pos[0]
                end = kw_pos[1]
                # utf-8 code consumes 3 unicode for a single word
                kws.append(Token(sub_sent_index,int((pre_phrases_len+kw_pos[0])/6),Seg_token(synonym[phrase[start:end]],'')))
            pre_phrases_len += len(phrase)
    return kws

def gen_model(infile='train.txt', obj_name = '伊利谷粒多'):
    entity_class,synonym,senti_dic = load_knw_base()

    lns = [ln.lower() for ln in open(infile).readlines()]
    kw_pair_wd_dis = {}
    kw_pair_snt_dis = {}
    kw_distr = nltk.FreqDist()

    for ln in lns:
#        kws = gen_model_get_kws(ln,obj_name,stop_dic)
        kws = gen_model_get_kws_knwbase(ln,synonym,obj_name)
        # Start Statistic
        for i in range(len(kws)):
            kw_distr.inc(kws[i].token.word)
            for j in range(len(kws)):
                if i == j:
                    continue
                kw_pair = kws[i].token.word+'$'+kws[j].token.word
                # word distance
                wd_dis_type = MORE_THAN_THREE_WORDS
                if abs(i-j) < 3:
                    wd_dis_type = LESS_THAN_THREE_WORDS
                kw_pair_wd_dis.setdefault(kw_pair,MyLib.create_and_init_frqdis(MORE_THAN_THREE_WORDS,LESS_THAN_THREE_WORDS))
                kw_pair_wd_dis[kw_pair].inc(wd_dis_type)

                # sentence distance
                snt_dis_type = MORE_THAN_ONE_SENT
                if abs(kws[i].sntnc - kws[j].sntnc) < 2:
                    snt_dis_type = LESS_THAN_ONE_SENT
                kw_pair_snt_dis.setdefault(kw_pair,MyLib.create_and_init_frqdis(LESS_THAN_ONE_SENT,MORE_THAN_ONE_SENT))
                kw_pair_snt_dis[kw_pair].inc(snt_dis_type)
    print '#WD_PAIR_DISTR'
    for kw_pair,wd_dises in kw_pair_wd_dis.items():
        for type,value in wd_dises.items():
            print kw_pair+'$'+str(type)+'$'+str(value)
    print '#SNT_PAIR_DISTR'
    for kw_pair,snt_dises in kw_pair_snt_dis.items():
        for type,value in snt_dises.items():
            print kw_pair+'$'+str(type)+'$'+str(value)
    print '#WD_DIST'
    for kw,count in kw_distr.items():
        print kw+'$'+str(count+1)


def load_mdl(infile = 'snti_mdl.txt'):
    lns = [ln.strip() for ln in open(infile).readlines()]
    kw_pair_wd_dis = {}
    kw_pair_snt_dis = {}
    kw_distr = nltk.FreqDist()
    feature_type = -1
    for ln in lns:
        if ln == '#WD_PAIR_DISTR':
            feature_type = 0
            continue
        elif ln == '#SNT_PAIR_DISTR':
            feature_type = 1
            continue
        elif ln == '#WD_DIST':
            feature_type = 2
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
            kw,value = ln.strip().split('$')
            kw_distr[kw] = int(value)
    return kw_pair_wd_dis,kw_pair_snt_dis,kw_distr


#def generate_segment_lst(ln,obj_name,stop_dic):
#    # Split tweets by punctuation [. ? !]
#    # Divide the tweet into sub_sentences
#    sub_sents = filter(lambda x: x != '',re.split(r"[!.?,]",
#            punc_replace(ln.decode('utf-8'))))
#    kws = []
#    obj_poss = []
#    wd_pos = 0
#    for sub_sent_index in range(len(sub_sents)):
#        for phrases in kw_util.tweet_filter(sub_sents[sub_sent_index]).strip().split(' '):
#            if len(phrases) < 1:
#                continue
#            # Filter word segments, words whose length is 1 will be removed
#            # Besides, time, location, quantity, degree words would be removed as well
#            tmpkws = MyLib.seg_and_filter(phrases,obj_name,stop_dic)
#            for kw in tmpkws:
#                kws.append(Token(sub_sent_index,wd_pos,kw))
#                if kw.flag == 'obj':
#                    obj_poss.append(wd_pos)
#                wd_pos += 1
#    return kws,obj_poss

def generate_segment_lst_know(ln,synonym,obj_name):
    know_dic = set(synonym.values())
    know_dic.add(obj_name)
    sub_sents = filter(lambda x: x != '',re.split(r"[!.?,]",punc_replace(ln.decode('utf-8'))))
    kws,obj_poss = [],[]
    pre_phrases_len = 0
    for sub_sent_index in range(len(sub_sents)):
        for phrases in kw_util.tweet_filter(sub_sents[sub_sent_index]).strip().split(' '):
            if len(phrases) < 1:
                continue
            phrases = phrases.encode('utf-8')
            kw_poses = kw_util.backward_maxmatch( phrases, know_dic, 100, 2 )
            for kw_pos in kw_poses:
                start,end = kw_pos[0],kw_pos[1]
                if phrases[start:end] == obj_name:
                    kws.append(Token(sub_sent_index,int((pre_phrases_len+start)/6),Seg_token(phrases[start:end],'obj')))
                    obj_poss.append((sub_sent_index,int((pre_phrases_len+start)/6)))
                else:
                    kws.append(Token(sub_sent_index,int((pre_phrases_len+start)/6),Seg_token(phrases[start:end],'')))
            pre_phrases_len += len(phrases)
    return kws,obj_poss

def decide_dis_feature_type(obj_poss,feature):
    # Distance Feature
    wd_dis_type,snt_dis_type = MORE_THAN_THREE_WORDS,MORE_THAN_ONE_SENT
    if len(obj_poss) != 0:
        for obj_pos in obj_poss:
            if abs(obj_pos[0]-feature.sntnc) < 1:
                snt_dis_type = LESS_THAN_ONE_SENT
            if abs(obj_pos[1] - feature.wrdpos) < 3:
                wd_dis_type = LESS_THAN_THREE_WORDS
    return wd_dis_type,snt_dis_type

def select_sentiment_word(kws,snt_dic,feature,kw_pair_wd_dis,kw_pair_snt_dis,kw_distr,total_pair_occur):
    # For each keyword, go through all words in the sub-sentence to mine best sentiment keywords
    max_likelihood = -1
    best_fit_senti = ''
    for kw in kws:
        if feature == kw or kw.token.flag == 'obj':
            continue
        # Only classify words in sentiment dictionary or adjective
        if kw.token.word in snt_dic:
            snt_lkhd = 1

            wd_dis_type,snt_dis_type = decide_dis_feature_type([(kw.sntnc,kw.wrdpos)],feature)
            feature_senti_pair = feature.token.word + '$' + kw.token.word
            if kw_pair_wd_dis.has_key(feature_senti_pair):
                snt_lkhd *= kw_pair_wd_dis[feature_senti_pair][wd_dis_type]\
                              / sum(kw_pair_wd_dis[feature_senti_pair].values())
            else:
                snt_lkhd *= 0.5


            # P(c=dic|kw-sentiment)
            if kw_pair_snt_dis.has_key(feature_senti_pair):
                snt_lkhd *= kw_pair_snt_dis[feature_senti_pair][snt_dis_type]\
                              / sum(kw_pair_snt_dis[feature_senti_pair].values())
            else:
                snt_lkhd *= 0.5

            # P(kw-sentiment)
            snt_lkhd = snt_lkhd * kw_distr.get(feature_senti_pair,1) / total_pair_occur
            if DEBUG:
                print 'Final Score:',snt_lkhd
            if snt_lkhd > max_likelihood:
                best_fit_senti = kw.token.word
                max_likelihood = snt_lkhd
    return best_fit_senti

def class_new(infile='test_data.txt',obj_name = '伊利谷粒多',model_name = 'model.txt',
              usrdic_file = 'new_words.txt',snt_file = 'sentiment_dict.txt'):
    # Load Model
    kw_pair_wd_dis,kw_pair_snt_dis,kw_distr = load_mdl(model_name)
    class_entity,synonym,sent_dic = load_knw_base()
    total_pair_occur = 0
    for pair,dist in kw_pair_snt_dis.items():
        total_pair_occur += sum(dist.values())

    # Initial Data
    lns = [ln.lower().strip() for ln in open(infile).readlines()]
    total_kw_occur = sum(kw_distr.values())
    # Classify Test Data
    for ln in lns:
        # Generate word segments list
        kws,obj_poss = generate_segment_lst_know(ln,synonym,obj_name)

        can_rst = {}
        if DEBUG:
            MyLib.print_seg(kws)
            print obj_poss
        for feature in kws:
            if feature.token.word == obj_name or feature.token.word in sent_dic:
                continue
            # Cal the probability of the distance between the keyword and the object
            likelihood = 1

            # Distance Feature
            wd_dis_type,snt_dis_type = decide_dis_feature_type(obj_poss,feature)

            # First compare <obj,keyword> pair
            # If the word pair didn't not exist in training data set, Compare <obj,flag> pair
            # if the <obj,flag> didn't exist in train data set, assign an equal value to each type
            # P(c=dis|obj-kw)
            obj_feature_pair = obj_name + '$' + feature.token.word
            if kw_pair_wd_dis.has_key(obj_feature_pair):
                likelihood *= kw_pair_wd_dis[obj_feature_pair][wd_dis_type] \
                              / sum(kw_pair_wd_dis[obj_feature_pair].values())
            else:
                likelihood *= 0.5

            if DEBUG:
                print 'Wb Dist Feature:',feature.token.word,likelihood,wd_dis_type

            if kw_pair_snt_dis.has_key(obj_feature_pair):
                likelihood *= kw_pair_snt_dis[obj_feature_pair][snt_dis_type] \
                              / sum(kw_pair_snt_dis[obj_feature_pair].values())
            else:
                likelihood *= 0.5

            if DEBUG:
                print 'Snt Dist Feature:',feature.token.word,likelihood,snt_dis_type

            best_fit_senti = select_sentiment_word(kws,sent_dic,feature,kw_pair_wd_dis,kw_pair_snt_dis,kw_distr,total_pair_occur)

            likelihood = likelihood * kw_distr.get(feature.token.word,1) / total_kw_occur
            if best_fit_senti == '':
                can_rst[feature.token.word] = likelihood
            else:
                can_rst[feature.token.word+'$'+best_fit_senti] = likelihood
#            if DEBUG:
#                print 'Feature final score:' + feature.token.word,likelihood,kw_dis.get(feature.token.word,0.000000000001)
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


