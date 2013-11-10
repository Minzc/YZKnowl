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
import prior_rules
import feature_senti_ruler


ALPHA = 1
DEBUG = False

STOP_DIC = 'stopwords.txt'
KNWB_PATH = 'knowl-base.txt'
FILE_PREFIX = 'yinlu'
TRAIN_FILE_PAHT = FILE_PREFIX + '_train_data.txt'
TEST_FILE_PAHT = FILE_PREFIX + '_test_data.txt'
MODEL_FILE_PATH = FILE_PREFIX + '_model.txt'
OBJ_NAME = u'银鹭花生牛奶'

class Local_Model:
    F_S_TYPE = {}
    S_AMB = {}
    F_NULL = {}
    CERNTAIN_PAIR = nltk.FreqDist()
    KW_DIS = nltk.FreqDist()
    FS_NUM = nltk.FreqDist()
    F_S_SET = {}
    TRAIN_SET_VOLUME = 0
    TOTAL_NULL_SENTI = 0
class Global_Model:
    F_S_TYPE = {}
    FS_NUM = nltk.FreqDist()
    S_AMB = {}
    KW_DIS = {}


class Dis_Type:
    LESS_THAN_ONE_WORDS = 1
    LESS_THAN_THREE_WORDS = 2
    MORE_THAN_THREE_WORDS = 3
    LESS_THAN_TWO_PHRASE = 4
    LESS_THAN_FOUR_PHRASE = 5
    MORE_THAN_FOUR_PHRASE = 6
    LESS_THAN_ONE_SENT = 7
    MORE_THAN_ONE_SENT = 8
    PRIOR = 9
    POSTERIOR = 10


class Result:
    def __init__(self):
        self.POS_FEATURE = {}
        self.NEG_FEATURE = {}
        self.TAG_TWEETS = {}
        self.TOTAL_TWEETS = 0
        self.POS_TAGS = nltk.FreqDist()
        self.NEG_TAGS = nltk.FreqDist()
        self.FEATURE = nltk.FreqDist()
        self.FEATURE_POS = nltk.FreqDist()
        self.FEATURE_NEG = nltk.FreqDist()


class Token:
    def __init__(self, sntnc_in_twt, phrase, word_strt_pos, word_end_pos, seg_token):
        self.sntnc = sntnc_in_twt
        self.phrase = phrase
        self.wrd_strt_pos = word_strt_pos
        self.wrd_end_pos = word_end_pos
        self.token = seg_token


class Seg_token:
    def __init__(self, word, flag):
        self.word = word
        self.flag = flag


def load_knw_base():
    """Load knowledge base to memory
    :Return entity_class    : entity -> classes. Value is a list
    :Return synonym         : instance -> entity.
    :Return sent_dic        : sentiment dictionary
    """
    lns = [ln.decode('utf-8').strip().lower() for ln in open(KNWB_PATH).readlines()]
    # indx -> entity , value -> class
    entity_class = {}
    # indx -> entity , value -> instance
    synonym = {}
    # sentiment
    # indx -> instance value -> entity
    sent_dic = {}
    degree_dic = set()
    for ln in lns:
        if ln.startswith('#') or len(ln) == 0:
            continue
        entity, instances, classes = ln.split('\t')
        if classes == u'食品':
            continue
        for cls in classes.split('|'):
            if len(cls) != 0:
                if u'情感词' in cls:
                    if u'正面' in cls:
                        sent_dic[entity] = 'p'
                    elif u'负面' in cls:
                        sent_dic[entity] = 'n'
                    elif u'程度' in cls:
                        degree_dic.add(entity)
                else:
                    entity_class.setdefault(entity, [])
                    entity_class[entity].append(cls)
        synonym[entity] = entity
        for instance in instances.split('|'):
            if len(instance) != 0:
                synonym[instance] = entity
    return entity_class, synonym, sent_dic, degree_dic


def if_ambiguity(kw1, kw2, senti_dic):
    """Judge if kw1 is an ambiguous word
    :param kw1: keyword to be judged
    :param kw2: accompanied keyword
    :param senti_dic: sentiment dictionary
    :Returns  : True if kw1 is not ambiguous
    """
    if kw1.phrase == kw2.phrase \
        and kw2.token.word in senti_dic \
        and kw1.token.word in senti_dic \
        and kw1.token.word != kw2.token.word:
        return False
    return True


def gen_model(infile=TRAIN_FILE_PAHT, obj_name=OBJ_NAME):
    """Given training data set, generate Model file
    : Param infile   : training data file path
    : Param obj_name : object name
    """
    entity_class, synonym, senti_dic, degree_dic = load_knw_base()
    tmp_lns, lns = [ln.decode('utf-8').lower() for ln in open(infile).readlines()], []
    sentiment_ambi = nltk.FreqDist()
    f_null_docs = nltk.FreqDist()
    kw_distr = nltk.FreqDist()
    total_docs = 0
    total_null_docs = 0

    # Format lns by prior rules
    for tmp_ln in tmp_lns:
        lns += prior_rules.prior_rules(tmp_ln)

    for ln in lns:
        ln = re.sub('//@.+', '', ln)
        if DEBUG:
            print 'Gen_Model', ln.encode('utf-8')
        kws, obj_poss = seg_ln(ln, synonym, obj_name, entity_class)
        kws = feature_senti_ruler.combine_sentiment(kws, senti_dic, degree_dic)
        if len(kws) != 0:
            total_docs += 1
            # Start Statistic
        if_has_senti = False
        for kw in kws:
            if kw.token.word in senti_dic:
                if_has_senti = True

        for kw1 in kws:
            not_ambiguity = True
            kw_distr.inc(kw1.token.word)
            # TODO: hard code
            if 'sentiment' in kw1.token.flag:
                kw_distr.inc('sentiment')
            else:
                kw_distr.inc('feature')

            for kw2 in kws:
                if kw1 == kw2:
                    continue

                instance_class_pair = kw1.token.word + '$sentiment'
                class_instance_pair = 'feature$' + kw2.token.word
                class_class_pair = 'feature$sentiment'
                not_ambiguity &= if_ambiguity(kw1, kw2, senti_dic)
                if_has_senti |= (kw2.token.word in senti_dic)

                kw_pair = kw1.token.word + '$' + kw2.token.word
                Local_Model.FS_NUM.inc(kw_pair)
                # Cerntain Pair Feature
                if feature_senti_ruler.abs_pos(kw1,kw2):
                    Local_Model.CERNTAIN_PAIR.inc(kw_pair)
                # Distance Feature
                wd_dis_type, snt_dis_type, phrase_dis_type, relative_pos \
                    = decide_dis_feature_type(kw1, kw2)

                Local_Model.F_S_TYPE.setdefault(kw_pair,
                                              MyLib.create_and_init_frqdis(
                                                  Dis_Type.LESS_THAN_ONE_WORDS,
                                                  Dis_Type.MORE_THAN_THREE_WORDS,
                                                  Dis_Type.LESS_THAN_THREE_WORDS,
                                                  Dis_Type.LESS_THAN_ONE_SENT,
                                                  Dis_Type.MORE_THAN_ONE_SENT,
                                                  Dis_Type.LESS_THAN_TWO_PHRASE,
                                                  Dis_Type.LESS_THAN_FOUR_PHRASE,
                                                  Dis_Type.MORE_THAN_FOUR_PHRASE,
                                                  Dis_Type.PRIOR,
                                                  Dis_Type.POSTERIOR
                                              ))
                Local_Model.F_S_TYPE.setdefault(instance_class_pair,
                                              MyLib.create_and_init_frqdis(
                                                  Dis_Type.LESS_THAN_ONE_WORDS,
                                                  Dis_Type.MORE_THAN_THREE_WORDS,
                                                  Dis_Type.LESS_THAN_THREE_WORDS,
                                                  Dis_Type.LESS_THAN_ONE_SENT,
                                                  Dis_Type.MORE_THAN_ONE_SENT,
                                                  Dis_Type.LESS_THAN_TWO_PHRASE,
                                                  Dis_Type.LESS_THAN_FOUR_PHRASE,
                                                  Dis_Type.MORE_THAN_FOUR_PHRASE,
                                                  Dis_Type.PRIOR,
                                                  Dis_Type.POSTERIOR
                                              ))
                Local_Model.F_S_TYPE.setdefault(class_instance_pair,
                                              MyLib.create_and_init_frqdis(
                                                  Dis_Type.LESS_THAN_ONE_WORDS,
                                                  Dis_Type.MORE_THAN_THREE_WORDS,
                                                  Dis_Type.LESS_THAN_THREE_WORDS,
                                                  Dis_Type.LESS_THAN_ONE_SENT,
                                                  Dis_Type.MORE_THAN_ONE_SENT,
                                                  Dis_Type.LESS_THAN_TWO_PHRASE,
                                                  Dis_Type.LESS_THAN_FOUR_PHRASE,
                                                  Dis_Type.MORE_THAN_FOUR_PHRASE,
                                                  Dis_Type.PRIOR,
                                                  Dis_Type.POSTERIOR
                                              ))

                Local_Model.F_S_TYPE.setdefault(class_class_pair,
                                              MyLib.create_and_init_frqdis(
                                                  Dis_Type.LESS_THAN_ONE_WORDS,
                                                  Dis_Type.MORE_THAN_THREE_WORDS,
                                                  Dis_Type.LESS_THAN_THREE_WORDS,
                                                  Dis_Type.LESS_THAN_ONE_SENT,
                                                  Dis_Type.MORE_THAN_ONE_SENT,
                                                  Dis_Type.LESS_THAN_TWO_PHRASE,
                                                  Dis_Type.LESS_THAN_FOUR_PHRASE,
                                                  Dis_Type.MORE_THAN_FOUR_PHRASE,
                                                  Dis_Type.PRIOR,
                                                  Dis_Type.POSTERIOR
                                              ))
                Local_Model.F_S_TYPE[kw_pair].inc(wd_dis_type)
                Local_Model.F_S_TYPE[kw_pair].inc(snt_dis_type)
                Local_Model.F_S_TYPE[kw_pair].inc(phrase_dis_type)
                Local_Model.F_S_TYPE[kw_pair].inc(relative_pos)

                if kw2.token.word in senti_dic:
                    Local_Model.F_S_TYPE[instance_class_pair].inc(wd_dis_type)
                    Local_Model.F_S_TYPE[instance_class_pair].inc(snt_dis_type)
                    Local_Model.F_S_TYPE[instance_class_pair].inc(phrase_dis_type)
                    Local_Model.F_S_TYPE[instance_class_pair].inc(relative_pos)
                if kw1.token.word not in senti_dic:
                    Local_Model.F_S_TYPE[class_instance_pair].inc(wd_dis_type)
                    Local_Model.F_S_TYPE[class_instance_pair].inc(snt_dis_type)
                    Local_Model.F_S_TYPE[class_instance_pair].inc(phrase_dis_type)
                    Local_Model.F_S_TYPE[class_instance_pair].inc(relative_pos)
                if kw1.token.word not in senti_dic and kw2.token.word in senti_dic:
                    Local_Model.F_S_TYPE[class_class_pair].inc(wd_dis_type)
                    Local_Model.F_S_TYPE[class_class_pair].inc(snt_dis_type)
                    Local_Model.F_S_TYPE[class_class_pair].inc(phrase_dis_type)
                    Local_Model.F_S_TYPE[class_class_pair].inc(relative_pos)
            # Sentiment Ambiguity
            if not_ambiguity:
                sentiment_ambi.inc(kw1.token.word)
            # Null Sentiment Feature
            if not if_has_senti:
                f_null_docs.inc(kw1.token.word)
        if not if_has_senti and len(kws) != 0:
            total_null_docs += 1

    # Output Model File
    print '#FEATURE_SENTI_TYPE_DISTR'
    for kw_pair, type_dises in Local_Model.F_S_TYPE.items():
        for dis_type, value in type_dises.items():
            print kw_pair.encode('utf-8') + '$' + str(dis_type) + '$' + str(value)
    print '#FEATURE_SENTI_DIST'
    for kw, count in Local_Model.FS_NUM.items():
        print kw.encode('utf-8') + '$' + str(count + 1)
    print '#AMBI_DIST'
    for kw, count in sentiment_ambi.items():
        print kw.encode('utf-8') + '$' + str(count + 1) + '$' + str(kw_distr.get(kw) + 2)
    print '#FEATURE_NULL'
    for kw, count in f_null_docs.items():
        print kw.encode('utf-8') + '$' + str(count + 1) + '$' + str(kw_distr.get(kw) + 2)
    print '#KW_DIS'
    for kw, count in kw_distr.items():
        print kw.encode('utf-8') + '$' + str(count)
    print '#TOTAL_DOC'
    print str(total_null_docs) + '$' + str(total_docs)
    print '#CERTAIN_PAIR_DIS'
    for kw_pair,count in Local_Model.CERNTAIN_PAIR.items():
        # Magic number
        if count > 5:
            print kw_pair.encode('utf-8')+'$'+str(count)


def load_glb_mdl(infile):
    """Load global model
    :Param infile: model file path
    """
    lns = [ln.decode('utf-8').lower().strip() for ln in open(infile).readlines()]
    for ln in lns:
        key, value = ln.split('\t')
        ln_elemnts = key.split('$')
        # Global model didn't smooth data when it's trained, the operation is needed to be done by ourselves
        if ln_elemnts[0] == 'FST_DIST'.lower():
            fs_pair = ln_elemnts[1] + '$' + ln_elemnts[2]
            feature_type = int(ln_elemnts[3])
            Global_Model.F_S_TYPE.setdefault(fs_pair,
                                             MyLib.create_and_init_frqdis(
                                                 Dis_Type.LESS_THAN_ONE_WORDS,
                                                 Dis_Type.MORE_THAN_THREE_WORDS,
                                                 Dis_Type.LESS_THAN_THREE_WORDS,
                                                 Dis_Type.LESS_THAN_ONE_SENT,
                                                 Dis_Type.MORE_THAN_ONE_SENT,
                                                 Dis_Type.LESS_THAN_TWO_PHRASE,
                                                 Dis_Type.LESS_THAN_FOUR_PHRASE,
                                                 Dis_Type.MORE_THAN_FOUR_PHRASE,
                                                 Dis_Type.PRIOR,
                                                 Dis_Type.POSTERIOR
                                             ))
            Global_Model.F_S_TYPE[fs_pair].inc(feature_type, int(value))
        elif ln_elemnts[0] == 'FS_DIST'.lower():
            fs_pair = ln_elemnts[1] + '$' + ln_elemnts[2]
            Global_Model.FS_NUM.inc(fs_pair,int(value))
        elif ln_elemnts[0] == 'AMBI_DIST'.lower():
            Global_Model.S_AMB[ln_elemnts[1]] = int(value)
        elif ln_elemnts[0] == 'KW_DIST'.lower():
            Global_Model.KW_DIS[ln_elemnts[1]] = int(value)
    for kw in Global_Model.S_AMB.keys():
        if kw in Global_Model.KW_DIS:
            Global_Model.S_AMB[kw] = 1 - Global_Model.S_AMB[kw] / (Global_Model.S_AMB[kw] + Global_Model.KW_DIS.get(kw,0))


def load_mdl(infile=MODEL_FILE_PATH):
    lns = [ln.decode('utf-8').lower().strip() for ln in open(infile).readlines()]
    feature_type = -1
    for ln in lns:
        if ln == '#FEATURE_SENTI_TYPE_DISTR'.lower():
            feature_type = 0
            continue
        elif ln == '#FEATURE_SENTI_DIST'.lower():
            feature_type = 1
            continue
        elif ln == '#AMBI_DIST'.lower():
            feature_type = 2
            continue
        elif ln == '#FEATURE_NULL'.lower():
            feature_type = 3
            continue
        elif ln == '#KW_DIS'.lower():
            feature_type = 4
            continue
        elif ln == '#TOTAL_DOC'.lower():
            feature_type = 5
            continue
        elif ln == '#CERTAIN_PAIR_DIS'.lower():
            feature_type = 6
            continue

        if feature_type == 0:
            kw1, kw2, dis_type, value = ln.strip().split('$')
            Local_Model.F_S_TYPE.setdefault(kw1 + '$' + kw2, nltk.FreqDist())
            Local_Model.F_S_TYPE[kw1 + '$' + kw2].inc(int(dis_type), int(value))
        elif feature_type == 1:
            kw1, kw2, value = ln.strip().split('$')
            Local_Model.FS_NUM[kw1 + '$' + kw2] = int(value)
            Local_Model.F_S_SET.setdefault(kw1, set())
            if kw2 != 'sentiment' and kw2 != 'feature':
                Local_Model.F_S_SET[kw1].add(kw2)
        elif feature_type == 2:
            sent_kw, not_amb_doc, total_doc = ln.strip().split('$')
            Local_Model.S_AMB[sent_kw] = int(not_amb_doc) / int(total_doc)
        elif feature_type == 3:
            feature, null_senti, total_doc = ln.strip().split('$')
            Local_Model.F_NULL[feature] = int(null_senti)
        elif feature_type == 4:
            kw, count = ln.strip().split('$')
            Local_Model.KW_DIS.inc(kw, int(count))
        elif feature_type == 5:
            global TOTAL_NULL_SENTI
            global TRAIN_SET_VOLUME

            total_null_doc, total_doc = ln.strip().split('$')
            TOTAL_NULL_SENTI = int(total_null_doc)
            TRAIN_SET_VOLUME = int(total_doc)
        elif feature_type == 6:
            feature,sentiment,count = ln.strip().split('$')
            Local_Model.CERNTAIN_PAIR.inc(feature+'$'+sentiment,int(count))


def seg_ln(ln, synonym, obj_name, entity_class):
    know_dic = set(synonym.keys())
    know_dic.add(obj_name)
    sub_sents = filter(lambda x: x != '', re.split(ur'[!.?…~;"#]', kw_util.punc_replace(ln)))
    kws, obj_poss = [], []
    pre_phrases_len = 0
    phrase_position = 0
    for index, sub_sent in enumerate(sub_sents):
        if DEBUG:
            print sub_sent.encode('utf-8')
        for phrase in kw_util.tweet_filter(sub_sent).strip().split(' '):
            if len(phrase) < 1 or len(phrase) > 30:
                continue
            kw_poses = kw_util.backward_maxmatch(phrase, know_dic, 100, 1)
            for kw_pos in kw_poses:
                start, end, abs_word_start, abs_word_end = kw_pos[0], kw_pos[1], pre_phrases_len + kw_pos[
                    0], pre_phrases_len + kw_pos[1]
                keyword = phrase[start:end]
                if synonym[phrase[start:end]] == obj_name:
                    obj_token = Token(index, phrase_position, abs_word_start, abs_word_end,
                                      Seg_token(synonym[keyword], 'obj'))
                    kws.append(obj_token)
                    obj_poss.append(obj_token)
                else:
                    # TODO: hard code
                    flag = 'sentiment'
                    if synonym[keyword] in entity_class:
                        flag = entity_class[synonym[keyword]][0]
                        # TODO: flag 需要细化
                    kws.append(
                        Token(index, phrase_position, abs_word_start, abs_word_end, Seg_token(synonym[keyword], flag)))
            pre_phrases_len += len(phrase)
            phrase_position += 1
    return kws, obj_poss


def decide_dis_feature_type(kw1, kw2):
    """Give two keywords, generate features between the given words
    : Param kw1               : keyword one
    : Param kw1               : keyword two
    : Returns wd_dis_type     : distance type in word level
    : Returns snt_dis_type    : distance type in phrase level
    : Returns phrase_dis_type : disatnce type in sentence level
    : Returns relative_pos    : relative position type
    """
    # absolute word distance
    word_distance = kw1.wrd_strt_pos - kw2.wrd_end_pos
    if word_distance < 0:
        word_distance = kw2.wrd_strt_pos - kw1.wrd_end_pos
        # Distance Feature
    wd_dis_type, snt_dis_type, phrase_dis_type, relative_pos = \
        Dis_Type.MORE_THAN_THREE_WORDS, \
        Dis_Type.MORE_THAN_ONE_SENT, \
        Dis_Type.MORE_THAN_FOUR_PHRASE, \
        Dis_Type.PRIOR
    # Sentence Distance Feature
    if abs(kw1.sntnc - kw2.sntnc) < 2:
        snt_dis_type = Dis_Type.LESS_THAN_ONE_SENT
        # Word Distance Feature
    if abs(word_distance / 2) < 1:
        wd_dis_type = Dis_Type.LESS_THAN_ONE_WORDS
    elif abs(word_distance / 2) < 3:
        wd_dis_type = Dis_Type.LESS_THAN_THREE_WORDS
        # Phrase Distance Feature
    if abs(kw1.phrase - kw2.phrase) < 2:
        phrase_dis_type = Dis_Type.LESS_THAN_TWO_PHRASE
    elif abs(kw1.phrase - kw2.phrase) < 4:
        phrase_dis_type = Dis_Type.LESS_THAN_FOUR_PHRASE
    if kw1.wrd_strt_pos > kw2.wrd_strt_pos:
        relative_pos = Dis_Type.POSTERIOR
    return wd_dis_type, snt_dis_type, phrase_dis_type, relative_pos


def cal_likelihood(kw, feature):
    snt_lkhd = 1
    wd_dis_type, snt_dis_type, phrase_dis_type, relative_pos \
        = decide_dis_feature_type(kw, feature)
    feature_senti_pair = kw.token.word + '$' + feature.token.word
    # P(c=dis|kw-sentiment)
    # Word Distance Feature
    if feature_senti_pair in Local_Model.F_S_TYPE:
        if DEBUG:
            print 'Local Model'
        total_pairs = \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_ONE_WORDS] + \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_THREE_WORDS] + \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.MORE_THAN_THREE_WORDS]
        snt_lkhd *= Local_Model.F_S_TYPE[feature_senti_pair][wd_dis_type] / total_pairs
    elif feature_senti_pair in Global_Model.F_S_TYPE:
        if DEBUG:
            print 'Global Model'
        total_pairs = \
        Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_ONE_WORDS] + \
        Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_THREE_WORDS] + \
        Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.MORE_THAN_THREE_WORDS]
        snt_lkhd = Global_Model.F_S_TYPE[feature_senti_pair][wd_dis_type] / total_pairs
    else:
        snt_lkhd *= 0.3
    if DEBUG:
        print 'SENTIMENT WORD:', feature_senti_pair, snt_lkhd, wd_dis_type
        # Sentence Distance Feature
    if feature_senti_pair in Local_Model.F_S_TYPE:
        if DEBUG:
            print 'Local Model'
        total_pairs = \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_ONE_SENT] + \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.MORE_THAN_ONE_SENT]
        snt_lkhd *= \
            Local_Model.F_S_TYPE[feature_senti_pair][snt_dis_type] / total_pairs
    elif feature_senti_pair in Global_Model.F_S_TYPE:
        if DEBUG:
            print 'Global Model'
        total_pairs = \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_ONE_SENT] + \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.MORE_THAN_ONE_SENT]
        snt_lkhd *= \
            Global_Model.F_S_TYPE[feature_senti_pair][snt_dis_type] / total_pairs
    else:
        snt_lkhd *= 0.5
    if DEBUG:
        print 'SENTIMENT SENTENCE:', feature_senti_pair, snt_lkhd, snt_dis_type
        # Phrase Distance Feature
    if feature_senti_pair in Local_Model.F_S_TYPE:
        if DEBUG:
            print 'Local Model'
        total_pairs = \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_TWO_PHRASE] + \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_FOUR_PHRASE] + \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.MORE_THAN_FOUR_PHRASE]
        snt_lkhd *= \
            Local_Model.F_S_TYPE[feature_senti_pair][phrase_dis_type] / total_pairs
    elif feature_senti_pair in Global_Model.F_S_TYPE:
        if DEBUG:
            print 'Global Model'
        total_pairs = \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_TWO_PHRASE] + \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_FOUR_PHRASE] + \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.MORE_THAN_FOUR_PHRASE]
        snt_lkhd *= \
            Global_Model.F_S_TYPE[feature_senti_pair][phrase_dis_type] / total_pairs
    else:
        snt_lkhd *= 0.3
    if DEBUG:
        print 'SENTIMENT PHRASE:', feature_senti_pair, snt_lkhd, phrase_dis_type
    # Relative Position Feature
    if feature_senti_pair in Local_Model.F_S_TYPE:
        if DEBUG:
            print 'Local Model'
        total_pairs = \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.PRIOR] + \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.POSTERIOR]
        snt_lkhd *= \
            Local_Model.F_S_TYPE[feature_senti_pair][relative_pos] / total_pairs
    elif feature_senti_pair in Global_Model.F_S_TYPE:
        if DEBUG:
            print 'Global Model'
        total_pairs = \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.PRIOR] + \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.POSTERIOR]
        snt_lkhd *= \
            Global_Model.F_S_TYPE[feature_senti_pair][relative_pos] / total_pairs
    else:
        snt_lkhd *= 0.5
    if DEBUG:
        print 'SENTIMENT RELATIVE:', feature_senti_pair, snt_lkhd, relative_pos

    return snt_lkhd, wd_dis_type


def cal_feature_sent_score(feature, sentiment, total_pair_occur):
    """Calculate the approved probability of the association of given feature and sentiment
    : Param feature          : feature object
    : Param sentiment        : sentiment object
    : Param total_pair_occur : total pairs occurred in the training data set
    : Returns snt_lkhd       : the likelihood of the association
    : Returns wd_dis_type    : the type of word distance between feature and sentiment
    """
    f_s_pair = feature.token.word + '$' + sentiment.token.word
    s_word, f_word = sentiment.token.word, feature.token.word
    # check if the pair occurred before
    pair_occurred = True
    if f_s_pair not in Local_Model.FS_NUM and f_s_pair not in Global_Model.FS_NUM:
        pair_occurred = False

    # |D(kw-sentiment)| default value is 1
    Nkw = Local_Model.FS_NUM.get(f_s_pair, 1)
    if not pair_occurred and ( f_word in Local_Model.F_S_TYPE):
        # if pair did not occur in neither global model or local model. use feature-class pair instead
        sentiment = Token(sentiment.sntnc,sentiment.phrase,sentiment.wrd_strt_pos,sentiment.wrd_end_pos,Seg_token('sentiment',sentiment.token.flag))
        Nkw = Local_Model.FS_NUM.get(f_word + '$sentiment', 1) / len(Local_Model.F_S_SET[f_word])

    # P(context | kw-sentiment) * P(kw-sentiment)
    snt_lkhd, wd_dis_type = cal_likelihood(feature, sentiment)
    snt_lkhd = snt_lkhd * Nkw / total_pair_occur

    # P{amb}
    if s_word in Global_Model.S_AMB:
        snt_lkhd = snt_lkhd * Global_Model.S_AMB[s_word]
    else:
        snt_lkhd = snt_lkhd * Local_Model.S_AMB.get(s_word, 1)

    # Filter compelled association
    if not feature_senti_ruler.abs_dis(feature, sentiment, Local_Model.CERNTAIN_PAIR):
        if DEBUG:
            print feature.token.word, sentiment.token.word, 'ABS_DIS'
        snt_lkhd = -1
    if feature_senti_ruler.ignore_unseen_senti(Local_Model.KW_DIS, sentiment, feature):
        snt_lkhd = -1

    if DEBUG:
        print s_word, Local_Model.S_AMB.get(sentiment.token.word, 1)
        print 'Sentiment Final Score:', f_s_pair, snt_lkhd, Local_Model.F_S_TYPE.get(f_s_pair, 1)
        print feature_senti_ruler.abs_pos(feature, sentiment)
    # Filter direct association
    if feature_senti_ruler.abs_pos(sentiment, feature):
        return 1, Dis_Type.LESS_THAN_THREE_WORDS

    return snt_lkhd, wd_dis_type


def select_sentiment_word_new(feature_list, sentiment_list, total_pair_occur, obj_poss):
    """Associate feature with best fit sentiment
    : Param feature_list     : a list of feature object
    : Param sentiment_list   : a list of sentiment object
    : Param total_pair_occur : total number of pairs
    : Return rst_fs_pair     : map object. Key -> feature word ; Value -> (feature_object, sentiment_object)
    """
    possible_pairs = [(f, s) for f in feature_list for s in sentiment_list if
                      feature_senti_ruler.obj_feature_close(obj_poss, f) and f.token.flag != u'商品']
    fs_pair = []
    for feature, sentiment in possible_pairs:
        likelihood, wd_dis_type = cal_feature_sent_score(feature, sentiment, total_pair_occur)
        # feature-sentiment pair has not been seen in train set
        if likelihood != -1:
            fs_pair.append(
                (likelihood, Dis_Type.MORE_THAN_THREE_WORDS - wd_dis_type, feature.token.word, sentiment.token.word))

    # append null association
    for feature in feature_list:
        # TODO: try different tactics, problem left
        null_likelihood = 0.5 * 0.5 * 0.3 * 0.5 * Local_Model.F_NULL.get(feature.token.word, 1) / total_pair_occur * ALPHA
        if DEBUG:
            print 'Feature null', \
                feature.token.word, \
                Local_Model.F_NULL.get(feature.token.word, 0.5), \
                'ALPHA:', ALPHA, \
                1 / total_pair_occur, \
                null_likelihood

        fs_pair.append((null_likelihood, 0, feature.token.word, 'null' + feature.token.word))

    # (likelihood,reversed_word_distance,feature,sentiment)
    sorted_fs_pair = sorted(fs_pair, key=operator.itemgetter(0, 1), reverse=True)
    # key -> feature, value -> (Feature_Obj,Senti_Obj)
    rst_fs_pair = {}
    used_sentiment, used_feature = set(), set()
    for likelihood, pair_score, feature, sentiment in sorted_fs_pair:
        if feature not in used_feature and sentiment not in used_sentiment and likelihood != -1:
            rst_fs_pair[feature] = (sentiment, pair_score)
            used_sentiment.add(sentiment)
            used_feature.add(feature)
    return rst_fs_pair


def cal_alpha():
    p_cover = 0.7
    global ALPHA
    ALPHA = (TOTAL_NULL_SENTI + (p_cover - 1) * TRAIN_SET_VOLUME) / (p_cover * TOTAL_NULL_SENTI)


def class_new(infile=TEST_FILE_PAHT, obj_name=OBJ_NAME, model_name=MODEL_FILE_PATH, ):
    """Classify test data
    : Param infile     : input file path
    : Param obj_name   : object name
    : Param model_name : model path
    """
    # TODO: Load Model
    load_mdl(model_name)
    load_glb_mdl('globalmodel1107/global_model.txt')

    entity_class, synonym, sent_dic, degree_dic = load_knw_base()

    # Initial Data
    lns = [ln.decode('utf-8').lower() for ln in open(infile).readlines()]
    total_pair_occur = sum(Local_Model.FS_NUM.values())
    cal_alpha()
    result = Result()

    # Classify Test Data
    for ln in lns:
        ln = re.sub('//@.+', '', ln)
        #        ln = replace_mention([u'@银鹭花生牛奶',u'@林俊杰-银鹭花生牛奶'],OBJ_NAME,ln)
        emoticons = [emoticon for emoticon in re.findall(r"\[.+?\]", ln)]
        can_rst, feature_sent_pairs = {}, {}
        for ln_after_pr_rule in prior_rules.prior_rules(ln, replace_punc=True):
            # Generate word segments list
            kws, obj_poss = seg_ln(ln_after_pr_rule, synonym, obj_name, entity_class)
            if DEBUG:
                print 'Before Combine'
                MyLib.print_seg(kws)
                for kw in kws:
                    print kw.token.word,
                print
            kws = feature_senti_ruler.combine_sentiment(kws, sent_dic, degree_dic)
            if DEBUG:
                print 'After Combine'
                MyLib.print_seg(kws)
                for obj_pos in obj_poss:
                    print 'obj', obj_pos.phrase,
                for kw in kws:
                    print kw.token.word,
                print
            if len(obj_poss) == 0:
                continue

            feature_list = set()
            sentiment_list = set()

            # Loop over all the candidate keywords,
            # select sentiment and feature list
            for kw in kws:
                if kw.token.word == obj_name:
                    continue
                if kw.token.word in entity_class:
                    feature_list.add(kw)
                if kw.token.word in sent_dic:
                    sentiment_list.add(kw)

                # Cal the probability of the distance between the keyword and the object
                max_likelihood = -1
                for obj_pos in obj_poss:
                    # First compare <obj,keyword> pair
                    likelihood, wd_dis_type = cal_likelihood(obj_pos, kw)
                    if likelihood > max_likelihood:
                        max_likelihood = likelihood

                likelihood = max_likelihood

                if can_rst.get(kw.token.word, 0) < likelihood:
                    can_rst[kw.token.word] = likelihood

            tmp_pairs = select_sentiment_word_new(feature_list, sentiment_list, total_pair_occur, obj_poss)
            # Combine Results
            for key, senti_value in tmp_pairs.items():
                if key in feature_sent_pairs and feature_sent_pairs[key][1] > senti_value[1]:
                    continue
                feature_sent_pairs[key] = senti_value
            feature_sent_pairs = dict(list(feature_sent_pairs.items()) + list(tmp_pairs.items()))
        can_rst = sorted(can_rst.items(), key=operator.itemgetter(1), reverse=True)
        MyLib.merge_rst(ln, sent_dic, can_rst, feature_sent_pairs, result)
    MyLib.print_rst(result)






def replace_mention(nick_name_lst, obj_name, ln):
    for nick_name in nick_name_lst:
        ln = ln.replace(nick_name, obj_name)
    return ln


# convert chines punctuation to english punctuation
# 1. period
# 2. exclaim
#


def train_data_clean(infile):
    ad_words = [u'关注', u'转发', u'获取', u'机会', u'赢取', u'推荐'
        , u'活动', u'好友' , u'支持' , u'话题' , u'详情' , u'地址' , u'赢' , u'抽奖' , u'好运' , u'中奖']
    lns = [ln.decode('utf-8') for ln in open(infile).readlines()]
    clean_lns = {}
    for ln in lns:
        ad_counter = 0
        if len(kw_util.regex_mention.sub("", ln).strip()) == 0:
            continue
        ln = re.sub('//@.+', '', ln)
        tmp_ln = kw_util.tweet_filter(ln)
        for ad_word in ad_words:
            if ad_word in ln:
                ad_counter += 1
        if kw_util.regex_url.search(ln) is not None:
            ad_counter += 1

        if ad_counter > 2:
            continue
        if not clean_lns.has_key(tmp_ln):
            clean_lns[tmp_ln] = ln
    for ln in clean_lns.values():
        print ln.strip().encode('utf-8')


def seg_files(infile=TEST_FILE_PAHT, obj_name=OBJ_NAME, usrdic_file='new_words.txt'):
    jieba.load_userdict(usrdic_file)
    stop_dic = {ln.strip() for ln in open(STOP_DIC).readlines()}
    lns = [ln.strip() for ln in open(infile).readlines()]

    for ln in lns:
        print ln
        kws = MyLib.seg_and_filter(kw_util.tweet_filter(ln.decode('utf-8')), obj_name, stop_dic)
        for kw in kws:
            print kw.word + '/' + kw.flag,
        print
        print 'origin segment:'
        kws = list(jieba.posseg.cut(kw_util.tweet_filter(ln.decode('utf-8'))))
        for i in range(len(kws)):
            kws[i].word = kws[i].word.encode('utf-8')
        for kw in kws:
            print kw.word + '/' + kw.flag
        print '\n'


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print """Usage: python FreqBase.py <cmd> <input_file> where
<input_file> line delimited text;
<cmd> = [gen_model|classify|gen_kw_frq|segment]
\tgen_model <input_file> <obj_name> [user_dic], generate model;
\tgen_kw_frq <input_file> <obj_name> [user_dic], generate keyword frequency statistics
\tsegment <infile> <obj_name> [user_dic], segment sentences
\tclassify <infile> <obj_name> <model_file> <kw_freq_file> [user_dic], classify test data

"""

    if sys.argv[1] == 'gen_model':
        if len(sys.argv) > 2:
            gen_model(sys.argv[2])
        else:
            gen_model()
    elif sys.argv[1] == 'classify':
        if len(sys.argv) < 4:
            class_new(sys.argv[2])
        else:
            class_new(sys.argv[2], sys.argv[3].decode('utf-8'), sys.argv[4])
    elif sys.argv[1] == 'segment':
        seg_files(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'test':
        class_new()
    elif sys.argv[1] == 'clean':
        train_data_clean(sys.argv[2])
