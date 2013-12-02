#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import re
from scripts.ruler import prior_rules

__author__ = 'congzicun'
KNWB_PATH = 'knowl-base.txt'
LOCAL_FEATURE = 'local_feature.txt'
from scripts.model.model import *
from scripts.util import MyLib, kw_util

FILE_PREFIX = 'yinlu'
MODEL_FILE_PATH = FILE_PREFIX + '_model.txt'


def load_data_set(filename):
    lns = []
    uniqln = set()
    for ln in open(filename).readlines():
        ln = kw_util.punc_replace(re.sub('//@.+', '', ln.decode('utf-8')))
        ln_seg = kw_util.tweet_filter(ln)
        if ln_seg not in uniqln and len(ln) != 0 and not prior_rules.is_ad(ln):
            uniqln.add(ln_seg)
            lns.append(ln)
    return lns


def _load_stopdic():
    stop_dic = [ln.strip().decode('utf-8') for ln in
                open('/Users/congzicun/Yunio/pycharm/YZKnowl/dictionary/stopwords.txt').readlines()]
    return stop_dic


def load_dic():
    total_dic = {kw.decode('utf-8').strip() for kw in
                 open('/Users/congzicun/Yunio/pycharm/YZKnowl/dictionary/real_final_dic.txt').readlines()}
    return total_dic


def load_tfidf(filename):
    tfidf = {}
    lns = [ln.decode('utf-8').strip() for ln in open(filename).readlines()]
    for ln in lns:
        kw, v = ln.split(' ')
        tfidf[kw] = v
    return tfidf


def load_knw_base(obj_name):
    lns = [ln.decode('utf-8').strip().lower() for ln in open(KNWB_PATH).readlines()]
    knowbase = kb()
    for ln in lns:
        if ln.startswith('#') or len(ln) == 0:
            continue
        entity, instances, classes = ln.split('\t')
        for cls in classes.split('|'):
            if len(cls) == 0:
                continue
            if u'情感词' in cls:
                if u'正面' in cls:
                    knowbase.sentiments[entity] = kb.POSITIVE
                if u'负面' in cls:
                    knowbase.sentiments[entity] = kb.NEGATIVE
                if u'程度' in cls:
                    knowbase.degree.add(entity)
            else:
                knowbase.features[entity] = kb.FEATURE
        knowbase.instances[entity] = entity

        for instance in instances.split('|'):
            if len(instance) != 0:
                knowbase.instances[instance] = entity
    knowbase.instances.setdefault(obj_name, obj_name)
    knowbase.features[obj_name] = knowbase.OBJECT
    knowbase.stop_dic = _load_stopdic()
    return knowbase


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
            Global_Model.FS_NUM.inc(fs_pair, int(value))
        elif ln_elemnts[0] == 'AMBI_DIST'.lower():
            Global_Model.S_AMB[ln_elemnts[1]] = int(value)
        elif ln_elemnts[0] == 'KW_DIST'.lower():
            Global_Model.KW_DIS[ln_elemnts[1]] = int(value)
    for kw in Global_Model.S_AMB.keys():
        if kw in Global_Model.KW_DIS:
            Global_Model.S_AMB[kw] = 1 - Global_Model.S_AMB[kw] / (
                Global_Model.S_AMB[kw] + Global_Model.KW_DIS.get(kw, 0))
    return Global_Model


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
            total_null_doc, total_doc = ln.strip().split('$')
            Local_Model.TOTAL_NULL_SENTI = int(total_null_doc)
            Local_Model.TRAIN_SET_VOLUME = int(total_doc)
        elif feature_type == 6:
            feature, sentiment, count = ln.strip().split('$')
            Local_Model.CERNTAIN_PAIR.inc(feature + '$' + sentiment, int(count))
    return Local_Model


def load_idf():
    idf = {}
    for ln in open('dictionary/idf.txt').readlines():
        kw, value = ln.decode('utf-8').strip().split(' ')
        idf[kw] = float(value)
    return idf


def load_amb():
    amb = {}
    for ln in open('dictionary/amb.txt').readlines():
        kw, value = ln.decode('utf-8').strip().split(' ')
        amb[kw] = float(value)
    return amb