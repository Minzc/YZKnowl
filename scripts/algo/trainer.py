#!/usr/bin/python
# -*- coding: utf-8 -*-
from scripts.util import file_loader, MyLib

__author__ = 'congzicun'
import nltk
from scripts.model.model import *
from scripts.ruler import prior_rules
from scripts.ruler import fs_ruler
from scripts.util.MyLib import seg_ln
from scripts.util.MyLib import decide_dis_feature_type
import re

FILE_PREFIX = 'yinlu'
OBJ_NAME = u'银鹭花生牛奶'
TRAIN_FILE_PAHT = FILE_PREFIX + '_train_data.txt'
DEBUG = False


def if_ambiguity(kw1, kw2, sentiment_dic):
    """Judge if kw1 is an ambiguous word
    :param kw1: keyword to be judged
    :param kw2: accompanied keyword
    :param sentiment_dic: sentiment dictionary
    :Returns  : True if kw1 is not ambiguous
    """
    if kw1.phrase == kw2.phrase \
        and kw2.token.word in sentiment_dic \
        and kw1.token.word in sentiment_dic \
            and kw1.token.word != kw2.token.word:
        return False
    return True


def set_dft_mdl_value(pair, local_model):
    local_model.F_S_TYPE.setdefault(pair,
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


def stat_info(f_null_docs, kw_distr, kws, senti_dic, sentiment_ambi, local_model):
    for kw1 in kws:
        not_ambiguity = True
        kw_distr.inc(kw1.token.word)
        if_has_senti = False

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
            not_ambiguity &= if_ambiguity(kw1, kw2, senti_dic)
            if_has_senti |= (kw2.token.word in senti_dic)

            kw_pair = kw1.token.word + '$' + kw2.token.word
            local_model.FS_NUM.inc(kw_pair)
            local_model.FS_NUM.inc(instance_class_pair)
            # Certain Pair Feature
            if fs_ruler.abs_pos(kw1, kw2):
                local_model.CERNTAIN_PAIR.inc(kw_pair)
                # Distance Feature
            wd_dis_type, snt_dis_type, phrase_dis_type, relative_pos \
                = decide_dis_feature_type(kw1, kw2)
            set_dft_mdl_value(kw_pair, local_model)
            set_dft_mdl_value(instance_class_pair, local_model)
            set_dft_mdl_value(class_instance_pair, local_model)

            local_model.F_S_TYPE[kw_pair].inc(wd_dis_type)
            local_model.F_S_TYPE[kw_pair].inc(snt_dis_type)
            local_model.F_S_TYPE[kw_pair].inc(phrase_dis_type)
            local_model.F_S_TYPE[kw_pair].inc(relative_pos)

            if kw2.token.word in senti_dic:
                local_model.F_S_TYPE[instance_class_pair].inc(wd_dis_type)
                local_model.F_S_TYPE[instance_class_pair].inc(snt_dis_type)
                local_model.F_S_TYPE[instance_class_pair].inc(phrase_dis_type)
                local_model.F_S_TYPE[instance_class_pair].inc(relative_pos)
            if kw1.token.word not in senti_dic:
                local_model.F_S_TYPE[class_instance_pair].inc(wd_dis_type)
                local_model.F_S_TYPE[class_instance_pair].inc(snt_dis_type)
                local_model.F_S_TYPE[class_instance_pair].inc(phrase_dis_type)
                local_model.F_S_TYPE[class_instance_pair].inc(relative_pos)
                # Sentiment Ambiguity
        if not_ambiguity:
            sentiment_ambi.inc(kw1.token.word)
            # Null Sentiment Feature
        if not if_has_senti:
            f_null_docs.inc(kw1.token.word)


def gen_model(infile=TRAIN_FILE_PAHT, obj_name=OBJ_NAME):
    """Given training data set, generate Model file
    : Param infile   : training data file path
    : Param obj_name : object name
    """
    local_model = Local_Model()
    entity_class, synonym, senti_dic, degree_dic = file_loader.load_knw_base()
    lns = [ln.decode('utf-8').lower() for ln in open(infile).readlines()]
    sentiment_ambi = nltk.FreqDist()
    f_null_docs = nltk.FreqDist()
    kw_distr = nltk.FreqDist()
    total_docs = 0
    total_null_docs = 0
    for ln in lns:
        ln = re.sub('//@.+', '', ln)
        kwnum = 0
        # Format ln by prior rules
        if_has_senti = False
        for ln_seg in prior_rules.prior_rules(ln):
            kws, obj_poss = seg_ln(ln_seg, synonym, obj_name, entity_class)
            kws = fs_ruler.combine_sentiment(kws, senti_dic, degree_dic)
            if len(kws) != 0:
                total_docs += 1
                kwnum += len(kws)
            # Start Statistic
            for kw in kws:
                if synonym[kw.token.word] in senti_dic:
                    if_has_senti = True

            stat_info(f_null_docs, kw_distr, kws, senti_dic, sentiment_ambi, local_model)
        if not if_has_senti and kwnum != 0:
            total_null_docs += 1

    # Output Model File
    print '#FEATURE_SENTI_TYPE_DISTR'
    for kw_pair, type_dises in local_model.F_S_TYPE.items():
        for dis_type, value in type_dises.items():
            print kw_pair.encode('utf-8') + '$' + str(dis_type) + '$' + str(value)
    print '#FEATURE_SENTI_DIST'
    for kw, count in local_model.FS_NUM.items():
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
    for kw_pair, count in local_model.CERNTAIN_PAIR.items():
        # Magic number
        if count > 5:
            print kw_pair.encode('utf-8') + '$' + str(count)
