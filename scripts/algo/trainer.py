#!/usr/bin/python
# -*- coding: utf-8 -*-
from scripts.util import file_loader, MyLib

__author__ = 'congzicun'
import nltk
from scripts.ruler import prior_rules
from scripts.ruler import feature_senti_ruler
from scripts.model.model import *
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


def gen_model(infile=TRAIN_FILE_PAHT, obj_name=OBJ_NAME):
    """Given training data set, generate Model file
    : Param infile   : training data file path
    : Param obj_name : object name
    """
    entity_class, synonym, senti_dic, degree_dic = file_loader.load_knw_base()
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
                if kw_pair == u'夺冠$开心':
                    print '#'*10 , ln.encode('utf-8')
                Local_Model.FS_NUM.inc(kw_pair)
                Local_Model.FS_NUM.inc(instance_class_pair)
                # Cerntain Pair Feature
                if feature_senti_ruler.abs_pos(kw1, kw2):
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
    for kw_pair, count in Local_Model.CERNTAIN_PAIR.items():
        # Magic number
        if count > 5:
            print kw_pair.encode('utf-8') + '$' + str(count)
