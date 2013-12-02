#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import nltk
from scripts.algo import local_kw_ext
from scripts.model.model import Local_Model, Dis_Type
from scripts.util import file_loader, kw_util, MyLib

__author__ = 'congzicun'


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


def stat(tokens, localmodel, kb):
    tokenpairs = [(token1, token2) for token1 in tokens for token2 in tokens if token1 is not token2]
    used_pair = set()
    for token1, token2 in tokenpairs:
        feature = MyLib.dcd_ds_ftr_type(token1, token2)
        pair = token1.keyword + '$' + token2.keyword
        tmppair = token1.origin + '$' + token2.origin
        increment = 1
        if token1.phrspos == token2.phrspos:
            increment = 1

        localmodel.F_S_TYPE.setdefault(pair, nltk.FreqDist())
        localmodel.F_S_TYPE[pair].inc(feature.word_dis, increment)
        localmodel.F_S_TYPE[pair].inc(feature.phrs_dis, increment)
        localmodel.F_S_TYPE[pair].inc(feature.snt_dis, increment)
        localmodel.F_S_TYPE[pair].inc(feature.rltv_dis, increment)
        if tmppair in used_pair:
            continue
        used_pair.add(tmppair)
        localmodel.FS_NUM.inc(pair, increment)

    has_sentiment = False
    for token in tokens:
        localmodel.KW_DIS.inc(token.keyword)
        if token.keyword in kb.sentiments:
            has_sentiment = True
    if not has_sentiment:
        localmodel.TOTAL_NULL_SENTI += 1


def output_mdl(local_model):
# Output Model File
    print '#FEATURE_SENTI_TYPE_DISTR'
    for kw_pair, type_dises in local_model.F_S_TYPE.items():
        for dis_type, value in type_dises.items():
            print kw_pair.encode('utf-8') + '$' + str(dis_type) + '$' + str(value)
    print '#FEATURE_SENTI_DIST'
    for kw, count in local_model.FS_NUM.items():
        print kw.encode('utf-8') + '$' + str(count)
        # print '#AMBI_DIST'
    # for kw, count in sentiment_ambi.items():
    #     print kw.encode('utf-8') + '$' + str(count + 1) + '$' + str(kw_distr.get(kw) + 2)
    # print '#FEATURE_NULL'
    # for kw, count in f_null_docs.items():
    #     print kw.encode('utf-8') + '$' + str(count + 1) + '$' + str(kw_distr.get(kw) + 2)
    print '#KW_DIS'
    for kw, count in local_model.KW_DIS.items():
        print kw.encode('utf-8') + '$' + str(count)
    print '#TOTAL_DOC'
    print str(local_model.TOTAL_NULL_SENTI) + '$' + str(local_model.TRAIN_SET_VOLUME)
    print '#CERTAIN_PAIR_DIS'
    for kw_pair, count in local_model.CERNTAIN_PAIR.items():
        # Magic number
        if count > 5:
            print kw_pair.encode('utf-8') + '$' + str(count)


def train(filename, objname):
    localmodel = Local_Model()
    dataset = file_loader.load_data_set(filename)
    dic = file_loader.load_dic()
    kb = file_loader.load_knw_base(objname)
    dic |= set(kb.instances.keys())
    tfidf_scores = local_kw_ext.extr_kw(dataset, kb, objname, dic)
    for tw in dataset:
        sentences = re.split(ur'[!.?…~;"#:— ]', kw_util.punc_replace(tw))
        for sen_index, sentence in enumerate(sentences):
            if len(sentence.strip()) == 0:
                continue
            localmodel.TRAIN_SET_VOLUME += 1
            tokenlst = MyLib.filter_sentiment(MyLib.seg_token(sentence, dic, kb), kb)
            tokenlst = [token for token in tokenlst if token.keyword in kb.instances]
            stat(tokenlst, localmodel, kb)
    output_mdl(localmodel)
