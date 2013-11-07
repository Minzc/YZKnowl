#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'congzicun'

NEGATIVE_EMOTICONS = ['[泪]','[抓狂]','[哼]','[鄙视]','[怒]','[威武]','[弱]']
POSITIVE_EMOTICONS = ['[嘻嘻]','[鼓掌]','[偷笑]','[太开心]','[哈哈]']
def abs_pos(senti_wrd,feature):
    """Handle special situation : [Sentiment Word]的[Feature]
    :Param ln       : message
    :Param senti_wrd: sentiment word:
    :Param feature  : feature
    :Returns        : True if the '[sentiment]的[feature]' form exists in the message
    """
    if feature.wrd_end_pos - senti_wrd.wrd_strt_pos == 0 and feature.phrase == senti_wrd.phrase:
        return True
    if senti_wrd.wrd_end_pos - feature.wrd_strt_pos == 0 and feature.phrase == senti_wrd.phrase:
        return True
    return False


def abs_dis(feature,senti_wrd,CERNTAIN_PAIR):
    """Decide if the phrase distance between feature and sentiment word is not larger than three
    :Param feature  : feature
    :Param senti_wrd: sentiment word
    :Returns        : True if the distance is less than three phrase
    """
    threshold = 2
    if feature.phrase < senti_wrd.phrase:
        threshold = 3
    word_dis = senti_wrd.wrd_strt_pos - feature.wrd_end_pos
    if word_dis < 0:
        word_dis = feature.wrd_strt_pos - senti_wrd.wrd_end_pos
    if CERNTAIN_PAIR.has_key(feature.token.word+'$'+senti_wrd.token.word):
        threshold += 1
    if abs(feature.phrase - senti_wrd.phrase) + \
       abs(feature.sntnc - senti_wrd.sntnc) + word_dis/7> threshold:
        return False
    return True

def if_rep_due_amb(senti_candi,best_fit_senti,SENTI_AMBI):
    for senti_kw in senti_candi:
        if senti_kw.phrase == best_fit_senti.phrase and \
           SENTI_AMBI.get(senti_kw.token.word,0.5) > SENTI_AMBI.get(best_fit_senti.token.word,0.5):
          best_fit_senti = senti_kw
    return best_fit_senti

def ignore_unseen_senti(KW_DIS,sentiment,feature):
    if sentiment.token.word not in KW_DIS \
    and sentiment.phrase != feature.phrase:
        return True
    return False

def ignore_feature(KW_PAIR_DISTR,feature,sentiment,wd_dis_type,snt_dis_type,phrase_dis_type,relative_pos):
    high_level_pair = feature.token.word+'$sentiment'
    if not KW_PAIR_DISTR.has_key(high_level_pair):
        high_level_pair = 'feature$'+sentiment.token.word
    if not KW_PAIR_DISTR.has_key(high_level_pair):
        high_level_pair = 'feature$sentiment'


def conbime_sentiment(kws,senti_dic):
    pre_kw = None
    removed_kw = []
    for kw in kws:
        if kw.token.word in senti_dic \
        and pre_kw != None\
        and pre_kw.token.word in senti_dic\
        and kw.wrd_strt_pos == pre_kw.wrd_end_pos\
        and pre_kw.phrase == kw.phrase:
            removed_kw.append(pre_kw)
        pre_kw = kw
    for kw in removed_kw:
        kws.remove(kw)
    return kws


def obj_feature_close(obj_poss,feature):
    smallest_dis = 10
    threashold = 4
    for obj_pos in obj_poss:
        tmp_dis = abs(obj_pos.phrase - feature.phrase)
        if tmp_dis < smallest_dis:
            smallest_dis = tmp_dis
    if smallest_dis > threashold:
        return False
    return True