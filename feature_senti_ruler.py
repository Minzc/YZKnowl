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
    if feature.wrd_strt_pos - senti_wrd.wrd_end_pos == 0:
        return True
    if senti_wrd.wrd_end_pos - feature.wrd_strt_pos == 0:
        return True
    return False


def abs_dis(feature,senti_wrd):
    """Decide if the phrase distance between feature and sentiment word is not larger than three
    :Param feature  : feature
    :Param senti_wrd: sentiment word
    :Returns        : True if the distance is less than three phrase
    """
    threshold = 2
    if feature.phrase < senti_wrd.phrase:
        threshold = 3
    if abs(feature.phrase - senti_wrd.phrase) + \
       abs(feature.sntnc - senti_wrd.sntnc) > threshold:
        return False
    return True

def if_rep_due_amb(senti_candi,best_fit_senti,SENTI_AMBI):
    for senti_kw in senti_candi:
        if senti_kw.phrase == best_fit_senti.phrase and \
           SENTI_AMBI.get(senti_kw.token.word,0.5) > SENTI_AMBI.get(best_fit_senti.token.word,0.5):
          best_fit_senti = senti_kw
    return best_fit_senti

def obj_feature_close(obj_poss,feature):
    smallest_dis = 10
    threashold = 4
    for obj_pos in obj_poss:
        tmp_dis = abs(obj_pos.phrase - feature.phrase)
        if tmp_dis > smallest_dis:
            smallest_dis = tmp_dis
    if smallest_dis > threashold:
        return False
    return True