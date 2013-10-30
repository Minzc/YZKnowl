#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'congzicun'
def absPos(senti_wrd,feature):
    """Handle special situation : [Sentiment Word]的[Feature]
    :Param ln       : message
    :Param senti_wrd: sentiment word:
    :Param feature  : feature
    :Returns        : True if the '[sentiment]的[feature]' form exists in the message
    """
    if abs(feature.wrd_strt_pos - senti_wrd.wrd_end_pos) < 2:
        return True
    return False
def absDis(feature,senti_wrd):
    """Decide if the phrase distance between feature and sentiment word is not larger than three
    :Param feature  : feature
    :Param senti_wrd: sentiment word
    :Returns        : True if the distance is less than three phrase
    """
    if abs(feature.phrase - senti_wrd.phrase) + \
       abs(feature.sntnc - senti_wrd.sntnc) > 3:
        return False
    return True