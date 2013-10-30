#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import kw_util
__author__ = 'congzicun'
def prior_rules(ln,replace_punc = True):
    """split enumeration message into independent sub-messages
    :Param ln           : message to be handled
    :Param replace_punc : if need to supersede chinese punctuation
    :Returns [ln]       : messages after filter
    """
    if replace_punc:
        ln = kw_util.punc_replace(ln)
    if ('1.' in ln and '2.' in ln) \
    or ('1,' in ln and '2,' in ln)\
    or ('1:' in ln and '2:' in ln):
        return filter(lambda x : x != '', re.split(r'\d+[.,]',ln))
    return [ln]