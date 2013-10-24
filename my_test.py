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
import FreqBase
if __name__ == '__main__':
    lns = [ln for ln in open('testCodeCrrct.txt')]
    for ln in lns:
        print ln.decode('utf-8')
        segs = FreqBase.punc_replace(ln)
        print segs
        for seg in re.split(r"[!.?]",segs):
            print seg
