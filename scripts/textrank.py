#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
from operator import itemgetter

import jieba.posseg as pseg

from pygraph.classes.graph import graph
from pygraph.classes.digraph import digraph
from pygraph.algorithms.pagerank import pagerank
from pygraph.classes.exceptions import AdditionError
from scripts.util import kw_util


def filter_for_tags(tagged, tags=['n', 'l']):
    return [item for item in tagged if item[1] in tags]


def unique_everseen(iterable, key=None):
    seen = set()
    seen_add = seen.add
    for element in itertools.ifilterfalse(seen.__contains__, iterable):
        #print element
        seen_add(element)
        yield element


def textrank():
    lns = [ln.decode('utf-8').strip() for ln in open('clean_data').readlines()]
    stopwords = {kw_util.tweet_filter(kw_util.punc_replace(ln.decode('utf-8'))) for ln in open('stopwords').readlines()}
    text = '.'.join(lns)
    
    words = pseg.cut(text)
    tagged = []
    textrank = {}
    
    for w in words:
        if w.word not in stopwords:
            tagged.append((w.word, w.flag))
    
    tagged = filter_for_tags(tagged)

    unique_word_set = unique_everseen([x[0] for x in tagged])

    gr = digraph()
    gr.add_nodes(list(unique_word_set))

    window_start = 0
    window_end = 1

    while True:

        window_words = tagged[window_start:window_end]
        if len(window_words) == 1:
            # print window_words[0][0], window_words[1][0]
            try:
                gr.add_edge((window_words[0][0], window_words[1][0]))
            except AdditionError, e:
                print 'already added %s, %s' % ((window_words[0][0].encode('utf-8'), window_words[1][0].encode('utf-8')))
        else:
            break

        window_start += 1
        window_end += 1
    print '###KEYWORDS##'
    index = 0
    calculated_page_rank = pagerank(gr)
    di = sorted(calculated_page_rank.iteritems(), key=itemgetter(1), reverse=True)
    for k, g in itertools.groupby(di, key=itemgetter(1)):
        for word in map(itemgetter(0), g):
            #textrank[word] = k
            if word not in stopwords and len(word) > 1:
                print word.encode('utf-8')
                index += 1
                if index == 51:
                    return

if __name__ == '__main__':
    textrank()
