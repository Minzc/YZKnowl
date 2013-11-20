#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'congzicun'
import jieba
import nltk
import re
from scripts.util import kw_util
from scripts.model.model import *
DEBUG = True

def combine_kw(kw_w_lst, obj_name):
    combined_w = []
    kw_combined = False
    for i in range(len(kw_w_lst) - 1, 0, -1):
        if kw_w_lst[i].word.encode('utf-8') == obj_name:
            kw_w_lst[i].flag = 'obj'
        if kw_combined:
            kw_combined = False
            continue
        word = kw_w_lst[i].word
        flag = kw_w_lst[i].flag

        if len(kw_w_lst[i - 1].word) == 1 and len(kw_w_lst[i].word) == 1:
            kw_combined = True
            if 'v' in kw_w_lst[i - 1].flag and 'ul' in kw_w_lst[i].flag:
                word = kw_w_lst[i - 1].word + kw_w_lst[i].word
                flag = 'a'
            elif 'a' in kw_w_lst[i - 1].flag and 'ul' in kw_w_lst[i].flag:
                word = kw_w_lst[i - 1].word + kw_w_lst[i].word
                flag = 'a'
            elif 'v' in kw_w_lst[i - 1].flag and 'v' in kw_w_lst[i].flag:
                word = kw_w_lst[i - 1].word + kw_w_lst[i].word
                flag = 'v'
            elif 'a' in kw_w_lst[i - 1].flag and 'v' in kw_w_lst[i].flag:
                word = kw_w_lst[i - 1].word + kw_w_lst[i].word
                flag = 'v'
            elif 'd' in kw_w_lst[i - 1].flag and 'v' in kw_w_lst[i].flag:
                word = kw_w_lst[i - 1].word + kw_w_lst[i].word
                flag = 'v'
            elif 'n' in kw_w_lst[i - 1].flag and 'ul' in kw_w_lst[i].flag:
                word = kw_w_lst[i - 1].word + kw_w_lst[i].word
                flag = 'a'
            elif 'z' in kw_w_lst[i - 1].flag and 'v' in kw_w_lst[i].flag:
                word = kw_w_lst[i - 1].word + kw_w_lst[i].word
                flag = 'a'
            elif 'd' in kw_w_lst[i - 1].flag and 'n' in kw_w_lst[i].flag:
                word = kw_w_lst[i - 1].word + kw_w_lst[i].word
                flag = 'a'
            elif 'd' in kw_w_lst[i - 1].flag and 'a' in kw_w_lst[i].flag:
                word = kw_w_lst[i - 1].word + kw_w_lst[i].word
                flag = 'a'
            elif 'a' in kw_w_lst[i - 1].flag and 'q' in kw_w_lst[i].flag:
                word = kw_w_lst[i - 1].word + kw_w_lst[i].word
                flag = 'a'
            else:
                kw_combined = False
        kw_w_lst[i].word = word
        kw_w_lst[i].flag = flag
        combined_w.append(kw_w_lst[i])

    if not kw_combined:
        if kw_w_lst[0].word.encode('utf-8') == obj_name:
            kw_w_lst[0].flag = 'obj'
        combined_w.append(kw_w_lst[0])
    return reversed(combined_w)


def seg_and_filter(ln, obj_name, stop_dic):
    kws = [kw for kw in combine_kw(list(jieba.posseg.cut(ln)), obj_name)
           if not (len(kw.word) == 1
                   or kw.word.encode('utf-8') in stop_dic
                   or 'f' in kw.flag
                   or 'q' in kw.flag
                   or 'm' in kw.flag
                   or 't' in kw.flag
                   or 'd' in kw.flag
                   or 'o' == kw.flag
                   or 'nz' == kw.flag)]
    for i in range(len(kws)):
        kws[i].word = kws[i].word.encode('utf-8')
    return kws


def create_and_init_frqdis(*argvs):
    tmp = nltk.FreqDist()
    for argv in argvs:
        tmp.inc(argv)
    return tmp


def print_seg(kws):
    for kw in kws:
        print kw.token.word + '/' + kw.token.flag + '/' + str(kw.sntnc), str(kw.wrd_strt_pos), str(kw.wrd_end_pos)
    print


def print_rst(result):
    print '1. summary                           |general-information|'
    print '2. positive opinion tuples           |pos-tags|'
    print '3. negative opinion tuples           |neg-tags|'
    print '4. feature sentiment tuples          |feature-sentiment|'
    print '5. detail analysis                   |detail-analysis|'
    print
    print '==========================================================='
    print '1. summary                            *general-information*'
    print
    print 'Total Tweets:', result.TOTAL_TWEETS
    print 'Positive Tweets:', sum(result.POS_TAGS.values())
    print 'Negative Tweets:', sum(result.NEG_TAGS.values())
    print
    print '==========================================================='
    print '2. positive opinion tuples                       *pos-tags*'
    print
    for sentiment, count in list(result.POS_TAGS.items())[:5]:
        print '#' + sentiment.encode('utf-8') + '(' + str(count) + ')' + '#'
        for feature, count in list(result.POS_FEATURE[sentiment].items())[:5]:
            print '|' + feature.encode('utf-8') + '-' + sentiment.encode('utf-8') + '|\t',
        print '\n'
    print
    print '==========================================================='
    print '3. negative opinion tuples                       *neg-tags*'
    print
    for sentiment, count in list(result.NEG_TAGS.items())[:5]:
        print '#' + sentiment.encode('utf-8') + '(' + str(count) + ')' + '#'
        for feature, count in list(result.NEG_FEATURE[sentiment].items())[:5]:
            print '|' + feature.encode('utf-8') + '-' + sentiment.encode('utf-8') + '|\t',
        print '\n'
    print
    print '==========================================================='
    print '4. feature sentiment tuples             *feature-sentiment*'
    print
    for feature, count in list(result.FEATURE.items())[:5]:
        print '#' + feature.encode('utf-8') + '(' + str(count) + ')' + '#'
        if feature in result.FEATURE_POS:
            for sentiment, count in list(result.FEATURE_POS[feature].items())[:5]:
                print '|' + feature.encode('utf-8') + '-' + sentiment.encode('utf-8') + '|\t',
            print '\n'
        if feature in result.FEATURE_NEG:
            for sentiment, count in list(result.FEATURE_NEG[feature].items())[:5]:
                print '|' + feature.encode('utf-8') + '-' + sentiment.encode('utf-8') + '|\t',
            print '\n'
    print
    print '==========================================================='
    print '5. detail analysis                        *detail-analysis*'
    print
    for tag_tuple, tws in result.TAG_TWEETS.items():
        feature, sentiment = tag_tuple.split('$')
        print '*' + feature.encode('utf-8') + '-' + sentiment.encode('utf-8') + '*'
        print
        for tw in tws:
            print tw.encode('utf-8')
        print '-----------------------------------------------------------'


def merge_rst(ln, sent_dic, feature_rst, feature_sent_pairs, result, synonym):
    feature = ''
    counter = 0
    for f, score in feature_rst:
        counter += 1
        if f in feature_sent_pairs and 'null' not in feature_sent_pairs[f][0]:
            feature = f
            break

    if feature == '':
        return
    sentiment = synonym.get(feature_sent_pairs[feature][0], '')
    origin_senti = feature_sent_pairs[feature][0]
    result.FEATURE.inc(feature)
    if sentiment not in sent_dic:
        sentiment = ''
        origin_senti = ''
        result.POS_FEATURE.setdefault(sentiment, nltk.FreqDist())
        result.POS_FEATURE[sentiment].inc(feature)
        result.FEATURE_POS.setdefault(feature, nltk.FreqDist())
        result.FEATURE_POS[feature].inc(sentiment)
        result.POS_TAGS.inc(sentiment)
    elif sent_dic[sentiment] == 'p':
        result.POS_FEATURE.setdefault(origin_senti, nltk.FreqDist())
        result.POS_FEATURE[origin_senti].inc(feature)
        result.FEATURE_POS.setdefault(feature, nltk.FreqDist())
        result.FEATURE_POS[feature].inc(origin_senti)
        result.POS_TAGS.inc(origin_senti)
    elif sent_dic[sentiment] == 'n':
        result.FEATURE_NEG.setdefault(feature, nltk.FreqDist())
        result.FEATURE_NEG[feature].inc(origin_senti)
        result.NEG_FEATURE.setdefault(origin_senti, nltk.FreqDist())
        result.NEG_FEATURE[origin_senti].inc(feature)
        result.NEG_TAGS.inc(origin_senti)

    result.TAG_TWEETS.setdefault(feature + '$' + origin_senti, [])
    result.TAG_TWEETS[feature + '$' + origin_senti].append(ln)
    result.TOTAL_TWEETS += 1


def seg_ln(ln, synonym, obj_name, entity_class):
    know_dic = set(synonym.keys())
    know_dic.add(obj_name)
    sub_sents = filter(lambda x: x != '', re.split(ur'[!.?…~;"#]', kw_util.punc_replace(ln)))
    kws, obj_poss = [], []
    pre_phrases_len = 0
    phrase_position = 0
    for index, sub_sent in enumerate(sub_sents):
        if DEBUG:
            print sub_sent.encode('utf-8')
        for phrase in kw_util.tweet_filter(sub_sent).strip().split(' '):
            if len(phrase) < 1 or len(phrase) > 30:
                continue
            kw_poses = kw_util.backward_maxmatch(phrase, know_dic, 100, 1)
            for kw_pos in kw_poses:
                start, end, abs_word_start, abs_word_end = kw_pos[0], kw_pos[1], pre_phrases_len + kw_pos[
                    0], pre_phrases_len + kw_pos[1]
                keyword = phrase[start:end]
                if synonym[phrase[start:end]] == obj_name:
                    obj_token = Token(index, phrase_position, abs_word_start, abs_word_end,
                                      Seg_token(synonym[keyword], 'obj', keyword))
                    kws.append(obj_token)
                    obj_poss.append(obj_token)
                else:
                    # TODO: hard code
                    flag = 'sentiment'
                    if synonym[keyword] in entity_class:
                        flag = entity_class[synonym[keyword]][0]
                        # TODO: flag 需要细化
                    kws.append(
                        Token(index, phrase_position, abs_word_start, abs_word_end, Seg_token(synonym[keyword], flag, keyword)))
            pre_phrases_len += len(phrase)
            phrase_position += 1
    return kws, obj_poss


def decide_dis_feature_type(kw1, kw2):
    """Give two keywords, generate features between the given words
    : Param kw1               : keyword one
    : Param kw1               : keyword two
    : Returns wd_dis_type     : distance type in word level
    : Returns snt_dis_type    : distance type in phrase level
    : Returns phrase_dis_type : disatnce type in sentence level
    : Returns relative_pos    : relative position type
    """
    # absolute word distance
    word_distance = kw1.wrd_strt_pos - kw2.wrd_end_pos
    if word_distance < 0:
        word_distance = kw2.wrd_strt_pos - kw1.wrd_end_pos
        # Distance Feature
    wd_dis_type, snt_dis_type, phrase_dis_type, relative_pos = \
        Dis_Type.MORE_THAN_THREE_WORDS, \
        Dis_Type.MORE_THAN_ONE_SENT, \
        Dis_Type.MORE_THAN_FOUR_PHRASE, \
        Dis_Type.PRIOR
    # Sentence Distance Feature
    if abs(kw1.sntnc - kw2.sntnc) < 2:
        snt_dis_type = Dis_Type.LESS_THAN_ONE_SENT
        # Word Distance Feature
    if abs(word_distance / 2) < 1:
        wd_dis_type = Dis_Type.LESS_THAN_ONE_WORDS
    elif abs(word_distance / 2) < 3:
        wd_dis_type = Dis_Type.LESS_THAN_THREE_WORDS
        # Phrase Distance Feature
    if abs(kw1.phrase - kw2.phrase) < 2:
        phrase_dis_type = Dis_Type.LESS_THAN_TWO_PHRASE
    elif abs(kw1.phrase - kw2.phrase) < 4:
        phrase_dis_type = Dis_Type.LESS_THAN_FOUR_PHRASE
    if kw1.wrd_strt_pos > kw2.wrd_strt_pos:
        relative_pos = Dis_Type.POSTERIOR
    return wd_dis_type, snt_dis_type, phrase_dis_type, relative_pos