#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
__author__ = 'congzicun'
from scripts.ruler import fs_ruler
from scripts.util import MyLib
from scripts.ruler import prior_rules
from scripts.model.model import *
from scripts.util import file_loader
from scripts.util import kw_util
from scripts.util.MyLib import decide_dis_feature_type
import operator
import re

ALPHA = 1
DEBUG = False

STOP_DIC = 'stopwords.txt'
FILE_PREFIX = 'yinlu'
TEST_FILE_PAHT = FILE_PREFIX + '_test_data.txt'
MODEL_FILE_PATH = FILE_PREFIX + '_model.txt'
OBJ_NAME = u'银鹭花生牛奶'


def cal_likelihood(kw, feature):
    snt_lkhd = 1
    wd_dis_type, snt_dis_type, phrase_dis_type, relative_pos \
        = decide_dis_feature_type(kw, feature)
    feature_senti_pair = kw.token.word + '$' + feature.token.word
    # P(c=dis|kw-sentiment)
    # Word Distance Feature
    if feature_senti_pair in Local_Model.F_S_TYPE:
        if DEBUG:
            print 'Local Model'
        total_pairs = \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_ONE_WORDS] + \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_THREE_WORDS] + \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.MORE_THAN_THREE_WORDS]
        snt_lkhd *= Local_Model.F_S_TYPE[feature_senti_pair][wd_dis_type] / total_pairs
    elif feature_senti_pair in Global_Model.F_S_TYPE:
        if DEBUG:
            print 'Global Model'
        total_pairs = \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_ONE_WORDS] + \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_THREE_WORDS] + \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.MORE_THAN_THREE_WORDS]
        snt_lkhd = Global_Model.F_S_TYPE[feature_senti_pair][wd_dis_type] / total_pairs
    else:
        snt_lkhd *= 0.3
    if DEBUG:
        print 'SENTIMENT WORD:', feature_senti_pair, snt_lkhd, wd_dis_type
        # Sentence Distance Feature
    if feature_senti_pair in Local_Model.F_S_TYPE:
        if DEBUG:
            print 'Local Model'
        total_pairs = \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_ONE_SENT] + \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.MORE_THAN_ONE_SENT]
        snt_lkhd *= \
            Local_Model.F_S_TYPE[feature_senti_pair][snt_dis_type] / total_pairs
    elif feature_senti_pair in Global_Model.F_S_TYPE:
        if DEBUG:
            print 'Global Model'
        total_pairs = \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_ONE_SENT] + \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.MORE_THAN_ONE_SENT]
        snt_lkhd *= \
            Global_Model.F_S_TYPE[feature_senti_pair][snt_dis_type] / total_pairs
    else:
        snt_lkhd *= 0.5
    if DEBUG:
        print 'SENTIMENT SENTENCE:', feature_senti_pair, snt_lkhd, snt_dis_type
        # Phrase Distance Feature
    if feature_senti_pair in Local_Model.F_S_TYPE:
        if DEBUG:
            print 'Local Model'
        total_pairs = \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_TWO_PHRASE] + \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_FOUR_PHRASE] + \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.MORE_THAN_FOUR_PHRASE]
        snt_lkhd *= \
            Local_Model.F_S_TYPE[feature_senti_pair][phrase_dis_type] / total_pairs
    elif feature_senti_pair in Global_Model.F_S_TYPE:
        if DEBUG:
            print 'Global Model'
        total_pairs = \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_TWO_PHRASE] + \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.LESS_THAN_FOUR_PHRASE] + \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.MORE_THAN_FOUR_PHRASE]
        snt_lkhd *= \
            Global_Model.F_S_TYPE[feature_senti_pair][phrase_dis_type] / total_pairs
    else:
        snt_lkhd *= 0.3
    if DEBUG:
        print 'SENTIMENT PHRASE:', feature_senti_pair, snt_lkhd, phrase_dis_type
        # Relative Position Feature
    if feature_senti_pair in Local_Model.F_S_TYPE:
        if DEBUG:
            print 'Local Model'
        total_pairs = \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.PRIOR] + \
            Local_Model.F_S_TYPE[feature_senti_pair][Dis_Type.POSTERIOR]
        snt_lkhd *= \
            Local_Model.F_S_TYPE[feature_senti_pair][relative_pos] / total_pairs
    elif feature_senti_pair in Global_Model.F_S_TYPE:
        if DEBUG:
            print 'Global Model'
        total_pairs = \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.PRIOR] + \
            Global_Model.F_S_TYPE[feature_senti_pair][Dis_Type.POSTERIOR]
        snt_lkhd *= \
            Global_Model.F_S_TYPE[feature_senti_pair][relative_pos] / total_pairs
    else:
        snt_lkhd *= 0.5
    if DEBUG:
        print 'SENTIMENT RELATIVE:', feature_senti_pair, snt_lkhd, relative_pos

    return snt_lkhd, wd_dis_type


def cal_feature_sent_score(feature, sentiment, total_pair_occur):
    """Calculate the approved probability of the association of given feature and sentiment
    : Param feature          : feature object
    : Param sentiment        : sentiment object
    : Param total_pair_occur : total pairs occurred in the training data set
    : Returns snt_lkhd       : the likelihood of the association
    : Returns wd_dis_type    : the type of word distance between feature and sentiment
    """
    f_s_pair = feature.token.word + '$' + sentiment.token.word
    s_word, f_word = sentiment.token.word, feature.token.word
    # check if the pair occurred before
    pair_occurred = True
    if f_s_pair not in Local_Model.FS_NUM and f_s_pair not in Global_Model.FS_NUM:
        pair_occurred = True

    # |D(kw-sentiment)| default value is 1
    Nkw = Local_Model.FS_NUM.get(f_s_pair, 1)
    if not pair_occurred and (f_word in Local_Model.F_S_SET):
        # if pair did not occur in neither global model or local model. use feature-class pair instead
        sentiment = Token(sentiment.sntnc, sentiment.phrase, sentiment.wrd_strt_pos, sentiment.wrd_end_pos,
                          Seg_token('sentiment', sentiment.token.flag, sentiment.token.origin))
        Nkw = Local_Model.FS_NUM.get(f_word + '$sentiment', 1) / len(Local_Model.F_S_SET[f_word])

    # P(context | kw-sentiment) * P(kw-sentiment)
    snt_lkhd, wd_dis_type = cal_likelihood(feature, sentiment)
    snt_lkhd = snt_lkhd * Nkw / total_pair_occur

    # P{amb}
    snt_lkhd = snt_lkhd * min(Global_Model.S_AMB.get(s_word, 1), Local_Model.S_AMB.get(s_word, 1))

    # Filter compelled association
    if not fs_ruler.abs_dis(feature, sentiment, Local_Model.CERNTAIN_PAIR):
        if DEBUG:
            print feature.token.word, sentiment.token.word, 'ABS_DIS'
        snt_lkhd = -1
    if fs_ruler.ignore_unseen_senti(Local_Model.KW_DIS, sentiment, feature):
        snt_lkhd = -1

    if DEBUG:
        print s_word, Global_Model.S_AMB.get(sentiment.token.word, 1)
        print 'Sentiment Final Score:', f_s_pair, snt_lkhd, Local_Model.F_S_TYPE.get(f_s_pair, 1)
        print fs_ruler.abs_pos(feature, sentiment)
        print 'abs pos', fs_ruler.abs_pos(sentiment, feature), sentiment.wrd_end_pos - feature.wrd_strt_pos < 2, \
            feature.phrase == sentiment.phrase
        # Filter direct association
    if fs_ruler.abs_pos(sentiment, feature) and Global_Model.S_AMB.get(sentiment.token.word, 1) > 0.3:
        snt_lkhd *= 2
        wd_dis_type = Dis_Type.LESS_THAN_THREE_WORDS
    return snt_lkhd, wd_dis_type


def fs_select2(feature_list, sentiment_list, total_pair_occur, obj_poss, tfidf):
    possible_pairs = [(f, s) for f in feature_list for s in sentiment_list if
                      fs_ruler.obj_feature_close(obj_poss, f) and f.token.flag != u'商品']
    fs_pair = []
    for feature, sentiment in possible_pairs:
        if feature.sntnc == sentiment.sntnc:
            word_distance = feature.wrd_strt_pos - sentiment.wrd_end_pos
            if word_distance < 0:
                word_distance = sentiment.wrd_strt_pos - feature.wrd_end_pos
            print feature.token.word.encode('utf-8'), sentiment.token.origin.encode('utf-8'), word_distance
            fs_pair.append((100 - word_distance, tfidf.get(sentiment.token.origin, 0), feature.token.origin, sentiment.token.origin))

    # (likelihood,reversed_word_distance,feature,sentiment)
    sorted_fs_pair = sorted(fs_pair, key=operator.itemgetter(0, 1), reverse=True)
    # key -> feature, value -> (Feature_Obj,Senti_Obj)
    return fs_cmp(sorted_fs_pair)


def fs_cmp(sorted_fs_pair):
    rst_fs_pair = {}
    used_sentiment, used_feature = set(), set()
    for likelihood, pair_score, feature, sentiment in sorted_fs_pair:
        if feature not in used_feature and sentiment not in used_sentiment and likelihood != -1:
            rst_fs_pair[feature] = (sentiment, likelihood)
            used_sentiment.add(sentiment)
            used_feature.add(feature)
    return rst_fs_pair


def select_sentiment_word_new(feature_list, sentiment_list, total_pair_occur, obj_poss):
    """Associate feature with best fit sentiment
    : Param feature_list     : a list of feature object
    : Param sentiment_list   : a list of sentiment object
    : Param total_pair_occur : total number of pairs
    : Return rst_fs_pair     : map object. Key -> feature word ; Value -> (feature_object, sentiment_object)
    """
    possible_pairs = [(f, s) for f in feature_list for s in sentiment_list if
                      fs_ruler.obj_feature_close(obj_poss, f) and f.token.flag != u'商品']
    fs_pair = []
    for feature, sentiment in possible_pairs:
        likelihood, wd_dis_type = cal_feature_sent_score(feature, sentiment, total_pair_occur)
        # feature-sentiment pair has not been seen in train set
        if likelihood != -1:
            fs_pair.append(
                (likelihood, Dis_Type.MORE_THAN_THREE_WORDS - wd_dis_type, feature.token.origin, sentiment.token.origin))

    # append null association
    for feature in feature_list:
        # TODO: try different tactics, problem left
        if DEBUG:
            print 'null', Local_Model.F_NULL.get(feature.token.word, 1)
        null_likelihood = 0.3 * 0.5 * 0.3 * 0.5 * Local_Model.F_NULL.get(feature.token.word,
                                                                         1) / total_pair_occur * ALPHA
        if DEBUG:
            print 'Feature null', \
                feature.token.word, \
                Local_Model.F_NULL.get(feature.token.word, 0.5), \
                'ALPHA:', ALPHA, \
                1 / total_pair_occur, \
                null_likelihood

        fs_pair.append((null_likelihood, 0, feature.token.word, 'null' + feature.token.word))

    # (likelihood,reversed_word_distance,feature,sentiment)
    sorted_fs_pair = sorted(fs_pair, key=operator.itemgetter(0, 1), reverse=True)
    # key -> feature, value -> (Feature_Obj,Senti_Obj)
    return fs_cmp(sorted_fs_pair)


def cal_alpha():
    p_cover = 0.7
    global ALPHA
    ALPHA = (Local_Model.TOTAL_NULL_SENTI + (p_cover - 1) * Local_Model.TRAIN_SET_VOLUME) / (p_cover * Local_Model.TOTAL_NULL_SENTI)


def class_new(infile=TEST_FILE_PAHT, obj_name=OBJ_NAME, model_name=MODEL_FILE_PATH):
    """Classify test data
    : Param infile     : input file path
    : Param obj_name   : object name
    : Param model_name : model path
    """
    # TODO: Load Model
    Local_Model = file_loader.load_mdl(model_name)
    tfidf = file_loader.load_tfidf('tfidf')

    # load_glb_mdl('globalmodel1107/global_model.txt')

    entity_class, synonym, sent_dic, degree_dic = file_loader.load_knw_base()
    synonym[obj_name] = obj_name

    # Initial Data
    lns = [ln.decode('utf-8').lower() for ln in open(infile).readlines()]
    total_pair_occur = sum(Local_Model.FS_NUM.values())
    cal_alpha()
    result = Result()
    # Classify Test Data
    for ln in lns:
        ln = re.sub('//@.+', '', ln)
        can_rst, feature_sent_pairs = {}, {}
        for ln_after_pr_rule in prior_rules.prior_rules(ln, replace_punc=True):
            # Generate word segments list
            kws, obj_poss = MyLib.seg_ln(ln_after_pr_rule, synonym, obj_name, entity_class)
            if DEBUG:
                print 'Before Combine'
                MyLib.print_seg(kws)
                for obj_pos in obj_poss:
                    print 'obj', obj_pos.phrase,
                for kw in kws:
                    print kw.token.word,
                print

            kws = fs_ruler.combine_sentiment(kws, sent_dic, degree_dic)
            if DEBUG:
                print 'After Combine'
                MyLib.print_seg(kws)
                for obj_pos in obj_poss:
                    print 'obj', obj_pos.phrase,
                for kw in kws:
                    print kw.token.word,
                print
            if len(obj_poss) == 0:
                continue

            feature_list = set()
            sentiment_list = set()

            # Loop over all the candidate keywords,
            # select sentiment and feature list
            for kw in kws:
                if kw.token.word == obj_name:
                    continue
                if kw.token.word in entity_class:
                    feature_list.add(kw)
                if kw.token.word in sent_dic:
                    sentiment_list.add(kw)

                # Cal the probability of the distance between the keyword and the object
                max_likelihood = -1
                for obj_pos in obj_poss:
                    # First compare <obj,keyword> pair
                    likelihood, wd_dis_type = cal_likelihood(obj_pos, kw)
                    # if feature_senti_ruler.abs_dis(obj_pos, kw, Local_Model.CERNTAIN_PAIR):
                    if likelihood > max_likelihood:
                        max_likelihood = likelihood

                likelihood = max_likelihood
                # ignore unseen feature
                if fs_ruler.ignore_unseen_feature(Local_Model.KW_DIS, kw):
                    likelihood = 0

                likelihood *= Local_Model.FS_NUM.get(obj_name + '$' + kw.token.word, 0)

                if can_rst.get(kw.token.word, 0) < likelihood:
                    can_rst[kw.token.word] = likelihood
            # remove none optional sentences
            if len(sentiment_list) == 0:
                continue

            # tmp_pairs = select_sentiment_word_new(feature_list, sentiment_list, total_pair_occur, obj_poss)
            tmp_pairs = fs_select2(feature_list, sentiment_list, total_pair_occur, obj_poss, tfidf)
            # Combine Results
            for key, senti_value in tmp_pairs.items():
                if key in feature_sent_pairs and feature_sent_pairs[key][1] > senti_value[1]:
                    continue
                feature_sent_pairs[key] = senti_value
            feature_sent_pairs = dict(list(feature_sent_pairs.items()) + list(tmp_pairs.items()))
        can_rst = sorted(can_rst.items(), key=operator.itemgetter(1), reverse=True)
        if True:
            print ln.encode('utf-8')
            for kw, value in feature_sent_pairs.items():
                print '(', kw.encode('utf-8'), ',', value[0].encode('utf-8'), value[1], ')'
        MyLib.merge_rst(ln, sent_dic, can_rst, feature_sent_pairs, result, synonym)
    MyLib.print_rst(result)
    print 'ALPHA is ' + str(ALPHA)


def replace_mention(nick_name_lst, obj_name, ln):
    for nick_name in nick_name_lst:
        ln = ln.replace(nick_name, obj_name)
    return ln


def seg_twts(filename):
    stop_dic = [ln.strip().decode('utf-8') for ln in open('/Users/congzicun/Yunio/pycharm/YZKnowl/dictionary/stopwords.txt').readlines()]
    total_dic = {kw.decode('utf-8').strip() for kw in open('/Users/congzicun/Yunio/pycharm/YZKnowl/dictionary/real_final_dic.txt').readlines()}
    for ln in open('local_feature.txt').readlines():
        total_dic.add(ln.decode('utf-8').strip())
    lns = [ln.decode('utf-8') for ln in open(filename).readlines()]
    for ln in lns:
        kwposes = kw_util.backward_maxmatch(ln, total_dic, 100, 2)
        kws = []
        for kwpos in kwposes:
            kw = ln[kwpos[0]:kwpos[1]]
            if kw in stop_dic:
                kw = u'停止词'
            kws.append(kw)
        print ' '.join(kws).encode('utf-8')

# convert chinese punctuation to english punctuation
# 1. period
# 2. exclaim
#


def train_data_clean(infile):
    ad_words = [u'关注', u'转发', u'获取', u'机会', u'赢取', u'推荐'
                , u'活动', u'好友', u'支持', u'话题', u'详情', u'地址', u'赢', u'抽奖', u'好运', u'中奖']
    ads = [u'视频', u'投票', u'【', u'《', u'博文', u'分享自', u'详情']
    lns = [ln.decode('utf-8').lower() for ln in open(infile).readlines()]
    clean_lns = {}
    total_dic = {kw.decode('utf-8').strip() for kw in open('dictionary/real_final_dic.txt').readlines()}
    uniset = set()
    for ln in lns:
        #######REPOST CONTENT#####
        ln = re.sub('//@.+', '', ln)
        if len(kw_util.regex_mention.sub("", ln).strip()) == 0:
            continue
        ########TWEET FILTER######
        tmp_ln = kw_util.tweet_filter(ln)
        ########AD WORDS#######
        ad_counter = 0
        for ad_word in ad_words:
            if ad_word in ln:
                ad_counter += 1

        #########URL###########
        has_url = False
        if kw_util.regex_url.search(ln) is not None:
            ad_counter += 1
            has_url = True
        if has_url:
            for kw in ads:
                if kw in ln:
                    #TODO: 同步到java
                    ad_counter += 2
        #########################
        if ad_counter > 2:
            continue
        word_dic = []
        kwposes = kw_util.backward_maxmatch(ln, total_dic, 100, 1)
        for kwpos in kwposes:
            word_dic.append(ln[kwpos[0]:kwpos[1]])
        if ' '.join(word_dic) not in uniset and tmp_ln not in clean_lns:
            clean_lns[tmp_ln] = ln
            uniset.add(' '.join(word_dic))
    for ln in clean_lns.values():
        print ln.strip().encode('utf-8')
