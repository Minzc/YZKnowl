#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'congzicun'

# NEGATIVE_EMOTICONS = ['[泪]','[抓狂]','[哼]','[鄙视]','[怒]','[威武]','[弱]']
# POSITIVE_EMOTICONS = [ '[嘻嘻]' , '[鼓掌]' , '[偷笑]' , '[太开心]' , '[哈哈]' ]


def abs_pos(senti_wrd, feature):
    """Handle special situation : [Sentiment Word]的[Feature]
    :Param ln       : message
    :Param senti_wrd: sentiment word:
    :Param feature  : feature
    :Returns        : True if the '[sentiment]的[feature]' form exists in the message
    """
    if feature.wrd_end_pos - senti_wrd.wrd_strt_pos == 0 and feature.phrase == senti_wrd.phrase:
        return True
    if senti_wrd.wrd_end_pos - feature.wrd_strt_pos < 2 and feature.phrase == senti_wrd.phrase:
        return True
    return False


def abs_dis(feature, senti_wrd, CERNTAIN_PAIR):
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
    if CERNTAIN_PAIR.has_key(feature.token.word + '$' + senti_wrd.token.word):
        threshold += 1
    if abs(feature.phrase - senti_wrd.phrase) + abs(feature.sntnc - senti_wrd.sntnc) + word_dis / 7 > threshold:
        return False
    return True


def if_rep_due_amb(senti_candi, best_fit_senti, SENTI_AMBI):
    for senti_kw in senti_candi:
        if senti_kw.phrase == best_fit_senti.phrase and \
                        SENTI_AMBI.get(senti_kw.token.word, 0.5) > SENTI_AMBI.get(best_fit_senti.token.word, 0.5):
            best_fit_senti = senti_kw
    return best_fit_senti


def ignore_unseen_senti(KW_DIS, sentiment, feature):
    if sentiment.token.word not in KW_DIS \
        and sentiment.phrase != feature.phrase:
        return True
    return False


def ignore_unseen_feature(KW_DIS, feature):
    if feature.token.word not in KW_DIS:
        return True
    return False

# def ignore_feature(KW_PAIR_DISTR, feature, sentiment, wd_dis_type, snt_dis_type, phrase_dis_type, relative_pos):
#     high_level_pair = feature.token.word + '$sentiment'
#     if high_level_pair not in KW_PAIR_DISTR:
#         high_level_pair = 'feature$' + sentiment.token.word
#     if high_level_pair not in KW_PAIR_DISTR:
#         high_level_pair = 'feature$sentiment'


def combine_sentiment(kws, sent_dic, degree_dic):
    """amend word segmenting result
    :Param kws: segment result
    :senti_dic: sentiment dictionary
    """
    kw_stat = []
    for indx, kw in enumerate(kws):
        kw_stat.append(True)
        # check if the keyword only contain single character
        if len(kw.token.word) == 1:
            kw_stat[indx] = False
            # combine '最好'
            if indx > 0:
                prior_word = kws[indx - 1].token.word
                if (kw.wrd_strt_pos == kws[indx - 1].wrd_end_pos and kws[indx - 1].phrase == kw.phrase) \
                    and (prior_word in degree_dic):
                    kw.wrd_strt_pos = kws[indx].wrd_strt_pos
                    kw_stat[indx] = True
                    # elif indx + 1 < len(kws):
                    #     posterior_word = kws[indx + 1].token.word
                    #     if (kw.wrd_strt_pos == kws[indx + 1].wrd_end_pos and kws[indx + 1].phrase == kw.phrase) \
                    #             and (posterior_word in degree_dic):
                    #         kw_stat[indx] = True
        # check contiguous sentiment word
        elif indx > 0:
            prior_word = kws[indx - 1].token.word
            if prior_word in sent_dic and kw.token.word in sent_dic \
                and kw.wrd_strt_pos == kws[indx - 1].wrd_end_pos and kws[indx - 1].phrase == kw.phrase:
                kw_stat[indx - 1] = False
                # remove degree word
        if kw.token.word in degree_dic:
            kw_stat[indx] = False
    clean_kws = []
    for indx, kw_s in enumerate(kw_stat):
        if kw_s:
            clean_kws.append(kws[indx])
    return clean_kws


def obj_feature_close(obj_poss, feature):
    smallest_dis = 10
    threshold = 4
    for obj_pos in obj_poss:
        tmp_dis = abs(obj_pos.phrase - feature.phrase)
        if tmp_dis < smallest_dis:
            smallest_dis = tmp_dis
    if smallest_dis > threshold:
        return False
    return True

######################


def filter_fs(obj_token, f_token, s_token):
    if _fs_too_far(f_token, s_token):
        return True
    # if _fs_wrong_pos(obj_token, f_token, s_token):
    #     return True
    pass


def _fs_too_far(f_token, s_token):
    threshold = 2
    if f_token.wdpos < s_token.wdpos:
        threshold = 3
    word_dis = abs(f_token.wdpos - s_token.wdpos)
    if abs(f_token.phrspos - s_token.phrspos) + float(word_dis) / 5 > threshold:
        return True
    return False


def _fs_wrong_pos(obj_token, f_token, s_token):
    return not (obj_token.phrspos == f_token.phrspos | f_token.phrspos == s_token.phrspos)


def _of_wrong_pos(obj_token, f_token):
    if f_token.wdpos < obj_token.wdpos:
        return True
    return False


def _f_too_far(obj_token, f_token):
    if abs(obj_token.phrspos - f_token.phrspos) > 4:
        return True
    return False


def filter_f(obj_token, f_token):
    if _of_wrong_pos(obj_token, f_token):
        return True
    if _f_too_far(obj_token, f_token):
        return True
    return False