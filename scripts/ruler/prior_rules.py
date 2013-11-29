#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
from scripts.util import kw_util

__author__ = 'congzicun'


def prior_rules(ln, replace_punc=True):
    """split enumeration message into independent sub-messages
    :Param ln           : message to be handled
    :Param replace_punc : if need to supersede chinese punctuation
    :Returns [ln]       : messages after filter
    """
    if replace_punc:
        ln = kw_util.punc_replace(ln)
    ln = re.sub(r'"[^"]{7}.*"', '', ln)
    ln = re.sub(r'\[[^]]{4}.*\]', '', ln)
    if ('1.' in ln and '2.' in ln) \
        or ('1,' in ln and '2,' in ln) \
        or ('1:' in ln and '2:' in ln) \
        or ('#' in ln) \
        or ('-' in ln) \
        or ('!' in ln):
        return filter(lambda x: x != '', re.split(r'\d+[.,]|#|!', ln))
    return [ln]


def is_ad(ln):
    ad_words = [u'关注', u'转发', u'获取', u'机会', u'赢取', u'推荐', u'活动',
                u'好友', u'支持', u'话题', u'详情', u'地址', u'赢', u'抽奖', u'好运', u'中奖'
                                                                       u'奖品', u'参加', u'有奖', u'分享', u'惊喜', u'官方旗舰店',
                u'大奖', u'详情']
    url_ad = [u'视频', u'投票', u'【', u'《', u'博文', u'分享自']
    ln = kw_util.regex_emoji.sub(' ', ln)
    ln = kw_util.regex_mention.sub(' ', ln)
    ad_counter = 0
    for adw in ad_words:
        if adw in ln:
            ad_counter += 1
        # URL
    has_url = False
    if kw_util.regex_url.search(ln) is not None:
        ad_counter += 1
        has_url = True
    if has_url:
        for kw in url_ad:
            if kw in ln:
                ad_counter += 2
    if ad_counter >= 2:
        return True
    return False


def _too_many_phrs(tw):
    phrsenum = len([s for s in re.split(ur'[!.?…~;"#:—,\s]', kw_util.punc_replace(tw)) if len(s.strip()) > 0])
    if phrsenum > 10:
        return True
    return False


def filter_tw(tw):
    if _too_many_phrs(tw):
        return True
    return False

