#!/usr/bin/python
# -*- coding: utf-8 -*- 


import sys
import re
import string
import itertools

import jieba
import nltk
import jieba.analyse
import jieba.posseg as pseg


#Common regular expressions
regex_cnpunc = re.compile(ur"[《》（）&%￥#@！{}【】？—、！；：。“”，…]")
regex_enpunc = re.compile("[" + string.punctuation + "]")
regex_emoji = re.compile(r"\[.+?\]")
regex_topic = re.compile(r"#(.+?)#")
regex_mention = re.compile(r"@(.+?)[\s$]")


#http://daringfireball.net/2010/07/improved_regex_for_matching_urls
regex_url = re.compile(
    r"(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")

max_wordlen, min_wordlen = 100, 2


def load_dict(dictfile, dict={}):
    """load dict file; line format:<word>\t<type> """
    lines = [ln.decode('utf-8').strip().lower().split('\t') for ln in open(dictfile).readlines()]
    for ln in lines:
        if len(ln) == 2 and ln[0][0] != '#':
            if ln[0] in dict:
                dict[ln[0]].add(ln[1])
            else:
                dict[ln[0]] = set([ln[1]])
    return dict


def backward_maxmatch( s, dict, maxWordLen, minWordLen ):
    wordList = []
    curL, curR = 0, len(s)
    while curR >= minWordLen:
        isMatched = False
        if curR - maxWordLen < 0:
            curL = 0
        else:
            curL = curR - maxWordLen
        while curR - curL >= minWordLen: # try all subsets backwards
            if s[curL:curR] in dict: # matched
                wordList.insert(0, (curL, curR))
                curR = curL
                if curR - maxWordLen < 0:
                    curL = 0
                else:
                    curL = curR - maxWordLen
                isMatched = True
                break
            else:  # not matched, try subset by moving left rightwards
                curL += 1

        # not matched, move the right end leftwards
        if not isMatched: curR -= 1
    return wordList


def tweet_filter( tweet ):
    """filter out the noise in a tweet for text analysis: url,emoji,"""
    tweet = regex_mention.sub(" ", tweet)
    tweet = tweet.strip()
    tweet = regex_url.sub(" ", tweet)
    tweet = regex_emoji.sub(" ", tweet)
    tweet = regex_cnpunc.sub(" ", tweet)
    tweet = regex_enpunc.sub(" ", tweet)
    return re.sub(r"\s+", " ", tweet)


def rm_mention(tweet):
    tweet = regex_mention.sub(" ", tweet)
    return re.sub(r"\s+", " ", tweet)


def punc_replace(ln):
    ln = re.sub(ur"。", '.', ln)
    ln = re.sub(ur"；", ';', ln)
    ln = re.sub(ur'！', '!', ln)
    ln = re.sub(ur'？', '?', ln)
    ln = re.sub(ur'，', ',', ln)
    ln = re.sub(ur'（', '(', ln)
    ln = re.sub(ur'）', ')', ln)
    ln = re.sub(u'\.\.', u'…', ln)
    ln = re.sub(u'～', '~', ln)
    ln = re.sub(u'、', ',', ln)
    ln = re.sub(u'“', '"', ln)
    ln = re.sub(u'”', '"', ln)
    ln = re.sub(u'【', '[', ln)
    ln = re.sub(u'】', ']', ln)
    ln = re.sub(u'『', '[', ln)
    ln = re.sub(u'』', ']', ln)
    ln = re.sub(u'＃', '#', ln)
    ln = re.sub(u'：', ':', ln)
    ln = re.sub(u'～', '~', ln)
    ln = re.sub(r'\(.*?\)', '', ln)
    return ln


def extract_tags( sentence, tag, minlen=2, topK=20 ):
    """Extract words with specified pos tags from a sentence"""
    words = pseg.cut(sentence)
    freq = {}
    for w in words:
        if len(w.word.strip()) < minlen or (w.flag is None) or (not w.flag.startswith(tag)): continue
        freq[w.word] = freq.get(w.word, 0.0) + 1.0
    total = sum(freq.values())
    freq = [(k, v / total) for k, v in freq.iteritems()]

    tf_idf_list = [(v * jieba.analyse.idf_freq.get(k, jieba.analyse.median_idf), k)
                   for k, v in freq]
    st_list = sorted(tf_idf_list, reverse=True)
    top_tuples = st_list[:topK]
    tags = [a[1] for a in top_tuples]
    return tags


def extract_key_pairs( s, dict ):
    """extract concept(tp/mn/n)*opinion(ps/ns/a) pairs"""
    s = tweet_filter(s)
    wordlist = backward_maxmatch(s, dict, max_wordlen, min_wordlen)
    wordlist = [(s[w[0]:w[1]], dict[s[w[0]:w[1]]]) for w in wordlist]
    concepts = [w[0] for w in wordlist if 'mn' in w[1] or 'tp' in w[1] or 'n' in w[1]]
    opinion = [w[0] for w in wordlist if 'ps' in w[1] or 'ns' in w[1] or 'a' in w[1]]
    return [(c, o) for c, o in itertools.product(concepts, opinion) if c != o]


def print_topk( dist, n):
    for k, count in dist.items()[:n]:
        print k.encode('utf-8') + "\t" + str(count)
        #    print k.encode('utf-8') + "\t" + str(count)


def print_match_words( ln, dict):
    wordlist = backward_maxmatch(ln, dict, 100, 2)
    print ln
    print "\n".join(["(%d,%d) %s %s" % (w[0], w[1],
                                        ln[w[0]:w[1]], dict[ln[w[0]:w[1]]]) for w in wordlist])


def gen_hot_mentions( infile, n=10):
    """Generate hot mentions ( sorted by frequency ) """
    lines = [ln.strip() for ln in open(infile).readlines()]
    mndist = nltk.FreqDist()
    for ln in lines:
        for w in regex_mention.findall(ln):
            mndist.inc(w)
    print_topk(mndist, n)


def gen_hot_topic( infile, n=10):
    """Generate hot topics ( sorted by frequency ) """
    lines = [ln.strip() for ln in open(infile).readlines()]
    topicdist = nltk.FreqDist()
    for ln in lines:
        for w in regex_topic.findall(ln):
            topicdist.inc(w)
    print_topk(topicdist, n)


def gen_hot_kw( infile, n=10):
    """Generate hot keywords (sorted by frequency) """
    lines = [ln.strip() for ln in open(infile).readlines()]
    kwdist = nltk.FreqDist()
    for ln in lines:
        for w in jieba.analyse.extract_tags(tweet_filter(ln)):
            kwdist.inc(w)
    print_topk(kwdist, n)


def gen_hot_tags( infile, tag, mintaglen=2, n=10):
    """Generate hot keywords (sorted by frequency) """
    lines = [ln.strip().decode('utf-8') for ln in open(infile).readlines()]
    kwdist = nltk.FreqDist()
    for ln in lines:
        for w in extract_tags(tweet_filter(ln), tag, mintaglen):
            kwdist.inc(w)
    print_topk(kwdist, n)


def gen_hot_emoji( infile, n=10 ):
    """Sort out hot emoji used in tweets (sorted by frequency) """
    lines = [ln.strip() for ln in open(infile).readlines()]
    icondist = nltk.FreqDist()
    for ln in lines:
        for w in re.findall(r"\[.+?\]", ln): icondist.inc(w)
    print_topk(icondist, n)


def gen_key_elems( infile, cdictfile, sdictfile ):
    """
        extract the keywords from each line based on 
            *cdict -- concept dictionary
            *sdict -- sentiment dictionary
    """
    dict = load_dict(cdictfile)
    dict = load_dict(sdictfile, dict)
    lines = [ln.strip().decode('utf-8') for ln in open(infile).readlines()]
    max_wordlen, min_wordlen = 100, 2
    for ln in lines:
        print ln.encode('utf-8')
        print "keypairs:" + "\t".join(['(' + c.encode('utf-8') + ',' + o.encode('utf-8') + ')'
                                       for c, o in extract_key_pairs(ln, dict)])

        ln = tweet_filter(ln)
        print "filter:" + ln.encode('utf-8')
        wordlist = backward_maxmatch(ln, dict, max_wordlen, min_wordlen)
        print "keyelems:" + "\t".join(["%s/(%s)" % (ln[w[0]:w[1]].encode('utf-8'),
                                                    ','.join(dict[ln[w[0]:w[1]]]).encode('utf-8'))
                                       for w in wordlist])
        print "\n"


def gen_sentiment_class( emojifile, infile ):
    """根据表情符号情感字典来标注每条tweet的情感倾向"""
    emlines = [ln.strip().split('\t') for ln in open(emojifile).readlines()
               if ln.find('#') != 0]
    emdict = dict([(ln[0], int(ln[1])) for ln in emlines if len(ln) >= 2])
    lines = [ln.strip() for ln in open(infile).readlines()]
    for ln in lines:
        score = sum([emdict[w] for w in re.findall(r"\[.+?\]", ln)
                     if w in emdict])
        print (score > 0 and "正面" or (score < 0 and "负面" or "中性")) + '\t' + ln


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print """Usage: python kw_util.py <cmd> <input_file> where
<input_file> line delimited text; 
<cmd> = [hotkw|emoji|emlabel|topic|mention]
\thotkw <input_file> <topk> <userdict>, generate top k hot keywords from the input file;
\t\t<topk> = 100 in default; 
\t\t<userdict> is optional for specifying user-defined dictionary(better word segmentation.) 
\temoji <input_file> <topk> list top k=(100 in default) emoji in tweets in descending order
\temlabel <input_file> <emoji_dict_file> label each line by its sentiment score
\ttopic <input_file> <topk>, generate top k hot topics from the input file;
\ttags <input_file> <ptag> <mintaglen> <topk>, gen top k pos-tagged words from the input file;
\tmention <input_file> <topk>, generate top k hot mentions from the input file;        
\tkeyelem <input_file> <cdict> <sdict>, extract key elements from the input file;

"""
    elif sys.argv[1] == "hotkw":
        topk = len(sys.argv) > 3 and int(sys.argv[3]) or 100
        if len(sys.argv) > 4: jieba.load_userdict(sys.argv[4])
        gen_hot_kw(sys.argv[2], topk)
    elif sys.argv[1] == "emoji":
        topk = len(sys.argv) > 3 and int(sys.argv[3]) or 100
        gen_hot_emoji(sys.argv[2], topk)
    elif sys.argv[1] == "emlabel":
        if len(sys.argv) < 4:
            print "<emoji_dict_file> is missing."
            exit()
        else:
            gen_sentiment_class(sys.argv[3], sys.argv[2])
    elif sys.argv[1] == "topic":
        topk = len(sys.argv) > 3 and int(sys.argv[3]) or 100
        gen_hot_topic(sys.argv[2], topk)
    elif sys.argv[1] == "tags":
        if len(sys.argv) < 5:
            print "Parameters are missing."
            exit()
        topk = len(sys.argv) > 5 and int(sys.argv[5]) or 100
        gen_hot_tags(sys.argv[2], sys.argv[3], int(sys.argv[4]), topk)
    elif sys.argv[1] == "mention":
        topk = len(sys.argv) > 3 and int(sys.argv[3]) or 100
        gen_hot_mentions(sys.argv[2], topk)
    elif sys.argv[1] == "keyelem":
        if len(sys.argv) < 5:
            print "Parameters are missing."
            exit()
        gen_key_elems(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print "Unrecognized command. Bye."

