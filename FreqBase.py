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
import prior_rules
import feature_senti_ruler


LESS_THAN_THREE_WORDS = 1
MORE_THAN_THREE_WORDS = 2
LESS_THAN_TWO_PHRASE  = 3
LESS_THAN_FOUR_PHRASE = 5
MORE_THAN_FOUR_PHRASE = 6
LESS_THAN_ONE_SENT    = 7
MORE_THAN_ONE_SENT    = 8
PRIOR                 = 9
POSTERIOR             = 10

APLHI = 0.5
DEBUG = False

STOP_DIC         = 'stopwords.txt'
KNWB_PATH        = 'knowl-base.txt'
FILE_PREFIX      = 'yinlu'
TRAIN_FILE_PAHT  = FILE_PREFIX + '_train_data.txt'
TEST_FILE_PAHT   = FILE_PREFIX + '_test_data.txt'
MODEL_FILE_PATH  = FILE_PREFIX + '_model.txt'
OBJ_NAME         = u'银鹭花生牛奶'

SENTI_AMBI       = {}
KW_PAIR_WD_DIS   = {}
KW_PAIR_SNT_DIS  = {}
KW_PAIR_PHRS_DIS = {}
KW_PAIR_RLTV_DIS = {}
KW_DISTR         = nltk.FreqDist()

class Token:
    def __init__(self,sntnc_in_twt,phrase,word_strt_pos,word_end_pos,seg_token):
        self.sntnc        = sntnc_in_twt
        self.phrase       = phrase
        self.wrd_strt_pos = word_strt_pos
        self.wrd_end_pos  = word_end_pos
        self.token        = seg_token
class Seg_token:
    def __init__(self,word,flag):
        self.word = word
        self.flag = flag

def load_knw_base():
    """Load knowledge base to memory
    :Return entity_class    : entity -> classes. Value is a list
    :Return synonym         : instance -> entity.
    :Return sent_dic        : sentiment dictionary
    """
    lns = [ln.decode('utf-8').strip().lower() for ln in open(KNWB_PATH).readlines()]
    # indx -> entity , value -> class
    entity_class = {}
    # indx -> entity , value -> instance
    synonym = {}
    # sentiment
    # indx -> instance value -> entity
    sent_dic = set()
    for ln in lns:
        if ln.startswith('#') or len(ln) ==0:
            continue
        entity,instances,classes = ln.split('\t')
        if classes == u'食品':
            continue
        for cls in classes.split('|'):
            if len(cls) != 0:
                if u'情感词' in cls:
                    sent_dic.add(entity)
                else:
                    entity_class.setdefault(entity,[])
                    entity_class[entity].append(cls)
        synonym[entity] = entity
        for instance in instances.split('|'):
            if len(instance) != 0:
                synonym[instance] = entity
    return entity_class,synonym,sent_dic

def if_ambiguity(kw1,kw2,senti_dic):
    """Judge if kw1 is an ambiguous word
    : Param kw1       : keyword to be judged
    : Param kw2       : accompanied keyword
    : Param senti_dic : sentiment dictionary
    : Returns         : True if kw1 is not ambiguous
    """
    if kw1.phrase == kw2.phrase \
       and kw2.token.word in senti_dic \
       and kw1.token.word in senti_dic \
       and kw1.token.word != kw2.token.word:
        return False
    return True

def gen_model(infile=TRAIN_FILE_PAHT, obj_name = OBJ_NAME):
    """Given training data set, generate Model file
    : Param infile   : training data file path
    : Param obj_name : object name
    """
    entity_class,synonym,senti_dic = load_knw_base()
    tmp_lns,lns = [ln.decode('utf-8').lower() for ln in open(infile).readlines()],[]
    sentiment_ambi = nltk.FreqDist()
    sentiment_occr = nltk.FreqDist()

    # Format lns by prior rules
    for tmp_ln in tmp_lns:
        lns += prior_rules.prior_rules(tmp_ln)

    for ln in lns:
        if DEBUG:
            print 'Gen_Model',ln.encode('utf-8')
        kws,obj_poss = generate_segment_lst_know(ln,synonym,obj_name)
        # Start Statistic
        for kw1 in kws:

            not_ambiguity = True
            sentiment_occr.inc(kw1.token.word)

            for kw2 in kws:
                if kw1 == kw2:
                    continue
                not_ambiguity &= if_ambiguity(kw1,kw2,senti_dic)

                kw_pair = kw1.token.word+'$'+kw2.token.word
                KW_DISTR.inc(kw_pair)
                wd_dis_type,snt_dis_type,phrase_dis_type,relative_pos \
                = decide_dis_feature_type(kw1,kw2)
                # word distance
                KW_PAIR_WD_DIS.setdefault(kw_pair,MyLib.create_and_init_frqdis(MORE_THAN_THREE_WORDS,LESS_THAN_THREE_WORDS))
                KW_PAIR_WD_DIS[kw_pair].inc(wd_dis_type)

                # sentence distance
                KW_PAIR_SNT_DIS.setdefault(kw_pair,MyLib.create_and_init_frqdis(LESS_THAN_ONE_SENT,MORE_THAN_ONE_SENT))
                KW_PAIR_SNT_DIS[kw_pair].inc(snt_dis_type)

                # phrase distance
                KW_PAIR_PHRS_DIS.setdefault(kw_pair,MyLib.create_and_init_frqdis(LESS_THAN_TWO_PHRASE,LESS_THAN_FOUR_PHRASE,MORE_THAN_FOUR_PHRASE))
                KW_PAIR_PHRS_DIS[kw_pair].inc(phrase_dis_type)

                # relative position
                KW_PAIR_RLTV_DIS.setdefault(kw_pair,MyLib.create_and_init_frqdis(PRIOR,POSTERIOR))
                KW_PAIR_RLTV_DIS[kw_pair].inc(relative_pos)
            if not_ambiguity:
                sentiment_ambi.inc(kw1.token.word)

    print '#WD_PAIR_DISTR'
    for kw_pair,wd_dises in KW_PAIR_WD_DIS.items():
        for type,value in wd_dises.items():
            print kw_pair.encode('utf-8')+'$'+str(type)+'$'+str(value)
    print '#SNT_PAIR_DISTR'
    for kw_pair,snt_dises in KW_PAIR_SNT_DIS.items():
        for type,value in snt_dises.items():
            print kw_pair.encode('utf-8')+'$'+str(type)+'$'+str(value)
    print '#PHRS_PAIR_DISTR'
    for kw_pair,phrs_dises in KW_PAIR_PHRS_DIS.items():
        for type,value in phrs_dises.items():
            print kw_pair.encode('utf-8')+'$'+str(type)+'$'+str(value)
    print '#RLTV_POS_DISTR'
    for kw_pair,rltv_dises in KW_PAIR_RLTV_DIS.items():
        for type,value in rltv_dises.items():
            print kw_pair.encode('utf-8')+'$'+str(type)+'$'+str(value)
    print '#WD_DIST'
    for kw,count in KW_DISTR.items():
        print kw.encode('utf-8')+'$'+str(count+1)
    print '#AMBI_DIST'
    for kw,count in sentiment_occr.items():
        print kw.encode('utf-8')+'$'+str(sentiment_ambi.get(kw,0)+1)+'$'+str(count+1)


def load_mdl(infile = MODEL_FILE_PATH):
    lns = [ln.decode('utf-8').lower().strip() for ln in open(infile).readlines()]
    feature_type = -1
    for ln in lns:
        if ln == '#WD_PAIR_DISTR'.lower():
            feature_type = 0
            continue
        elif ln == '#SNT_PAIR_DISTR'.lower():
            feature_type = 1
            continue
        elif ln == '#WD_DIST'.lower():
            feature_type = 2
            continue
        elif ln == '#PHRS_PAIR_DISTR'.lower():
            feature_type = 3
            continue
        elif ln == '#RLTV_POS_DISTR'.lower():
            feature_type = 4
            continue
        elif ln == '#AMBI_DIST'.lower():
            feature_type = 5
            continue

        if feature_type == 0:
            kw1,kw2,type,value = ln.strip().split('$')
            KW_PAIR_WD_DIS.setdefault(kw1+'$'+kw2,nltk.FreqDist())
            KW_PAIR_WD_DIS[kw1+'$'+kw2].inc(int(type),int(value))
        elif feature_type == 1:
            kw1,kw2,type,value = ln.strip().split('$')
            KW_PAIR_SNT_DIS.setdefault(kw1+'$'+kw2,nltk.FreqDist())
            KW_PAIR_SNT_DIS[kw1+'$'+kw2].inc(int(type),int(value))
        elif feature_type == 2:
            kw1,kw2,value = ln.strip().split('$')
            KW_DISTR[kw1+'$'+kw2] = int(value)
        elif feature_type == 3:
            kw1,kw2,type,value = ln.strip().split('$')
            KW_PAIR_PHRS_DIS.setdefault(kw1+'$'+kw2,nltk.FreqDist())
            KW_PAIR_PHRS_DIS[kw1+'$'+kw2].inc(int(type),int(value))
        elif feature_type == 4:
            kw1,kw2,type,value = ln.strip().split('$')
            KW_PAIR_RLTV_DIS.setdefault(kw1+'$'+kw2,nltk.FreqDist())
            KW_PAIR_RLTV_DIS[kw1+'$'+kw2].inc(int(type),int(value))
        elif feature_type == 5:
            sent_kw,not_amb_doc,total_doc = ln.strip().split('$')
            SENTI_AMBI[sent_kw] = int(not_amb_doc) / int(total_doc)

def generate_segment_lst_know(ln,synonym,obj_name):
    know_dic = set(synonym.keys())
    know_dic.add(obj_name)
    sub_sents = filter(lambda x: x != '',re.split(ur'[!.?…~;"]',kw_util.punc_replace(ln)))
    kws,obj_poss = [],[]
    pre_phrases_len = 0
    phrase_position = 0
    for index,sub_sent in enumerate(sub_sents):
        if DEBUG:
            print sub_sent.encode('utf-8')
        for phrases in kw_util.tweet_filter(sub_sent).strip().split(' '):
            if len(phrases) < 1:
                continue
            kw_poses = kw_util.backward_maxmatch( phrases, know_dic, 100, 2 )
            for kw_pos in kw_poses:
                start,end,abs_word_start,abs_word_end = kw_pos[0],\
                                                        kw_pos[1],\
                                                        pre_phrases_len+kw_pos[0],\
                                                        pre_phrases_len+kw_pos[1]
                keyword = phrases[start:end]
                if phrases[start:end] == obj_name:
                    obj_token = Token(index,phrase_position,abs_word_start,abs_word_end,Seg_token(synonym[keyword],'obj'))
                    kws.append(obj_token)
                    obj_poss.append(obj_token)
                else:
                    kws.append(Token(index,phrase_position,abs_word_start,abs_word_end,Seg_token(synonym[keyword],'')))
            pre_phrases_len += len(phrases)
            phrase_position += 1
    return kws,obj_poss


def decide_dis_feature_type(kw1,kw2):
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
    wd_dis_type,snt_dis_type,phrase_dis_type,relative_pos = MORE_THAN_THREE_WORDS,\
                                               MORE_THAN_ONE_SENT,\
                                               MORE_THAN_FOUR_PHRASE,\
                                               PRIOR
    # Sentence Distance Feature
    if abs(kw1.sntnc-kw2.sntnc) < 2:
        snt_dis_type = LESS_THAN_ONE_SENT
    # Word Distance Feature
    if abs(word_distance/2) < 3:
        wd_dis_type = LESS_THAN_THREE_WORDS
    # Phrase Distance Feature
    if abs(kw1.phrase - kw2.phrase) < 2:
        phrase_dis_type = LESS_THAN_TWO_PHRASE
    elif abs(kw1.phrase - kw2.phrase) < 4:
        phrase_dis_type = LESS_THAN_FOUR_PHRASE
    if kw1.wrd_strt_pos > kw2.wrd_strt_pos:
        relative_pos = POSTERIOR
    return wd_dis_type,snt_dis_type,phrase_dis_type,relative_pos

def cal_likelihood(kw,feature):
    snt_lkhd = 1
    wd_dis_type,snt_dis_type,phrase_dis_type,relative_pos\
    = decide_dis_feature_type(kw,feature)
    feature_senti_pair = feature.token.word + '$' + kw.token.word
    # P(c=dis|kw-sentiment)
    # Word Distance Feature
    if KW_PAIR_WD_DIS.has_key(feature_senti_pair):
        snt_lkhd *= KW_PAIR_WD_DIS[feature_senti_pair][wd_dis_type]\
                    / sum(KW_PAIR_WD_DIS[feature_senti_pair].values())
    else:
        snt_lkhd *= 0.5
    if DEBUG:
        print 'SENTIMENT WORD:',feature_senti_pair,snt_lkhd,wd_dis_type
        # Sentence Distance Feature
    if KW_PAIR_SNT_DIS.has_key(feature_senti_pair):
        snt_lkhd *= KW_PAIR_SNT_DIS[feature_senti_pair][snt_dis_type]\
                    / sum(KW_PAIR_SNT_DIS[feature_senti_pair].values())
    else:
        snt_lkhd *= 0.5
    if DEBUG:
        print 'SENTIMENT SENTENCE:',feature_senti_pair,snt_lkhd,snt_dis_type
        # Phrase Distance Feature
    if KW_PAIR_PHRS_DIS.has_key(feature_senti_pair):
        snt_lkhd *= KW_PAIR_PHRS_DIS[feature_senti_pair][phrase_dis_type]\
                    / sum(KW_PAIR_PHRS_DIS[feature_senti_pair].values())
    else:
        snt_lkhd *= 0.3
    if DEBUG:
        print 'SENTIMENT PHRASE:',feature_senti_pair,snt_lkhd,phrase_dis_type
    # Relative Position Feature
    if KW_PAIR_RLTV_DIS.has_key(feature_senti_pair):
        snt_lkhd *= KW_PAIR_RLTV_DIS[feature_senti_pair][relative_pos]\
                    / sum(KW_PAIR_RLTV_DIS[feature_senti_pair].values())
    else:
        snt_lkhd *= 0.5

    return snt_lkhd,wd_dis_type

def cal_feature_sent_score(feature,sentiment,total_pair_occur):
    """Calculate the approved probability of the association of given feature and sentiment
    : Param feature          : feature object
    : Param sentiment        : sentiment object
    : Param total_pair_occur : total pairs occurred in the training data set
    : Returns snt_lkhd       : the likelihood of the association
    : Returns wd_dis_type    : the type of word distance between feature and sentiment
    """
    feature_senti_pair = feature.token.word + '$' + sentiment.token.word
    snt_lkhd,wd_dis_type = cal_likelihood(sentiment,feature)
    # P(kw-sentiment)
    snt_lkhd = snt_lkhd * KW_DISTR.get(feature_senti_pair,1) / total_pair_occur
    # P{amb}
    snt_lkhd = snt_lkhd * SENTI_AMBI.get(sentiment.token.word,1)
    if DEBUG:
        print sentiment.token.word,SENTI_AMBI.get(sentiment.token.word,1)
        print 'Sentiment Final Score:',feature_senti_pair,snt_lkhd,KW_DISTR.get(feature_senti_pair,1)
        print feature_senti_ruler.abs_dis(feature,sentiment)

    # Filter compelled association
    if not feature_senti_ruler.abs_dis(feature,sentiment):
        snt_lkhd = -1
    # Filter direct association
    if feature_senti_ruler.abs_pos(sentiment,feature):
        return 1,LESS_THAN_THREE_WORDS

    return snt_lkhd,wd_dis_type


def select_sentiment_word_new(feature_list,sentiment_list,total_pair_occur,obj_poss):
    """Associate feature with best fit sentiment
    : Param feature_list     : a list of feature object
    : Param sentiment_list   : a list of sentiment object
    : Param total_pair_occur : total number of pairs
    : Return rst_fs_pair     : map object. Key -> feature word ; Value -> (feature_object, sentiment_object)
    """
    possible_pairs = [(f,s) for f in feature_list for s in sentiment_list if not feature_senti_ruler.obj_feature_close(obj_poss,f)]
    fs_pair_likelihood, fs_pair_wd_dis_type = {}, {}
    for possible_pair in possible_pairs:
        feature_sent_pair = possible_pair[0].token.word + '$' + possible_pair[1].token.word
        likelihood,wd_dis_type = cal_feature_sent_score(possible_pair[0],possible_pair[1],total_pair_occur)
        if fs_pair_likelihood.get(feature_sent_pair,-1) < likelihood:
            fs_pair_likelihood[feature_sent_pair] = likelihood
        if fs_pair_wd_dis_type.get(feature_sent_pair,MORE_THAN_THREE_WORDS) >= wd_dis_type:
            fs_pair_wd_dis_type[feature_sent_pair] = wd_dis_type

    sorted_fs_pair = sorted(fs_pair_likelihood.items(), key=operator.itemgetter(1),reverse = True)
    # key -> feature, value -> (Feature_Obj,Senti_Obj)
    rst_fs_pair = {}
    used_sentiment = set()
    prv_score = -1
    print '######################'
    for feature_sent_pair,pair_score in sorted_fs_pair:
        print feature_sent_pair,pair_score
        wd_dis_type = fs_pair_wd_dis_type[feature_sent_pair]
        add_pair = False
        feature_word,senti_word = feature_sent_pair.split('$')
        # 1. feature has not occurred before
        if not rst_fs_pair.has_key(feature_word):
            if senti_word not in used_sentiment:
                add_pair = True
        # 2. two pairs have the same score, select the nearest
        elif pair_score == prv_score:
            prv_pair = rst_fs_pair[feature_word]
            prv_dis_type = fs_pair_wd_dis_type[prv_pair[0]+'$'+prv_pair[1]]
            if prv_dis_type > wd_dis_type:
                used_sentiment.remove(prv_pair[1])
                add_pair = True
        print 'add_pair',add_pair
        if add_pair:
            rst_fs_pair[feature_word] = (feature_word,senti_word)
            used_sentiment.add(senti_word)
        prv_score = pair_score
    print '###############'
    return rst_fs_pair



def class_new(infile=TEST_FILE_PAHT,obj_name = OBJ_NAME,model_name = MODEL_FILE_PATH,):
    """Classify test data
    : Param infile     : input file path
    : Param obj_name   : object name
    : Param model_name : model path
    """
    # Load Model
    load_mdl(model_name)

    entity_class,synonym,sent_dic = load_knw_base()
    total_pair_occur = 0
    for pair,dist in KW_PAIR_SNT_DIS.items():
        total_pair_occur += sum(dist.values())

    # Initial Data
    lns = [ln.decode('utf-8').lower() for ln in open(infile).readlines()]
    total_pair_occur = sum(KW_DISTR.values())
    # Classify Test Data
    for ln in lns:
#        ln = replace_mention([u'@银鹭花生牛奶',u'@林俊杰-银鹭花生牛奶'],OBJ_NAME,ln)
        emoticons = [emoticon for emoticon in re.findall(r"\[.+?\]", ln)]
        can_rst,feature_sent_pairs = {},{}
        for ln_after_pr_rule in prior_rules.prior_rules(ln,replace_punc=True):
            # Generate word segments list
            kws,obj_poss = generate_segment_lst_know(ln_after_pr_rule,synonym,obj_name)
            if DEBUG:
                MyLib.print_seg(kws)
                print obj_poss
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
                    likelihood,wd_dis_type = cal_likelihood(kw,obj_pos)
                    if likelihood > max_likelihood:
                        max_likelihood = likelihood

                likelihood = max_likelihood

                if can_rst.get(kw.token.word,0) < likelihood:
                    can_rst[kw.token.word] = likelihood


            tmp_pairs = select_sentiment_word_new(feature_list,sentiment_list,total_pair_occur)
            feature_sent_pairs = dict(list(feature_sent_pairs.items())+list(tmp_pairs.items()))
        can_rst = sorted(can_rst.items(),key=operator.itemgetter(1),reverse=True)
        print_rst(can_rst,feature_sent_pairs,ln,False)


def replace_mention(nick_name_lst,obj_name,ln):
    for nick_name in nick_name_lst:
        ln = ln.replace(nick_name,obj_name)
    return ln


# convert chines punctuation to english punctuation
# 1. period
# 2. exclaim
#


def train_data_clean(infile):
    lns = [ln.strip() for ln in open(infile).readlines()]
    clean_lns = {}
    for ln in lns:
        if len(kw_util.regex_mention.sub("",ln.decode('utf-8')).strip()) == 0:
            continue
        tmp_ln = kw_util.tweet_filter(ln)
        if not clean_lns.has_key(tmp_ln):
            clean_lns[tmp_ln] = ln
    for ln in clean_lns.values():
        print ln
def print_rst(can_rst,feature_sent_pairs,ln,all_rst=False):
    has_pair = False
    if len(feature_sent_pairs) != 0:
        has_pair = True
    if all_rst or has_pair:
        print ln.strip().encode('utf-8')
        for k,v in can_rst:
            if feature_sent_pairs.has_key(k):
                print '(',k.encode('utf-8'),feature_sent_pairs[k][1].encode('utf-8'),')',
        print '\n'

def seg_files(infile=TEST_FILE_PAHT, obj_name = OBJ_NAME, usrdic_file = 'new_words.txt'):
    jieba.load_userdict(usrdic_file)
    stop_dic = {ln.strip() for ln in open(STOP_DIC).readlines()}
    lns = [ln.strip() for ln in open(infile).readlines()]

    for ln in lns:
        print ln
        kws = MyLib.seg_and_filter(kw_util.tweet_filter(ln.decode('utf-8')),obj_name,stop_dic)
        for kw in kws:
            print kw.word+'/'+kw.flag,
        print
        print 'origin segment:'
        kws = list(jieba.posseg.cut(kw_util.tweet_filter(ln.decode('utf-8'))))
        for i in range(len(kws)):
            kws[i].word = kws[i].word.encode('utf-8')
        for kw in kws:
            print kw.word+'/'+kw.flag
        print '\n'

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print """Usage: python FreqBase.py <cmd> <input_file> where
<input_file> line delimited text;
<cmd> = [gen_model|classify|gen_kw_frq|segment]
\tgen_model <input_file> <obj_name> [user_dic], generate model;
\tgen_kw_frq <input_file> <obj_name> [user_dic], generate keyword frequency statistics
\tsegment <infile> <obj_name> [user_dic], segment sentences
\tclassify <infile> <obj_name> <model_file> <kw_freq_file> [user_dic], classify test data

"""

    if sys.argv[1] == 'gen_model':
        if len(sys.argv) > 2:
            gen_model(sys.argv[2])
        else:
            gen_model()
    elif sys.argv[1] == 'classify':
        if len(sys.argv) < 4:
            class_new(sys.argv[2])
        else:
            class_new(sys.argv[2],sys.argv[3].decode('utf-8'),sys.argv[4])
    elif sys.argv[1] == 'segment':
        seg_files(sys.argv[2],sys.argv[3])
    elif sys.argv[1] == 'test':
        class_new()
    elif sys.argv[1] == 'clean':
        train_data_clean(sys.argv[2])


