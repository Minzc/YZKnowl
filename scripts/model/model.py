__author__ = 'congzicun'
import nltk


class Local_Model:
    F_S_TYPE = {}
    S_AMB = {}
    F_NULL = {}
    CERNTAIN_PAIR = nltk.FreqDist()
    KW_DIS = nltk.FreqDist()
    FS_NUM = nltk.FreqDist()
    F_S_SET = {}
    TRAIN_SET_VOLUME = 0
    TOTAL_NULL_SENTI = 0


class Global_Model:
    F_S_TYPE = {}
    FS_NUM = nltk.FreqDist()
    S_AMB = {}
    KW_DIS = {}


class Dis_Type:
    LESS_THAN_ONE_WORDS = 1
    LESS_THAN_THREE_WORDS = 2
    MORE_THAN_THREE_WORDS = 3
    LESS_THAN_TWO_PHRASE = 4
    LESS_THAN_FOUR_PHRASE = 5
    MORE_THAN_FOUR_PHRASE = 6
    LESS_THAN_ONE_SENT = 7
    MORE_THAN_ONE_SENT = 8
    PRIOR = 9
    POSTERIOR = 10

    def __init__(self):
        self.word_dis = Dis_Type.MORE_THAN_THREE_WORDS
        self.phrs_dis = Dis_Type.MORE_THAN_FOUR_PHRASE
        self.snt_dis = Dis_Type.MORE_THAN_ONE_SENT
        self.rltv_dis = Dis_Type.POSTERIOR


class Result:
    def __init__(self):
        self.POS_FEATURE = {}
        self.NEG_FEATURE = {}
        self.TAG_TWEETS = {}
        self.TOTAL_TWEETS = 0
        self.POS_TAGS = nltk.FreqDist()
        self.NEG_TAGS = nltk.FreqDist()
        self.FEATURE = nltk.FreqDist()
        self.FEATURE_POS = nltk.FreqDist()
        self.FEATURE_NEG = nltk.FreqDist()


class Tokennew:
    def __init__(self, keyword, origin, wdpos, phrspos, sntncpos):
        self.keyword = keyword
        self.origin = origin
        self.wdpos = wdpos
        self.phrspos = phrspos
        self.sntncpos = sntncpos


class Token:
    def __init__(self, sntnc_in_twt, phrase, word_strt_pos, word_end_pos, seg_token):
        self.sntnc = sntnc_in_twt
        self.phrase = phrase
        self.wrd_strt_pos = word_strt_pos
        self.wrd_end_pos = word_end_pos
        self.token = seg_token


class Seg_token:
    def __init__(self, word, flag, origin):
        self.word = word
        self.flag = flag
        self.origin = origin


class kb:
    POSITIVE = 'p'
    NEGATIVE = 'n'
    OBJECT = 'obj'
    FEATURE = 'f'
    # key -> instance
    # value -> entity
    instances = {}
    # key -> sentiment
    # value -> polar
    sentiments = {}
    # key -> feature
    features = {}
    # key -> degree words
    degree = set()
    stop_dic = set()

