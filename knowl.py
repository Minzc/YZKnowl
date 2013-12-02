from scripts.util import file_loader, trans_redis

__author__ = 'congzicun'
import sys
from scripts.algo import local_kw_ext, tester, train
from scripts.algo import FreqBase

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

    elif sys.argv[1] == 'gen_model':
        train.train(sys.argv[2], sys.argv[3].decode('utf-8'))
    elif sys.argv[1] == 'test':
        tester.test(sys.argv[2], sys.argv[3].decode('utf-8'), sys.argv[4])
        # FreqBase.class_new()
    elif sys.argv[1] == 'clean':
        FreqBase.train_data_clean(sys.argv[2])
    elif sys.argv[1] == 'tfidf':
        kb = file_loader.load_knw_base(sys.argv[3])
        dic = file_loader.load_dic()
        lns = file_loader.load_data_set(sys.argv[2])
        rst = local_kw_ext.extr_kw(lns, kb, dic)
        for k, v in rst.items():
            print k.encode('utf-8'), v
    elif sys.argv[1] == 'seg':
        FreqBase.seg_twts(sys.argv[2])
    elif sys.argv[1] == 'trans':
        trans_redis.trans_kv()
