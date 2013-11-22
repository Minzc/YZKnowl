__author__ = 'congzicun'
import sys
from scripts.algo import trainer
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
        if len(sys.argv) > 2:
            trainer.gen_model(sys.argv[2])
        else:
            trainer.gen_model()
    elif sys.argv[1] == 'classify':
        if len(sys.argv) < 4:
            FreqBase.class_new(sys.argv[2])
        else:
            FreqBase.class_new(sys.argv[2], sys.argv[3].decode('utf-8'), sys.argv[4])
    elif sys.argv[1] == 'test':
        FreqBase.class_new()
    elif sys.argv[1] == 'clean':
        FreqBase.train_data_clean(sys.argv[2])
