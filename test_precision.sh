alias python='/usr/local/bin/python2.7'
#a="""
python FreqBase.py gen_model  xiaomi/xiaomi_train_data.txt > xiaomi/model.txt
echo 'Finish Train Xiaomi'
python FreqBase.py classify xiaomi/precision_test.txt '小米手机' xiaomi/model.txt > xiaomi/xiaomi_rst.txt
echo 'Finish Xiaomi'
#"""

b="""
python FreqBase.py gen_model  yili/yili_train_data.txt > yili/model.txt
echo 'Finish Train Yili'
python FreqBase.py classify yili/precision_test.txt '伊利谷粒多' yili/model.txt > yili/yili_rst.txt
echo 'Finish Yili'
"""

c="""
python FreqBase.py gen_model  yinlu/yinlu_train_data.txt > yinlu/model.txt
echo 'Finish Train Yinlu'
python FreqBase.py classify yinlu/precision_test.txt '银鹭花生牛奶' yinlu/model.txt > yinlu/yinlu_rst.txt
echo 'Finish Yinlu'
"""
