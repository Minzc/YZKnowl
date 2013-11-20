idf = {ln.decode('utf-8').strip().split(' ')[0]:ln.decode('utf-8').strip().split(' ')[1] for ln in open('idf.txt').readlines()}

for ln in open('noun.ttx').readlines():
    kw,_,_ = ln.decode('utf-8').split(' ')
    print kw.encode('utf-8'),idf.get(kw,11.9547675029)
