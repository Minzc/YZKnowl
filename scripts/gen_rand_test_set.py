import random
import sys
lns = [ln for ln in open(sys.argv[1]).readlines()]
gened = set()
for i in range(int(sys.argv[2])):
    index = random.randint(0,len(lns))
    if index not in gened:
        print lns[index],
        gened.add(index)
