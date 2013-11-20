import json
import sys
lns = [ln.decode('utf-8') for ln in open(sys.argv[1]).readlines()]
for ln in lns:
    jsobj = json.loads(ln)
    print jsobj['tweetContent'].encode('utf-8')
