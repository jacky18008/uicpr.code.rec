import sys
import gzip, json
import argparse
import random
import math
from collections import defaultdict
from math import log, sqrt
from multiprocessing import Queue
from multiprocessing import Process

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('--train', help='data.ui.train')
parser.add_argument('--test', help='data.ui.test')
parser.add_argument('--item', help='rec.item')
parser.add_argument('--remove_dup', type=int, default=0, help='data.ui.test')
parser.add_argument('--embedding', help='data.rep')
parser.add_argument('--task', help='[ui, ii]')
parser.add_argument('--metric', default='dot', help='sim metric')
parser.add_argument('--test_as_query', type=int, default=0, help='to use test items as query')
parser.add_argument('--rec_as_query', type=int, default=0, help='to use rec items as query')
parser.add_argument('--train_item', type=int, default=1, help='to use train items')
parser.add_argument('--test_item', type=int, default=1, help='to use test items')
parser.add_argument('--topk', type=int, default=200, help='# of recommendations')
parser.add_argument('--num_test', type=int, default=0, help='sampled tests')
parser.add_argument('--worker', type=int, default=1, help='# of threads')
args = parser.parse_args()
seed = 2020

def eu_distance(v, v2):
    try:
        return -(sum( (a-b)**2 for a,b in zip(v, v2) )) # omit sqrt
    except:
        return -100.

def cosine_distance(v, v2):
    try:
        score = sum( (a*b) for a,b in zip(v, v2) ) / ( sqrt(sum(a*a for a in v)*sum(b*b for b in v2)) )
        if math.isnan(score):
            return 0
        return score
    except:
        return 0

def dot_distance(v, v2):
    try:
        score = sum( (a*b) for a,b in zip(v, v2) )
        if math.isnan(score):
            return 0
        return score
    except:
        return 0

def bias_dot_distance(v, v2):
    try:
        return sum( (a*b) for a,b in zip(v[1:], v2[1:]) ) + v[0] + v2[0]
    except:
        return 0

def get_top_rec(pairs, q):

    global train_ans
    global test_ans
    global args
    num_sampled = 1000

    for pair in pairs:

        uid, qids = pair
        match_results = [ uid, str(len(test_ans[uid])) ]
        #_recommendation_pool = list(random.sample(recommendation_pool, 500))
        #_recommendation_pool += list(test_ans[uid].keys())
        #_recommendation_pool = set(_recommendation_pool[:])
        _recommendation_pool = recommendation_pool[:]

        scores = defaultdict(lambda: 0.)
        if args.metric == 'cosine':
            for qid in qids:
                for rid in recommendation_pool:
                    if rid in train_ans[uid]: continue
                    if rid == qid: continue
                    if qid in representation and rid in representation:
                        score = cosine_distance(representation[qid], representation[rid])
                        scores[rid] += score

        elif args.metric == 'dot':
            for qid in qids:
                for rid in _recommendation_pool:
                    if args.remove_dup:
                        if rid == qid: continue
                        if rid in train_ans[uid]:
                            continue
                    if qid in representation and rid in representation:
                        score = dot_distance(representation[qid], representation[rid])
                        scores[rid] += score
                    else:
                        scores[rid] += 0.
                    #if rid in train_ans[uid]:
                    #    scores[rid] += 1.

        elif args.metric == 'pop':
             for rid in recommendation_pool:
                scores[rid] = pop[rid]

        for rid in sorted(scores, key=scores.get, reverse=True)[:args.topk]:
            if rid in test_ans[uid]:
                match_results.append('1')
                #point = float(test_ans[uid][rid])
                #if point > 5: point = 5
                #match_results.append(str(point))
            else:
                match_results.append('0')

        q.put(' '.join(match_results))

rec_items = {}
if (args.item):
    print ('load item data from', args.item)
    with open(args.item) as f:
        for line in f:
            iid = line.rstrip('\n')
            rec_items[iid] = 1

print ('load train data from', args.train)
uids = {}
iids = {}
observed = {}
train_ans = defaultdict(dict)
pop = defaultdict(lambda: 0.)
with open(args.train) as f:
    for line in f:
        uid, iid, target = line.rstrip().split('\t')
        if args.train_item:
            rec_items[iid] = 1
        uids[uid] = 1
        iids[iid] = 1
        observed[uid] = 1
        observed[iid] = 1
        pop[iid] += 1.
        if args.task in ['ui', 'ii']:
            train_ans[uid][iid] = 1
        else:
            print('wrong task')
            exit()

print ('load answer data from', args.test)
test_query = defaultdict(dict)
test_ans = defaultdict(dict)
with open(args.test) as f:
    for line in f:
        try:
            uid, iid, target = line.rstrip().split('\t')
        except:
            print(f"error on: {f}, {line}")
        observed[uid] = 1
        observed[iid] = 1
        if uid not in uids: continue
        if args.test_item:
            rec_items[iid] = 1
        if args.remove_dup:
            if iid in train_ans[uid]:
                continue
        test_query[uid][iid] = float(target)
        if iid in rec_items:
            test_ans[uid][iid] = float(target)

print ("load representation from", args.embedding)
representation = {}
recommendation_pool = []
with open(args.embedding, 'r') as f:
    lines = f.readlines()
    for line in lines[:]:
        line = line.rstrip().split('\t')
        ID = line[0]
        if ID in observed:
            representation[ID] = list(map(float, line[1].split(' ')))

        if args.task in ['ui', 'ii']:
            if ID in rec_items:
                recommendation_pool.append(ID)

query_pairs = []

if args.task in ['ii']: # item-item recommendations
    for uid in test_ans:
        qids = list(train_ans[uid].keys())
        if args.test_as_query:
            if args.rec_as_query:
                qids = list(random.sample(test_query[uid].keys(), 1))
            else:
                qids = list(random.sample(test_ans[uid].keys(), 1))
            if args.remove_dup:
                for qid in qids:
                    del test_ans[uid][qid]

        if len(qids) and len(test_ans[uid]):
            query_pairs.append( (uid, qids) )

    random.Random(seed).shuffle(query_pairs)
    if args.num_test > 0:
        query_pairs = query_pairs[:args.num_test]
    else:
        query_pairs = query_pairs

elif args.task == 'ui': # user recommendations

    for uid in test_ans:
        query_pairs.append( (uid, [uid]) )

    random.Random(seed).shuffle(query_pairs)
    if args.num_test > 0:
        query_pairs = query_pairs[:args.num_test]
    else:
        query_pairs = query_pairs

q = Queue()
procs = []
recs = []

step = int(len(query_pairs)/args.worker + 1)
start = 0
end = step
for p in range(args.worker):
    p = Process( target=get_top_rec, args=(query_pairs[start:end], q))
    start += step
    end += step
    end = min(end, len(query_pairs))

    procs.append(p)
    p.start()

out = []
counter = 0
total_count = len(query_pairs)
for i in range(total_count):
    counter += 1
    if counter % 200 == 0:
        sys.stderr.write("%d / %d\n" % (counter, total_count))
    res = q.get()
    out.append("%s" % (res))

for p in procs:
    p.join()

print ('write the result to', args.embedding+'.'+args.task+'.rec')
with open(args.embedding+'.'+args.task+'.rec', 'w') as f:
    f.write('%s\n' % ('\n'.join(out)))


