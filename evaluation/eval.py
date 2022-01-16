import sys, math

rec_file = sys.argv[1]
at_k = int(sys.argv[2])

eval_ap = 0.
eval_map = 0.
eval_AP = 0.
eval_MAP = 0.
eval_num = 0.
eval_recall =0.
#eval_Recall =0.
eval_hit = 0.
eval_NDCG = 0.
with open(rec_file, 'r') as f:
    for line in f:
        line = line.rstrip().split(' ')
        uid = line[0]
        len_ans = min(at_k, int(line[1]))
        if len_ans == 0: continue

        # AP
        score = float(sum( map(int, line[2:at_k+2]) ))

        eval_num += 1
        eval_ap = score/at_k
        eval_map += eval_ap
        eval_recall += score/int(line[1])

        # MAP
        eval_AP = 0.
        match = 0.
        pos = 0.
        for score in map(int, line[2:at_k+2]):
            pos += 1.
            if score == 1:
                match += 1.
                eval_AP += match/pos
        eval_MAP += eval_AP/len_ans
        
        if "1" in line[2:at_k+2]:
            eval_hit += 1.
        
        # NDCG
        DCG = 0.
        # score_list = list(map(int, line[2:at_k+2]))
        # LGCN-based score
        score_list = list(map(int, line[2:len_ans+2]))
        for i, score in enumerate(score_list):
            if score == 0: continue
            DCG += ((2^score-1)/math.log(i+2, 2))
        score_list = sorted(score_list, reverse = True)
        IDCG = 0.
        for i, score in enumerate(score_list):
            if score == 0: continue
            IDCG += ((2^score-1)/math.log(i+2, 2))
        if IDCG != 0:
            eval_NDCG += DCG/IDCG

# print ('REC@%d: %f' % (at_k, eval_recall/eval_num))
# print ('HIT@%d: %f' % (at_k, eval_map/eval_num))
# print ('MAP@%d: %f' % (at_k, eval_MAP/eval_num))
print ('REC@%d: %f' % (at_k, eval_recall/eval_num))
print ('PRE@%d: %f' % (at_k, eval_map/eval_num))
print ('HIT@%d: %f' % (at_k, eval_hit/eval_num))
print ('MAP@%d: %f' % (at_k, eval_MAP/eval_num))
print ('NDCG@%d: %f' % (at_k, eval_NDCG/eval_num))
print ('F1@%d: %f' % (at_k, 2*(eval_map/eval_num)*(eval_recall/eval_num)/((eval_map/eval_num)+(eval_recall/eval_num))))
