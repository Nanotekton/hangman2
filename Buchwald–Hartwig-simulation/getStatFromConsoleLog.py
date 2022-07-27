import sys, statistics
#:: 0
#:: [34, 35, 33, 30, 32, 29, 18, 36, 43, 42, 47, 41, 44, 46, 45, 39, 10, 0, 17, 14, 25, 9, 6, 3, 11, 23, 8, 40, 1, 24, 28, 13, 20, 27, 37, 15, 4, 12, 19, 2, 22, 7, 38, 31, 16, 21, 26, 5]
#:: [0.2399, 0.239, 0.2443, 0.2629, 0.246, 0.2637, 0.3026, 0.2385, 0.2258, 0.2265]
#:: pred
#:: [0.3643, 0.363, 0.361, 0.3592, 0.3567, 0.3549, 0.3546, 0.3542, 0.3431, 0.3424]
#:: REAIN MAX
#:: 0.3643
#:: {'Ypred': 0.3643495086368057, 'Ytrue': 0.2399286722154517, 'condtions': 'P2-t-Bu;Xphos', 'training': array([False,  True]), 'rankTrue': 34}
#:: MAX
#:: 0.5268608975105854


steps = dict()
ignored=0
for line in open(sys.argv[1]):
    if 'MAX' in line and 'pred' in line:
        try:
            step, rankList, trueY, _, predY, _, _, _, _, _ =  line.split('\t')
        except:
            print( "IGNORE", len(line.split('\t') ), line )
            ignored +=1
            continue
        try:
            step=int(step)
            rankList=eval(rankList)
        except:
            print("RA", step, rankList)
            ignored +=1
            continue
        trueY=eval(trueY)
        try:
            predY=eval(predY)
        except:
            ignored +=1
            print("PRED Y", predY)
        if not step in steps:
            steps[step] = {'rank': [ ], 'trueY':[], 'predY':[] }
        steps[step]['rank'].append(rankList)
        steps[step]['trueY'].append(trueY)
        steps[step]['predY'].append(predY)
for i in sorted(steps):
    rank1 = [ ranklist.index(0) for ranklist in steps[i]['rank'] ]
    yield1 = [ ranklist[0] for ranklist in steps[i]['trueY'] ]
    if len(rank1) > 1:
        print(i, len(rank1), statistics.mean(rank1), statistics.stdev(rank1), statistics.mean(yield1), statistics.stdev(yield1) )
print("IGNORED ", ignored)