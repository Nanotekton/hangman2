import sys, statistics, numpy

steps = dict()
ignored=0
collected =0
#for line in open(sys.argv[1]):
for fn in sys.argv[1:]:
    for line in open(fn):
        if 'NEXTCOND' in line:
            #INFO:root:pred:NEXTCOND 24 0 ::::
            pref, val = line.split('::::')
            replica, step =  pref.split('NEXTCOND')[1].split()
            step = int(step)
            data= eval(val.replace('array', '' ) )
            #print( data[0] )
            rankList = [ x['rankTrue'] for x in data]
            trueY = [x['Ytrue'] for x in data] 
            predY = [x['Ypred'] for x in data]
            #break
            if not step in steps:
                steps[step] = {'rank': [ ], 'trueY':[], 'predY':[] }
            steps[step]['rank'].append(rankList)
            steps[step]['trueY'].append(trueY)
            steps[step]['predY'].append(predY)
            collected +=1
            #if collected %100 == 0:
            #    print(collected)
#print()
for i in sorted(steps):
    rank1 = [ ranklist.index(0) for ranklist in steps[i]['rank'] ]
    yield1 = [ ranklist[0] for ranklist in steps[i]['trueY'] ]
    if len(rank1) > 1:
        #quanRank = statistics.quantiles(rank1, n=4)
        #quanYield = statistics.quantiles(yield1, n=4)
        qRank = numpy.quantile( numpy.array(rank1), [0.25, 0.75])
        qRank1 = qRank[0]
        qRank2 = qRank[1]
        qYield = numpy.quantile( numpy.array(yield1), [0.25, 0.75])
        qYield1 = qYield[0]
        qYield2 = qYield[1]
        print(i, len(rank1), statistics.mean(rank1), statistics.stdev(rank1), qRank1, qRank2, statistics.mean(yield1), statistics.stdev(yield1), qYield1, qYield2 )
print("IGNORED ", ignored)