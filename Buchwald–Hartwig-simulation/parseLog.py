import sys, statistics

def parseFile(fn, topN=1):
    prevLen = None
    maxYield = None
    allIterations= [ [],  ]
    for line in open(fn):
        if line.startswith('INFO:root:pred:NEXTCOND '):
            dane = eval(line[23:].replace('array', '') )
            thisLen = len(dane) 
            if prevLen and prevLen < thisLen:
                allIterations.append( [] )
            ranki = [ d[2]['rank'] for d in dane]
            maxyields = max([ d[2]['mean'] for d in dane])
            if maxYield == None:
                maxYield = maxyields
            rankBest=0
            try:
                rankBest = ranki.index(1)
            except:
                rankBest = -1
            #print( dane[0][0], thisLen, rankBest, maxyields)
            if topN > 1:
                #print("dane0", dane[0][0], dane[1][0], len(dane) , topN)
                if len(dane) > 1:
                    toAdd = [ statistics.mean( [ dane[x][0] for x in range(min(topN,len(dane)-1)) ]), rankBest+1]
            else:
                toAdd = [dane[0][0], rankBest+1]
            if rankBest == -1:
                #print("MAX", maxYield)
                toAdd = [ maxYield, 1]
            #0.5876223503053188 19 -1 0.7096
            allIterations[-1].append( toAdd)
            prevLen = thisLen
    return allIterations

def makeStat(allData, iters=100):
    Ntop=3
    for i in range(iters):
        yields = [ x[i][0] for x in allData if len(x)> i]
        pos = [ x[i][1] for x in allData if len(x)>i]
        try:
            print(i+1, statistics.mean(yields), statistics.mean(pos), sep='\t' )
        except:
            print("PROBLEM WITH", yields, pos)
if __name__ == "__main__":
    allData = []
    for fn in sys.argv[1:]:
        allData.extend(parseFile(fn, topN=1))
    makeStat(allData, iters=230)