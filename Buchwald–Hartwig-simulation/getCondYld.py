import statistics

dane = [ x.split('\t') for x in open('inputFeatures.withinp')]

condYld = dict()
condVals = [ set() for i in range(3) ]

for line in dane:
    yld = line[-2]
    cond = tuple(line[-1].strip().split(';')[1:])
    if not cond in condYld:
        condYld[cond]=[]
    condYld[cond].append( float(yld) )
    
    for i,x in enumerate(cond):
        condVals[i].add(x)

for i in condVals:
    print( len(i), i)

for c in condYld:
    if len(condYld[c]) == 1:
        continue
    print( statistics.mean( condYld[c]), *c, sep='\t')