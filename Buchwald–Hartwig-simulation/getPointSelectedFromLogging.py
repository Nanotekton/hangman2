import sys, statistics, numpy

replicas = dict()
ignored = 0
collected = 0
pselect = dict()

for line in open(sys.argv[1]):
    #INFO:root:added 14 0 ::: (185, 0.4853195164075993)
    if line.startswith('INFO:root:added '):
        pref, val = line.split(':::')
        _, replica, step =  pref.split()
        step = int(step)
        replica = int(replica)
        val = eval(val )
        if not step in replicas:
            replicas[step]=[]
        replicas[step].append(val)
        if not val[0] in pselect:
            pselect[val[0] ] = []
        pselect[val[0]].append(step)
####
lightRep = []
for step in replicas:
    num = [ x[0] for x in replicas[step] ]
    freq = [num.count(n) for n in  set(num)]
    print(step, sorted(freq, reverse=True) )

for point in sorted(pselect, key= lambda x:statistics.mean(pselect[x]) ):
    print("point id", point, "included in step", pselect[point] )