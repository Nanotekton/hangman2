import sys, statistics

singleCond = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 41, 42, 43, 44, 47, 48, 
    52, 53, 59, 69, 73, 75, 76, 88, 94, 101, 103, 105, 107, 114, 116, 121, 144, 148, 185, 188, 194, 201, 212, 215, 229, 240, 244, 246, 259, 261, 263, 275, 277, 317, 320, 321, 322,
    323, 325, 332, }

def parseFiles(files):
    reset = False
    runs = []
    for fn in files:
        thisRun = {'added':[], 'prediction':[], 'bestRemainer':[]}
        for line in open(fn):
            #added (221, 0.25368184780701997, 0.43155)
            #0       [2, 25, 9, 35, 11, 8, 10, 31, 88, 20]   [0.93, 0.85, 0.91, 0.77, 0.91, 0.92, 0.91, 0.81, 0.59, 0.87]    0.95
            if line.startswith('added'):
                try:
                    elems = eval(line[5:])
                except:
                    print("problem with line", line)
                    break
                thisRun['added'].append( elems[-1])
                reset = False
            elif len(line.split('\t')) == 4:
                elems = line.split('\t')
                allowedpos = [ i for i,x in enumerate(eval(elems[1])) if not x in singleCond]
                predict = [ x for i,x in enumerate(eval(elems[2])) if i in allowedpos]
                thisRun['prediction'].append( predict  )
                thisRun['bestRemainer'].append( float( elems[3]) )
            else:
                reset = True
                if len(thisRun['added'])>1:
                    runs.append(thisRun)
                    thisRun = {'added':[], 'prediction':[], 'bestRemainer':[]}
    return runs

if __name__ == "__main__":
    files = sys.argv[1:]
    runs = parseFiles(files)
    numPoints = 300
    best = [ [] for  i in range(numPoints)  ]
    for run in runs:
        minlen = min([len(run[k]) for k in run])
        if minlen < numPoints:
            continue
        for i in range(numPoints):
            if False: #run['bestRemainer'][i] != 0.95: 
            #    #break
                best[i].append( 0.95 )
            else:
                if len(run['prediction'][i]) ==0:
                    print("ignore run")
                    break
                best[i].append( run['prediction'][i][0] )
    for i in range(numPoints):
        mean = statistics.mean(best[i])
        print (i, round(mean,3), len(best[i]), best[i], sep='\t' )