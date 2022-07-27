import numpy, statistics 
import logging, time, multiprocessing

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class nextPointPredictor():
    def __init__(self, model=None, selector=None, includeSbsForCondition=None):
        allowedSbsSelection=('random', 'maxunc')
        if not includeSbsForCondition in allowedSbsSelection:
            raise
        #each model need to have add least: self.data, predict(), add_point(), retrain(), status()
        self.models = dict() #model objects
        #each acv need to have: run(model, )
        self.selector = dict() #acvizition object
        self.default_selector = None
        self.default_model = None
        self.step_done = 0
        self.includeSbs = includeSbsForCondition
        if model:
            self.add_model(model)
        if selector:
            self.add_selector(selector)

    def add_model(self, model, name=None, overwrite=False, default=True):
        if name == None: #get new name
            for i in range(0, 1000):
                if not i in self.models:
                    name = i
                    break
        if name in self.models:
            if overwrite:
                self.models[name]=model
            else:
                raise
        else:
            self.models[name]=model
        if default:
            self.default_model = name

    def add_selector(self, selector, name=None, overwrite=False, default=True):
        if name == None: #get new name
            for i in range(0, 1000):
                if not i in self.selector:
                    name = i
                    break
        if name in self.selector:
            if overwrite:
                self.selector[name] = selector
            else:
                raise
        else:
            self.selector[name] = selector
        if default:
            self.default_selector = name

    def default_model(self, name=None):
        if name != None:
            if not name in self.models:
                raise
            self.default_model = name
        return self.default_model

    def default_selector(self, name=None):
        if name != None:
            if not name in self.selector:
                raise
            self.default_selector = name
        return self.default_selector

    def step(self, ):
        self.step_done += 1
        #print("DO STEP")
        next_point, allPoints = self.selector[self.default_selector].run( self.models[self.default_model], allsbs=self.includeSbs )
        self.models[self.default_model].add_point(next_point)
        self.models[self.default_model].retrain(restart=True)
        return( next_point, float(allPoints.iloc[next_point]['yield']) )

    def predict(self):
        return self.models[ self.default_model].predict()

    def status(self):
        return self.models[ self.default_model].status()


def iterativeSearch(predictor, itr):
    #bestYield = []
    maxYield = None
    allYields = []
    bestPoses =[]
    allYpred = []
    for i in range(220):
        try:
            res = predictor.step()
        except:
            print("cannot predict next step")
            #raise
            break
        print("added", res)
        #if res[-1] == maxYield:
        #    print("FOUND",i, " a")
        #    break
        logging.info("added %s", res)
        cond  = predictor.status()
        #print("COND", cond)
        logging.info("pred:NEXTCOND %s %s :::: %s", itr, i, cond)
        bestPos = [ x['rankTrue'] for x in cond[:50]]
        print("BEST", len(bestPos), len(set(bestPos)) )
        allYldsPred = [ round(x['Ypred'],4) for x in cond]
        allYldsTrue = [ round(x['Ytrue'],4) for x in cond]
        #print("ALL YIELDS", allYlds)
        if not maxYield:
            maxYield = max([x['Ytrue'] for x in cond])
        print(i, bestPos, allYldsTrue[:10], "pred", allYldsPred[:10], "REAIN MAX", max(allYldsPred), cond[0], "MAX", maxYield, sep='\t')
        #bestYield.append( allYlds[0])
        allYields.append(allYldsTrue)
        bestPoses.append( bestPos)
        #if maxYield and max(allYlds) <  maxYield:
        #if bestPos[0] == 0:
        #    print("FOUND",i)
        #    break
    return allYields, bestPoses



def singleExperimentRun(inputDictAndItr):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    import models, selectAlgos
    inputDict, itr =inputDictAndItr
    if inputDict['predTech'] == 'NNBS':
        model = models.bootstrapNN(inputDict['inputData'], verbose=False, minConditionsPerRx=inputDict['minConditionsPerRx'], percent=inputDict['initialPortion'], 
                            numModels=100, allsbs=inputDict['selectAllSbsPerConditions'])
    elif inputDict['predTech'] == 'NNTE':
        model = models.ensembleNN(inputDict['inputData'], verbose=False, minConditionsPerRx=inputDict['minConditionsPerRx'], percent=inputDict['initialPortion'], 
                            numModels=100, allsbs=inputDict['selectAllSbsPerConditions'])
    elif inputDict['predTech'] == 'GP':
        model = models.GP(inputDict['inputData'], verbose=False, minConditionsPerRx=inputDict['minConditionsPerRx'], percent=inputDict['initialPortion'], 
                        allsbs=inputDict['selectAllSbsPerConditions'])
    if inputDict['selector'] == 'random':
        sel = selectAlgos.randomSelector()
    elif inputDict['selector'] == 'EI':
        sel = selectAlgos.ExpectedImprovmentSelector()
    elif inputDict['selector'] == 'PI':
        sel = selectAlgos.ProbabilityOfImprovmentSelector()
    elif inputDict['selector'] == 'UCB':
        sel = selectAlgos.UCBSelector()
    predictor = nextPointPredictor(model=model, selector=sel, includeSbsForCondition=inputDict['sbsSelection'] )
    predictor.status()
    try:
        yields, pos = iterativeSearch(predictor, itr)
        logging.info('bestYieldSoFarIn %s', yields)
        return ("OK", yields, pos)
    except:
        print("NOT FOUND")
        #return("FAILED", )
        raise

def subProcFunc( qin,qout):
    while True:
        task = qin.get()
        res = singleExperimentRun(task)
        qout.put(res)


def dataToPlot(pred='NNTE', selector='EI', nproc=50, inputData=None, sbsSelection='random', minConditionsPerRx=1, selectAllSbsPerConditions=True, initialPortion=10 ):
    import models, selectAlgos
    if inputData ==None:
        raise
    bestYields = []
    bestPoses =[]
    notFound=0
    inputDict={'inputData':inputData, 'minConditionsPerRx':minConditionsPerRx, 'initialPortion':initialPortion, 'selectAllSbsPerConditions':selectAllSbsPerConditions,
                'predTech':pred, 'selector':selector, 'sbsSelection':sbsSelection}
    qin = multiprocessing.Queue()
    qout = multiprocessing.Queue()
    proceses =[]
    for i in range(nproc):
        p = multiprocessing.Process( target=subProcFunc, args=( qin,qout) )
        p.start()
        proceses.append(p)
    repetNum =300
    for i in range(repetNum):
        #model = models.NN('inputFeatures.withinp', verbose=False)
        qin.put( (inputDict,i) )
    for i in range(repetNum):
        res = qout.get()
        logging.info(str(i)+ " ::: "+  str(res) )
        if res[0] != 'OK':
            print("PROBLEMATIC RESULT", res)
            print("IGNORED !!!")
        else:
            bestYields.append( res[1])
            bestPoses.append( res[2])
    for  i in range(nproc):
        proceses[i].kill()
    maxLen = max([ len(x) for x in bestYields])
    maxY = max([ max([max(li) for li in lili])  for lili in bestYields])
    #print("MAX Y", maxY, "BY", bestYields, "\nBEST POS", len(bestPoses), [len(gen) for gen in bestPoses[0]], bestPoses[0][0])
    #raise
    allmeans=[]
    for i in range( maxLen+1):
        allY=[]
        allPos =[]
        for onetrain in bestYields:
            try:
                #print("ONETR", onetrain[i])
                allY.append(onetrain[i][0])
            except:
                #raise
                allY.append(maxY)
        for onetrain in bestPoses:
            try:
                allPos.append( onetrain[i].index(0) )
            except:
                #means it is not in top-50
                allPos.append(50)
        print("I", i, allY, allPos)
        allmeans.append( (statistics.mean(allY), statistics.mean(allPos)) )
    return allmeans

if __name__ == "__main__":
    import sys
    pred='GP' #'NNTE'
    select ='EI'
    sbsSelect='random' #'maxunc'
    if len(sys.argv) >=2:
        pred = sys.argv[1]
    if len(sys.argv) >= 3:
        select = sys.argv[2]
    inpData = 'inputDataScience1.withinp' # input.s3'
    fnlog = 'sci1_logs_'+pred+'_'+select+'_SearchPairs_'+'_sbsSelector_'+sbsSelect+str(time.time())
    logging.basicConfig(filename=fnlog+'.log',level=logging.INFO)
    res=dataToPlot(pred=pred, selector=select, nproc=50, inputData=inpData, selectAllSbsPerConditions=True, sbsSelection=sbsSelect, initialPortion=2, minConditionsPerRx=2 )
    print(res)
    fnres= open(fnlog+'.tsv', 'w')
    print('iteration', 'yield', 'position of best condition', sep='\t', file=fnres)
    for i, gen in enumerate(res):
        print(i, gen[0], gen[1], sep='\t', file=fnres)
    fnres.close()
