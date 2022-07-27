import numpy, statistics 
import logging, time


class nextPointPredictor():
    def __init__(self, model=None, optimizer=None, includeSbsForCondition=None):
        allowedSbsSelection=('random', 'maxunc')
        if not includeSbsForCondition in allowedSbsSelection:
            raise
        #each model need to have add least: self.data, predict(), add_point(), retrain(), status()
        self.models = dict() #model objects
        #each acv need to have: run(model, )
        self.acv = dict() #acvizition object
        self.default_acv = None
        self.default_model = None
        self.step_done = 0
        self.includeSbs = includeSbsForCondition
        if model:
            self.add_model(model)
        if optimizer:
            self.add_optimizer(optimizer)

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

    def add_optimizer(self, acv, name=None, overwrite=False, default=True):
        if name == None: #get new name
            for i in range(0, 1000):
                if not i in self.acv:
                    name = i
                    break
        if name in self.acv:
            if overwrite:
                self.acv[name] = acv
            else:
                raise
        else:
            self.acv[name] = acv
        if default:
            self.default_acv = name

    def default_model(self, name=None):
        if name != None:
            if not name in self.models:
                raise
            self.default_model = name
        return self.default_model

    def default_optimizer(self, name=None):
        if name != None:
            if not name in self.acv:
                raise
            self.default_acv = name
        return self.default_acv

    def step(self, ):
        self.step_done += 1
        #print("DO STEP")
        next_point, allPoints = self.acv[self.default_acv].run( self.models[self.default_model], allsbs=self.includeSbs )
        print("NEXT POINT", next_point)
        #raise
        yields = self.models[self.default_model].add_point(next_point)
        self.models[self.default_model].retrain(restart=True)
        return( next_point, float(allPoints['Ynew'][next_point]), numpy.mean(yields) )

    def predict(self):
        return self.models[ self.default_model].predict()

    def status(self):
        return self.models[ self.default_model].status()


def iterativeSearch(predictor):
    #bestYield = []
    maxYield = None
    allYields = []
    bestPoses =[]
    allYpred = []
    for i in range(600):
        try:
            res = predictor.step()
        except:
            print("cannot predict next step")
            raise
            break
        print("added", res)
        if res[-1] == maxYield:
            print("FOUND",i, " a")
            break
        logging.info("added %s", res)
        cond  = predictor.status()
        #print("COND", cond)
        logging.info("pred:NEXTCOND %s", cond)
        bestPos = [ x[2]['rank'] for x in cond[:50]]
        allYlds = [ round(x[2]['mean'],4) for x in cond]
        #print("ALL YIELDS", allYlds)
        if not maxYield:
            maxYield = max(allYlds)
        print(i, bestPos, allYlds[:10], max(allYlds), sep='\t')
        #bestYield.append( allYlds[0])
        allYields.append(allYlds)
        bestPoses.append( bestPos)
        if maxYield and max(allYlds) <  maxYield:
            print("FOUND",i)
            break
    return allYields, bestPoses



def dataToPlot(pred='NNTE', select='EI', nproc=50, inputData=None, sbsSelection='random', minConditionsPerRx=1, selectAllSbsPerConditions=True ):
    import models, selectAlgos
    if inputData ==None:
        raise
    bestYields = []
    bestPoses =[]
    notFound=0
    for i in range(300):
        #model = models.NN('inputFeatures.withinp', verbose=False)
        if pred == 'NNBS':
            model = models.bootstrapNN(inputData, verbose=False, minConditionsPerRx=minConditionsPerRx, percent=10, numModels=100, allsbs=selectAllSbsPerConditions)
        elif pred == 'NNTE':
            model = models.ensembleNN(inputData, verbose=False, minConditionsPerRx=minConditionsPerRx, percent=10, numModels=100, allsbs=selectAllSbsPerConditions)
        elif pred == 'GP':
            model = models.GP(inputData, verbose=False, minConditionsPerRx=minConditionsPerRx, percent=10, allsbs=selectAllSbsPerConditions)
        if select == 'random':
            sel = selectAlgos.randomSelector()
        elif select == 'EI':
            sel = selectAlgos.ExpectedImprovmentSelector()
        elif select == 'PI':
            sel = selectAlgos.ProbabilityOfImprovmentSelector()
        elif select == 'UCB':
            sel = selectAlgos.UCBSelector()
        predictor = nextPointPredictor(model=model, optimizer=sel, includeSbsForCondition=sbsSelection )
        predictor.status()
        try:
            yields, pos = iterativeSearch(predictor)
            bestYields.append(yields)
            bestPoses.append(pos)
            if i ==0:
                print("yields", yields[0])
            logging.info('bestYieldSoFarIn %s %s',i, yields)
        except:
            print("NOT FOUND")
            raise
            notFound+=1
    maxLen = max([ len(x) for x in bestYields])
    maxY = max([ max([max(li) for li in lili])  for lili in bestYields])
    print("MAX Y", maxY, "BY", bestYields, "\nBEST POS", len(bestPoses), [len(gen) for gen in bestPoses[0]], bestPoses[0][0])
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
                allPos.append(0)
        print("I", i, allY, allPos)
        allmeans.append( (statistics.mean(allY), statistics.mean(allPos)) )
    return allmeans

if __name__ == "__main__":
    import sys
    pred='GP' #'NNTE'
    select ='EI'
    if len(sys.argv) >=2:
        pred = sys.argv[1]
    if len(sys.argv) >= 3:
        select = sys.argv[2]
    inpData='inputFeatures.withinp'
    fnlog = 'logs_'+pred+'_'+select+'_SearchPairs_'+str(time.time())
    logging.basicConfig(filename=fnlog+'.log',level=logging.INFO)
    res=dataToPlot(pred=pred, select=select, inputData=inpData, selectAllSbsPerConditions=True, sbsSelection='random', minConditionsPerRx=2 )
    print(res)
    fnres= open(fnlog+'.tsv', 'w')
    print('iteration', 'yield', 'position of best condition', sep='\t', file=fnres)
    for i, gen in enumerate(res):
        print(i, gen[0], gen[1], sep='\t', file=fnres)
    fnres.close()