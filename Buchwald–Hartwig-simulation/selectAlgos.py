import random, scipy, numpy

def getPosInNewIdx(newIdxes, fromFullIdxes):
    if type(fromFullIdxes) == int:
        return newIdxes.index(fromFullIdxes)
    else:
        posInNew = []
        missed=0
        for x in fromFullIdxes:
            if x in newIdxes:
                posInNew.append( newIdxes.index(x) )
            else:
                #print("NOT IN NEW", x)
                missed+=1
        if missed > 1:
            print("missed", missed, "of", len(fromFullIdxes) )
        return posInNew

def proposeNextPoint(fulldata, mode, excludeSeen=True ):
    """
        mode:
            - random - within selected conditions take randomly substrate
            - maxunc - within selected conditions take substrates with the highest enc
    """
    allowedMode = ('nogroup', 'random', 'maxunc', 'var_red')
    if not mode in allowedMode:
        raise
    if mode == 'nogroup':
        posFullIdx = numpy.argmax(EI)
        return getPosInNewIdx(res['newIdxes'], posFullIdx)
    
    bestEIcondname = fulldata.groupby('conditions')['EI'].mean().idxmax()
    allSbsWithEIbest = fulldata[ fulldata.conditions == bestEIcondname]
    if mode == 'random':
        for i in range(10):
            onePosInNewForNextStep = random.choice( allSbsWithEIbest.index.tolist())
            if fulldata.iloc[onePosInNewForNextStep]['training'] == False: #new point accept
                break
        else:
            print("AA", allSbsWithEIbest)
            print("ZZ", bestEIcondname)
            print("SSS", fulldata)
            #print("ZZZ", fulldata.groupby('conditions')['EI'].max() )
            raise
    elif mode == 'maxunc':
        highUncDict = allSbsWithEIbest['Ystd'].to_dict()
        highUncSorted = sorted(highUncDict, reverse=True, key= lambda x:highUncDict[x])
        for i in range(10):
            onePosInNewForNextStep = highUncSorted[i]
            if fulldata.iloc[onePosInNewForNextStep]['training'] == False: #new point accept
                break
        else:
            raise
    elif mode == 'var_red':
        highUncDict = allSbsWithEIbest['score'].to_dict()
        highUncSorted = sorted(highUncDict, reverse=True, key= lambda x:highUncDict[x])
        for i in range(10):
            onePosInNewForNextStep = highUncSorted[i]
            if fulldata.iloc[onePosInNewForNextStep]['training'] == False: #new point accept
                break
        else:
            raise

    return onePosInNewForNextStep

#each acv need to have: run(model, ), predict(model,)
class Selector():
    def __init__(self, ):
        pass

    def makeModelPrediction(self, model):
        res = model.predict()
        fulldata = model.fulldata.copy()
        #print("SHAPRE RES SE:", res['Ymean'].shape, res['Ystd'].shape, "==>", fulldata.shape)
        fulldata['YpredMean'] = res['Ymean']
        fulldata['YpredStd'] = res['Ystd']
        fulldata['Y'] = res['Ymean']
        fulldata['Ystd'] = res['Ystd']
        fulldata['score'] = res['score']
        fulldata.loc[ fulldata.training == True, 'Y'] = fulldata[ fulldata.training ==True]['yield']
        fulldata.loc[ fulldata.training == True, 'Ystd'] = 0
        return fulldata

class randomSelector(Selector ):
    def run(self, model, allsbs=False):
        fulldata = self.makeModelPrediction(model)
        listOfNotSeen = model.fulldata[ model.fulldata.training == False].index.tolist()
        return random.choice( listOfNotSeen ), fulldata

class bestPointSelector(Selector):
    def run(self, model):
        fulldata = self.makeModelPrediction(model)
        fulldata[ fulldata.training == False]['YpredStd'].idxmax(), fulldata

class ExpectedImprovmentSelector(Selector):
    ##https://github.com/romylorenz/AcquisitionFunction/blob/master/AcquisitionFunctions.py
    def run(self, model, epsilon=0.01, groupBeforeCalc=True, allsbs=False):
        fulldata = self.makeModelPrediction(model)
        YmaxSeen = fulldata[ fulldata.training == True]['yield'].max()
        notSeenData = fulldata[ fulldata.training ==False]
        YmaxPredUnseen = notSeenData['YpredMean'].max()
        Ymax = max([YmaxSeen, YmaxPredUnseen])
        #Z = (fulldata['YpredMean'] - YMax - epsilon)/fulldata['YpredStd']
        if groupBeforeCalc:
            #Tak, z tym że niepewności nie uśredniasz a sumujesz, gdyż wariancja sumy 
            #~ suma wariancji (dla zmiennych w przybliżeniu niezależnych). Dokładnie 
            #rzecz biorąc, wariancja sumy = suma wariancji + suma kowariancji: 
            fulldata['avgGroupYield'] = fulldata.groupby('conditions')['Y'].transform('mean')
            fulldata['sumGroupStd']  = fulldata.groupby('conditions')['Ystd'].transform('sum')
            notSeenData = fulldata[ fulldata.training ==False]
            Z = (notSeenData['avgGroupYield'] - Ymax - epsilon) / notSeenData['sumGroupStd']
            #Z = (fulldata['YpredMean'] - Ymax - epsilon)/fulldata['YpredStd']
            EI = (notSeenData['avgGroupYield'] - Ymax - epsilon)* scipy.stats.norm.cdf(Z) + notSeenData['sumGroupStd']*scipy.stats.norm.pdf(Z)
        else:
            Z = (notSeenData['YpredMean'] - Ymax - epsilon) / notSeenData['YpredStd']
            #Z = (fulldata['YpredMean'] - Ymax - epsilon)/fulldata['YpredStd']
            EI = (notSeenData['YpredMean'] - Ymax - epsilon)* scipy.stats.norm.cdf(Z) + notSeenData['YpredStd']*scipy.stats.norm.pdf(Z)
        fulldata['EI']=0
        fulldata.loc[ fulldata.training ==False, 'EI'] = EI
        return proposeNextPoint(fulldata, allsbs), fulldata

class ProbabilityOfImprovmentSelector(Selector):
    ##https://github.com/romylorenz/AcquisitionFunction/blob/master/AcquisitionFunctions.py
    def run(self, model, epsilon=0.01, groupBeforeCalc=True, allsbs=False):
        fulldata = self.makeModelPrediction(model)
        YmaxSeen = fulldata[ fulldata.training == True]['yield'].max()
        notSeenData = fulldata[ fulldata.training ==False]
        YmaxPredUnseen = notSeenData['YpredMean'].max()
        Ymax = max([YmaxSeen, YmaxPredUnseen])
        #Z = (fulldata['YpredMean'] - YMax - epsilon)/fulldata['YpredStd']
        if groupBeforeCalc:
            #Tak, z tym że niepewności nie uśredniasz a sumujesz, gdyż wariancja sumy 
            #~ suma wariancji (dla zmiennych w przybliżeniu niezależnych). Dokładnie 
            #rzecz biorąc, wariancja sumy = suma wariancji + suma kowariancji: 
            fulldata['avgGroupYield'] = fulldata.groupby('conditions')['Y'].transform('mean')
            fulldata['sumGroupStd']  = fulldata.groupby('conditions')['Ystd'].transform('sum')
            notSeenData = fulldata[ fulldata.training ==False]
            Z = (notSeenData['avgGroupYield'] - Ymax - epsilon) / notSeenData['sumGroupStd']
            #Z = (fulldata['YpredMean'] - Ymax - epsilon)/fulldata['YpredStd']
            EI = scipy.stats.norm.cdf(Z)
        else:
            Z = (notSeenData['YpredMean'] - Ymax - epsilon) / notSeenData['YpredStd']
            #Z = (fulldata['YpredMean'] - Ymax - epsilon)/fulldata['YpredStd']
            EI = scipy.stats.norm.cdf(Z)
        fulldata['EI']=0
        fulldata.loc[ fulldata.training ==False, 'EI'] = EI
        return proposeNextPoint(fulldata, allsbs), fulldata


class UCBSelector(Selector):
    ##https://github.com/romylorenz/AcquisitionFunction/blob/master/AcquisitionFunctions.py
    def __init__(self,):
        self.t=0

    def run(self, model, delta=0.1, d=3, t=None, v=1, groupBeforeCalc=True, allsbs=False):
        res = model.predict()
        # res => return {'Ytrain':Ytrain, 'uncTrain':Ystd[self.idxes], 'maxYtrain':Ymax[self.idxes], 
        # 'Ynew':Ynew, 'uncNew':Ystd[self.idxesNew], 'maxYnew':Ymax[self.idxesNew], 'allsbs':joinConds 'newIdxes':}
        ##'allsbs': {('P(tBu)3', 'NaOH', 'MeCN'): {'position': [0, 264],
        if groupBeforeCalc:
            #Tak, z tym że niepewności nie uśredniasz a sumujesz, gdyż wariancja sumy 
            #~ suma wariancji (dla zmiennych w przybliżeniu niezależnych). Dokładnie 
            #rzecz biorąc, wariancja sumy = suma wariancji + suma kowariancji: 
            fulldata['avgGroupYield'] = fulldata.groupby('conditions')['Y'].transform('mean')
            fulldata['sumGroupStd']  = fulldata.groupby('conditions')['Ystd'].transform('sum')
            mu = fulldata['avgGroupYield']
            std = fulldata['sumGroupStd']
        else:
            mu = fulldata['Y']
            std= fulldata['Ystd']

        #muNew = res['Ynew']
        #stdNew = res['uncNew']
        #d = dimensionality
        #t = iteration
        if not t:
            self.t+=1
            t= self.t

        Kappa = numpy.sqrt( v* (2*  numpy.log( (t**(d/2. + 2))*(numpy.pi**2)/(3. * delta)  )))
        EI = muNew + Kappa * stdNew
        fulldata['EI']=0
        fulldata.loc[ fulldata.training ==False, 'EI'] = EI
        return proposeNextPoint(fulldata, allsbs),  fulldata
