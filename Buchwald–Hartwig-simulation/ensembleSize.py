import numpy, random, os, sys, time, scipy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

import logging
logging.getLogger('tensorflow').disabled = True

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import optimizers, regularizers, utils
from sklearn import preprocessing

import gpflow
from scipy.stats import norm
from gpflow.ci_utils import ci_niter
from gpflow.optimizers import NaturalGradient
from gpflow.optimizers.natgrad import XiSqrtMeanVar
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from gp_module import *
from hyperas.distributions import choice, uniform
import pandas as pd
from sklearn.preprocessing import StandardScaler
import gp_module



def buildModelNN1(inputDim, training=False, verbose=False):
    input_img = Input(shape=(inputDim,))
    hide1 = Dense(20, activation='elu')(input_img)
    #hide1 = Dense(80, activation='elu')(input_img)
    hide1 = Dropout(0.41133303577)(hide1,  training=training)
    hide9 = Dense(5, activation='elu')(hide1)
    hide9 = Dropout(0.34081821625 )(hide9,  training=training)
    outyield = Dense(1, activation='linear')(hide9)
    model = Model(input_img, outyield)
    optim = optimizers.Adam( lr=0.00502977  )
    model.compile(optimizer=optim, loss='mean_squared_error', metrics=["mean_absolute_error",])
    #model.compile(optimizer=optim, loss='mean_squared_error', metrics=["mean_absolute_error", customLoss])
    if verbose:
        model.summary()
    return model

def buildModelNNmany( inputDim, training=False, verbose=False):
    #MAE: 0.15267878361046314 SPACE:{'Dense': 15, 'Dense_1': 15, 'activation': 'relu', 'activation_1': 'sigmoid', 'activation_2': 'elu', 
    #'batch_size': 10, 'epochs': 40, 'lr': 0.004187918071545324}
    input_img = Input(shape=(inputDim,))
    hide1 = Dense(15, activation='relu')(input_img)
    hide9 = Dense(15, activation='sigmoid')(hide1)
    outyield = Dense(1, activation='elu')(hide9)
    model = Model(input_img, outyield)
    optim = optimizers.Adam( lr=0.004187918071545324 )
    model.compile(optimizer=optim, loss='mean_squared_error', metrics=["mean_absolute_error",])
    #model.compile(optimizer=optim, loss='mean_squared_error', metrics=["mean_absolute_error", customLoss])
    if verbose:
        model.summary()
    return model



def getSelectedData(fn, percent=0.05, selection='worst', allsbs=True, minConditionsPerRx=1, randfloat=None, inputOrder=None):
        fulldata = pd.read_csv(fn, sep='\t', header=None)
        rowNum, columnNum = fulldata.shape
        fulldata = fulldata.rename(columns={ columnNum-2:'yield', columnNum-1:'substrateAndConditions'})
        idxlist = fulldata.columns.tolist()[:-2]
        fulldata[ ['substrate', 'conditions'] ] = fulldata['substrateAndConditions'].str.split(';', n=1, expand=True)
        if minConditionsPerRx > 1:
            #allSbs = dict(fulldata.groupby('substrate')['conditions'].count())
            allSbs = dict(fulldata.groupby('conditions')['substrate'].count())
            conditionsToRm = [sbs for sbs in allSbs if allSbs[sbs] < minConditionsPerRx]
            fulldata = fulldata[~fulldata['conditions'].isin( conditionsToRm) ]
            fulldata = fulldata.reset_index(drop=True)
        yields = fulldata['yield'].to_numpy()
        bestConditions = fulldata.groupby('conditions')['yield'].mean().idxmax() 
        #print("FF", fulldata)
        oneHotSet=set([0,1])
        for xcol in idxlist:
            if set(fulldata[xcol].unique()).issubset( oneHotSet):
                continue
            fulldata[xcol] = preprocessing.scale( fulldata[xcol])
        #scale y if needed
        maxY = fulldata['yield'].max()
        if maxY > 1:
            fulldata['yield']= fulldata['yield']/100.0
        fulldata['training'] = False
        if percent > 1:
            numToSelect = percent
        else:
            numToSelect = int(len(yields)*percent)
        if selection == 'worst':
            idxesSorted = yields.argsort()
        elif selection == 'best':
            idxesSorted = yields.argsort()[::-1]
        elif selection == 'random':
            idxesSorted = yields.argsort()
            #numpy.random.shuffle(idxesSorted)
            if randfloat == None: raise
            random.seed()#randfloat)
            idxesSorted= list(idxesSorted)
            random.shuffle(idxesSorted )
        elif selection == 'given':
            idxesSorted = inputOrder
        for i in range(numToSelect):
            pos = idxesSorted[i]
            fulldata.loc[pos, 'training'] = True
            if allsbs:
                thiscond= fulldata.loc[pos].conditions
                additionalPos = fulldata[ fulldata.conditions == thiscond].index
                print("IDX", thiscond, "==", additionalPos)
                print("RLY", fulldata[ fulldata.conditions == thiscond].shape )
                for idx in additionalPos:
                    fulldata.loc[idx, 'training'] = True
            if len(fulldata[ fulldata.training == True]) >= numToSelect:
                break
        print( fulldata[ fulldata.training == True].shape )
        print('to select', numToSelect, "IDXESSORTED", idxesSorted[:20], "SELE", selection )
        #raise
        print( len(fulldata), "I", idxlist)
        return fulldata, idxlist, bestConditions



def proposeNextPoint(fulldata, mode, excludeSeen=True ):
    """
        mode:
            - random - within selected conditions take randomly substrate
            - maxunc - within selected conditions take substrates with the highest enc
    """
    allowedMode = ('nogroup', 'random', 'maxunc')
    if not mode in allowedMode:
        print("MODE IS", mode)
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

    return onePosInNewForNextStep





def makeModelPrediction(fulldata, res):
    #ODfulldata = model.fulldata.copy()
    #print("SHAPRE RES SE:", res['Ymean'].shape, res['Ystd'].shape, "==>", fulldata.shape)
    fulldata['YpredMean'] = res['Ymean']
    fulldata['YpredStd'] = res['Ystd']
    fulldata['Y'] = res['Ymean']
    fulldata['Ystd'] = res['Ystd']
    fulldata.loc[ fulldata.training == True, 'Y'] = fulldata[ fulldata.training ==True]['yield']
    fulldata.loc[ fulldata.training == True, 'Ystd'] = 0
    return fulldata


##https://github.com/romylorenz/AcquisitionFunction/blob/master/AcquisitionFunctions.py
def EIselector(fulldata, epsilon=0.01, groupBeforeCalc=True, mode='maxunc'):
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
    return proposeNextPoint(fulldata, mode), fulldata


##https://github.com/romylorenz/AcquisitionFunction/blob/master/AcquisitionFunctions.py
def PIselector(fulldata, epsilon=0.01, groupBeforeCalc=True, mode='maxunc'):
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
    return proposeNextPoint(fulldata, mode), fulldata








class GPensemble():
    def __init__(self, fn, percent=0.05, selection='random', allsbs=True, verbose=False, initialize=True, minConditionsPerRx=1, randfloat=None, numModels=100, inputOrder=None):
        self.allsbs = allsbs
        self.verbose = verbose
        self.fulldata, self.inputDataNames = self.getSelectedData(fn, percent=percent, selection=selection, allsbs=allsbs, minConditionsPerRx=minConditionsPerRx, 
                            randfloat=randfloat, inputOrder=inputOrder)
        self.conds = self.trueYcond()
        inputDim = len(self.inputDataNames)
        print("INPUT DIM", inputDim)
        #old: l2=0.001, dropout=0.3 batch_size=32 hidden_n=56
        dropOut = 0.845918563446334
        self.config = {'kernel_base': 'matern32', 'model_kind': 'vgp', 'mean_NN': False, 'noise':0.01, 'kernel_NN':True, 'dot_rbf': False, 'num_inducing_points': None,
                'kernel_cfg':{'embedding':100, 'l2':0.05, 'dropout':dropOut, 'batch_size':128, 'hidden_n':200, 'hidden_l':1, 'output_dim':10, 'input_dim':(inputDim,), 
                              'output_act':'relu', 'input_act':'relu', 'input_bias':True},
                'train_cfg':{ 'iter_mean': None, 'iter': 10, 'iter_internal': 5, 'iter_post': 5}}
        self.models = []
        self.init_weight_list = []
        for i in range(numModels):
            model, weight = self.trainModel()
            self.models.append(model)
            self.init_weight_list.append(weight)

    def retrain(self, restart=True):
        for i,m in enumerate(self.models):
            model, weight = self.trainModel( weight=self.init_weight_list[i])
            self.models[i]=model





def GPpredict(fulldata, inputDataNames, models, numSamples=100):
        #evaluate status
        fullX = fulldata[inputDataNames].to_numpy()
        samples = []
        for model in models:
            predYsamples = model.predict_f_samples(fullX, num_samples=numSamples)
            YpredSamples = predYsamples.numpy()
            YpredSamples = numpy.reshape( YpredSamples, YpredSamples.shape[0:2] )
            #print(YpredSamples.shape)
            samples.append(YpredSamples)
        full = numpy.concatenate( samples)
        mean = numpy.mean(full, axis=0)
        std = numpy.std(full, axis=0)
        return {'Ymean':mean, 'Ystd':std}



def trainGPmodel(config, fulldata, inputDataNames, weight=None):
        kernel, weight = gp_module.make_kernel_from_config(config, weight=weight)
        adam_opt = tf.optimizers.Adam(0.001) #re-initialize in each fold
        #print("SHAPES", self.X.shape, self.Y.shape, "NEW IDXES", len(self.idxesNew) )
        Xtrain = fulldata[fulldata.training == True][inputDataNames].to_numpy()
        Ytrain = fulldata[fulldata.training == True]['yield'].to_numpy()
        #Xtest  = fulldata[fulldata.training == False][inputDataNames].to_numpy()
        #Ytest  = fulldata[fulldata.training == False]['yield'].to_numpy()
        Ytrain = Ytrain.reshape(-1,1)
        print("TRAINS", Xtrain.shape, Ytrain.shape)
        #m, training_loss = gp_module.make_GP_model_from_cfg(self.config, (self.X[self.idxesNew], self.Y[self.idxesNew]), kernel=kernel, support=None)
        m, training_loss = gp_module.make_GP_model_from_cfg(config, (Xtrain, Ytrain), kernel=kernel, support=None)
        iterations = ci_niter(config['train_cfg']['iter'])
        config['train_cfg']['iter'] = 5
        do_natural_grad = (config['model_kind'] in ['vgp', 'svgp']) # and not args.freeze_var
        if do_natural_grad:
            variational_params = [(m.q_mu, m.q_sqrt)]
            natgrad_opt = NaturalGradient(gamma=1.0)
            natgrad_opt.minimize(training_loss, var_list=variational_params)
            set_trainable(m.q_mu, False)
            set_trainable(m.q_sqrt, False)
      
        @tf.function
        def optimization_step():
            adam_opt.minimize(training_loss, var_list=m.trainable_variables)
      
        for _ in range(iterations):
            if do_natural_grad:
                natgrad_opt.minimize(training_loss, var_list=variational_params)
            #print("ieration")
            optimization_step()
            #print("ITER after step")
        return m, weight


def getModelIndexes(length, ensSize, maxSample):
    for i in range(maxSample):
            yield random.sample( range(ensSize), k=length)
    
    
    

def testSize(ensType='GP', ensSize=2_000, maxSample=15_000 ):
    """
    input parameters:
        ensType - {GP NN}
    """
    inputData = 'input.s3'
    initPortion = 100
    sbsSelection = 'random'
    selectAllSbsPerConditions = False
    selectors = ['PI', 'EI']
    fnlog = 'sci2_logs_enssize__SearchPairs_'+'_sbsSelector_'+sbsSelection+'numModels'+str(ensSize)+'_'+str(time.time())
    logging.basicConfig(filename=fnlog+'.log',level=logging.INFO)
    #res=dataToPlot(pred=pred, selector=select, inputData=inpData, selectAllSbsPerConditions=False, sbsSelection=sbsSelect, initialPortion=2, nproc=1, numModels=numModels )
    #def dataToPlot(pred='NNTE', selector='EI', nproc=50, inputData=None, sbsSelection='random', minConditionsPerRx=1, selectAllSbsPerConditions=True, initialPortion=10, numModels=100 ):

    #inputDict={'inputData':inputData, 'minConditionsPerRx':minConditionsPerRx, 'initialPortion':initialPortion, 'selectAllSbsPerConditions':selectAllSbsPerConditions,
    #            'predTech':ensType, 'selector':selector, 'sbsSelection':sbsSelection, 'numModels': ensSize, }
    #GPensemble(
    #self.allsbs = allsbs
    #self.verbose = verbose
    fulldata, inputDataNames, bestCondition = getSelectedData(inputData, percent=initPortion, selection=sbsSelection, allsbs=selectAllSbsPerConditions, minConditionsPerRx=1, randfloat=random.random())
    #self.conds = self.trueYcond()
    inputDim = len(inputDataNames)
    print("INPUT DIM", inputDim)
    #old: l2=0.001, dropout=0.3 batch_size=32 hidden_n=56
    dropOut = 0.845918563446334
    config = {'kernel_base': 'matern32', 'model_kind': 'vgp', 'mean_NN': False, 'noise':0.01, 'kernel_NN':True, 'dot_rbf': False, 'num_inducing_points': None,
                'kernel_cfg':{'embedding':100, 'l2':0.05, 'dropout':dropOut, 'batch_size':128, 'hidden_n':200, 'hidden_l':1, 'output_dim':10, 'input_dim':(inputDim,), 
                              'output_act':'relu', 'input_act':'relu', 'input_bias':True},
                'train_cfg':{ 'iter_mean': None, 'iter': 10, 'iter_internal': 5, 'iter_post': 5}}
    models = []
    init_weight_list = []
    for i in range(ensSize):
        if ensType == 'GP':
            model, weight = trainGPmodel(config, fulldata, inputDataNames)
        models.append(model)
        init_weight_list.append(weight)

    #model = models.GPensemble(inputDict['inputData'], verbose=False, minConditionsPerRx=inputDict['minConditionsPerRx'], percent=inputDict['initialPortion'], 
    #                    allsbs=inputDict['selectAllSbsPerConditions'], randfloat=randfloat, numModels=inputDict['numModels'])
    res = GPpredict(fulldata, inputDataNames, models)
    fulldataCopy = makeModelPrediction(fulldata, res)
    #fulldata.loc[ fulldata.training == True, 'Y'] = fulldata[ fulldata.training ==True]['yield']
    EIsel = EIselector(fulldataCopy)
    PIsel = PIselector(fulldataCopy)
    logging.info("FC %s", list(fulldata['conditions']))
    print("full: EI", EIsel[0], fulldata.loc[EIsel[0]]['conditions'] , "PI", PIsel[0],  fulldata.loc[PIsel[0]]['conditions'])
    logging.info("full: EI %i %s PI: %i %s", EIsel[0], fulldata.loc[EIsel[0]]['conditions'] , PIsel[0],  fulldata.loc[PIsel[0]]['conditions'])
    y = EIsel[1].loc[ EIsel[1].training == False]['YpredMean'].to_numpy()
    unc = EIsel[1].loc[ EIsel[1].training == False]['YpredStd'].to_numpy()
    logging.info("full mean %s", list(y) )
    logging.info("full unc %s", list(unc) )
    for length in range(1,ensSize):
        idxesIter = getModelIndexes(length, ensSize, maxSample)
        PIs=[]
        EIs=[]
        predYield = []
        predUnc = []
        for idxes in idxesIter:
            print("IDXES", idxes)
            selectedModels = [ models[m] for m in idxes]
            res = GPpredict(fulldata, inputDataNames, models)
            fulldataCopy = makeModelPrediction(fulldata, res)
            EIsel = EIselector(fulldataCopy)
            PIsel = PIselector(fulldataCopy)
            PIs.append(PIsel[0])
            EIs.append(EIsel[0])
            #print("EI", EIsel[0], "PI", PIsel[0])
            y = EIsel[1].loc[ EIsel[1].training == False]['YpredMean'].to_numpy()
            y= numpy.reshape( y, (y.shape[0],1) )
            unc = EIsel[1].loc[ EIsel[1].training == False]['YpredStd'].to_numpy()
            unc = numpy.reshape(unc, (unc.shape[0],1) )
            #print("YYY", type(y), y.shape)
            predYield.append(y)
            predUnc.append(unc)
        print("END FOR", length )
        fullY = numpy.concatenate( predYield, axis=1)
        fullUnc = numpy.concatenate( predUnc, axis=1)
        meanY = numpy.mean(fullY, axis=1)
        stdY = numpy.std(fullY, axis=1)
        meanUnc = numpy.mean(fullUnc, axis=1)
        stdUnc = numpy.std(fullUnc, axis=1)
        logging.info("meanY %i %s", length, list(meanY) )
        logging.info("stdY %i %s", length, list(stdY))
        logging.info("meanUnc %i %s", length, list(meanUnc) )
        logging.info("stdUnc %i %s", length, list(stdUnc))
        logging.info("PI %i %s", length, list(PIs) )
        logging.info("EI %i %s", length, list(EIs) )
        
        #print(std)




if __name__ == "__main__":
    testSize( ensSize=2000, maxSample=500)