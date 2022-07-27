import numpy, random, os, sys, abc
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, regularizers, utils
from sklearn import preprocessing
import tensorflow as tf

import gpflow
from scipy.stats import norm
from gpflow.ci_utils import ci_niter
from gpflow.optimizers import NaturalGradient
from gpflow.optimizers.natgrad import XiSqrtMeanVar
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
import numpy as np
from gp_module import *
from hyperas.distributions import choice, uniform
import pandas as pd
from sklearn.preprocessing import StandardScaler
import gp_module
from block_matrix_tools import compute_scores
import logging
from nn_var_red import compute_ensemble_scores

def get_indices_of_maximum(fulldata, pred):
   fulldata['pred_tmp'] = pred
   fulldata['avgGroupYield_tmp'] = fulldata.groupby('conditions')['pred_tmp'].transform('mean')
   max_val = fulldata['avgGroupYield_tmp'].max()
   max_mask = fulldata['avgGroupYield_tmp']==max_val
   max_indices = np.where(max_mask)[0]
   return max_indices

def calc_NNTE_score(models, fullX, train_idx, fulldata):
#def compute_ensemble_scores(ensemble, test, optimum, mode):
   #choose maxima
   n_thompson = 10
   optimum = []
   model_indices = np.random.choice(range(len(models)), n_thompson)
   for model_idx in model_indices:
      model = models[model_idx]
      #TRANSFORM
      maxima_idx = get_indices_of_maximum(fulldata, model.predict(fullX))
      optimum.extend(maxima_idx)

   optimum = np.unique(optimum)
   print('Optimum: N-',len(optimum), optimum.dtype)

   #check indices
   test_idx = ~train_idx
   if np.array(train_idx).dtype.name=='bool':
      test_idx = ~train_idx
      test_idx = np.where(test_idx)[0]
      train_idx = np.where(train_idx)[0]
   else:
      test_idx = [x for x in np.arange(fullX.shape[0]) if x not in train_idx]
      train_idx.sort()
      test_idx.sort()

   test = fullX[test_idx]
   optimum_vals = fullX[optimum]
   print('before SCORES, shapes', test.shape, fullX.shape, 'test_idx:', test_idx.shape, 'optimum vals ',optimum_vals.shape)
   ensemble_scores = compute_ensemble_scores(models, test, optimum_vals, 'grad')['scores']
   #{'scores':scores, 'F':F, 'optimum_av':optimum_av, 'optimum_std':optimum_std, 'test_av':test_av, 'test_std':test_std}
   print('NNTE scores shape: %s'%str(ensemble_scores.shape))
   logging.info('NNTE scores shape: %s'%str(ensemble_scores.shape))

   #transform to full shape
   scores = np.zeros(fullX.shape[0])
   for idx, score in enumerate(ensemble_scores):
      scores[test_idx[idx]] = score

   return scores

def calc_GPE_score(models, fullX, train_idx, fulldata):
   #choose maxima
   n_thompson = 10
   samples = models[0].predict_f_samples(fullX, num_samples=n_thompson).numpy()
   for model in models[1:]:
      samples += model.predict_f_samples(fullX, num_samples=n_thompson).numpy()

   samples = numpy.reshape( samples, samples.shape[0:2] )
   sample_maxima_idx = []
   for sample in samples:
      sample_maxima_idx.extend(get_indices_of_maximum(fulldata, sample))
   sample_maxima_idx = np.unique(sample_maxima_idx) #WHERE's TRANSFORM????

   ensemble_scores = []
   for model in models:
      K = model.kernel(fullX).numpy()
      noise = model.likelihood.variance.numpy()
      ensemble_scores.append(np.array(compute_scores(K, noise, train_idx, sample_maxima_idx)))
   print('GPE score 0 shape: %s'%str(ensemble_scores[0].shape))
   logging.info('GPE score 0 shape: %s'%str(ensemble_scores[0].shape))
   return np.mean(ensemble_scores, axis=0)

def calc_plainGP_score(model, fullX, train_idx, fulldata):
  #choose maxima
  n_thompson = 10
  samples = model.predict_f_samples(fullX, num_samples=n_thompson).numpy()

  samples = numpy.reshape( samples, samples.shape[0:2] )
  sample_maxima_idx = []
  for sample in samples:
     sample_maxima_idx.extend(get_indices_of_maximum(fulldata, sample))
  sample_maxima_idx = np.unique(sample_maxima_idx) #WHERE's TRANSFORM????

  K = model.kernel(fullX).numpy()
  noise = model.likelihood.variance.numpy()
  score = np.array(compute_scores(K, noise, train_idx, sample_maxima_idx))
  logging.info('plainGP score 0 shape: %s'%str(score[0].shape))
  return score

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



def trainNN(model=None, epochs=None, batch=None, fulldata=None, xindex=None, verbose=False):
    if not model  or not epochs  or not batch   or not xindex:
        print("EPOCHS", epochs, "BATCH", batch)
    Xtrain = fulldata[ fulldata.training == True][ xindex].values
    Ytrain = fulldata[ fulldata.training == True]['yield'].values
    Xtest = fulldata[ fulldata.training == False][ xindex].values
    Ytest = fulldata[ fulldata.training == False]['yield'].values
    print("SETTTINGS train:", Xtrain.shape, Ytrain.shape, "E", epochs, batch, "test:", Xtest.shape, Ytest.shape)
    history = model.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch, shuffle=True, validation_data=(Xtest, Ytest), verbose=verbose)
    return history, model


class singleModelBase(abc.ABC):
    def __init__(self, fn, percent=0.05, selection='random', allsbs=True, verbose=False, initialize=True, epochs=(30,10), minConditionsPerRx=1, batch=20, randfloat=None):
        self.allsbs = allsbs
        self.verbose = verbose
        self.epochs=epochs
        self.batch=batch
        self.fulldata, self.inputDataNames = self.getSelectedData(fn, percent=percent, selection=selection, allsbs=allsbs, minConditionsPerRx=minConditionsPerRx, randfloat=randfloat)
        self.conds = self.trueYcond()
        self.model = buildModelNN1( len(self.inutDataNames) )
        if initialize == True:
            hist, model = trainNN(model=self.model, epochs = self.epochs[0], batch=self.batch, fulldata=self.fulldata, xindex=self.inputDataNames)
            self.model = model

    def add_point(self, idx):
        before= len(self.fulldata[ self.fulldata.training == True]) 
        self.fulldata.loc[idx, 'training'] = True
        if self.allsbs: #add other sbs par in the same conditions
            conds = self.fulldata.conditions[idx]
            sameSbs = self.fulldata[self.fulldata.conditions == conds].index.tolist()
            for i in sameSbs:
                self.fulldata.loc[i, 'training'] = True
                print("ADD", i)
        print("before AFTER ADD POINT training", before, len(self.fulldata[ self.fulldata.training == True]), "ALL SBS", self.allsbs )

    def getSelectedData(self, fn, percent=0.05, selection='worst', allsbs=True, minConditionsPerRx=1, randfloat=None, inputOrder=None):
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
        yields = fulldata['yield'].values
        self.bestConditions = fulldata.groupby('conditions')['yield'].mean().idxmax() 
        #scale input
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
        #print( fulldata[ fulldata.training == True].shape )
        print('to select', numToSelect, "IDXESSORTED", idxesSorted[:20], "SELE", selection )
        #raise
        print( len(fulldata), "I", idxlist)
        return fulldata, idxlist

    def trueYcond(self):
        condsY = dict()
        maxAvg=0
        #maxAvgPred=0
        allconds = self.fulldata.conditions.unique().tolist()
        for cond in allconds:
            vals = self.fulldata[ self.fulldata.conditions == cond]['yield' ]
            avg = numpy.mean(vals)
            #avgPred = numpy.mean( self.fulldata[ self.fulldata.conditions == cond]['Ypred'])
            condsY[cond] = {'values':vals, 'mean':avg, }
            if avg > maxAvg:
                maxAvg=avg
        for r, cond in enumerate(sorted( condsY, reverse=True, key= lambda x:condsY[x]['mean'])):
            condsY[cond]['rankTrue'] = r
            #print("RRRANK", r, cond, "NUMCOND" )
            condsY[cond]['diff'] = maxAvg - condsY[cond]['mean']
        #for r, cond in enumerate(sorted( condsY, reverse=True, key= lambda x:condsY[x]['predMean'])):
            #condsY[cond]['rankPred'] = r
            #print("RRRANK", r, cond, "NUMCOND" )
            #condsY[cond]['diff'] = maxAvg - condsY[cond]['mean']
        return condsY



    def predict(self, ):
        res = self.model.predict(self.fulldata[ self.inputDataNames].values )
        if self.allsbs:
            print("not implemented ")
            raise
        return {'YfullPred':res}



class multiModelBase(singleModelBase):
    def __init__(self, fn, percent=0.05, selection='random', allsbs=True, verbose=False, initialize=True, epochs=(40,10), batch=10, numModels=100, minConditionsPerRx=1, randfloat=None):
        self.allsbs = allsbs
        self.verbose = verbose
        self.epochs=epochs
        self.batch=batch
        self.make_score=False
        self.fulldata, self.inputDataNames = self.getSelectedData(fn, percent=percent, selection=selection, allsbs=allsbs, minConditionsPerRx=minConditionsPerRx, randfloat=randfloat)
        self.conds = self.trueYcond()
        self.models = [ buildModelNNmany( inputDim=len(self.inputDataNames), verbose=self.verbose ) for i in range(numModels)]
        self.init_weights_list =[]
        if initialize:
            for modid,mod in enumerate(self.models):
                print(modid)
                hist, model = trainNN(model=self.models[modid], epochs = self.epochs[0], batch=self.batch, fulldata=self.fulldata, xindex=self.inputDataNames)
                self.models[modid] = model
                self.init_weights_list.append( model.get_weights() )



    def predict(self,):
        Ypreds = numpy.zeros( ( len(self.models), len(self.fulldata)))
        fullX = self.fulldata[ self.inputDataNames].values
        for modelNum, model in enumerate(self.models):
            pred = model.predict(fullX )
            Ypreds[modelNum] = pred.reshape(-1)
        Yfull = numpy.mean(Ypreds, axis=0)
        Ystd = numpy.std(Ypreds, axis=0)
        if self.make_score:
           logging.info('computing score')
           train_mask = self.fulldata.training == True
           score = calc_NNTE_score(self.models, fullX, train_mask, self.fulldata)
           logging.info('NNTE score shape %s'%str(score.shape))
           print('NNTE score shape %s'%str(score.shape))
        else:
           score = Ystd
        return {'Ymean':Yfull, 'Ystd':Ystd, 'score':score}



    def retrain(self, epochs=None, restart=False):
        if not epochs:
            if restart:
                epochs = self.epochs[0]
            else:
                epochs = self.epochs[1]
        if restart:
            for mid,m in enumerate(self.models):
                self.models[mid].set_weights(self.init_weights_list[mid])
        #return self.trainModel(epochs=epochs, batch=self.batch)
                trainNN(model=self.models[mid], epochs=epochs, batch=self.batch, fulldata=self.fulldata, xindex=self.inputDataNames, verbose=self.verbose)


    def status(self):
        preds = self.predict()
        allconds = self.fulldata.conditions.unique().tolist()
        predictedBest = []
        for cond in allconds:
            pos = self.fulldata[ self.fulldata.conditions == cond].index.tolist()
            yields = self.fulldata.loc[ pos]['yield']
            predY = preds['Ymean'][ pos]
            training = self.fulldata.loc[ pos]['training'].unique()
            predictedBest.append( {'Ypred':numpy.mean(predY), 'Ytrue':numpy.mean(yields), 'condtions':cond, 'training':training})

        predictBestSorted = sorted(predictedBest, reverse=True, key= lambda x: x['Ytrue'])
        for rank, dictio in enumerate( predictBestSorted):
            dictio['rankTrue']=rank
        #return predictedBestSorted
        return sorted(predictBestSorted, reverse=True, key= lambda x: x['Ypred'])

class bootstrapNN(multiModelBase):

    def __init__(self, fn, percent=0.05, selection='random', allsbs=True, verbose=False, initialize=True, epochs=(40,10), batch=10,numModels=100, minConditionsPerRx=1, random=None):
        super().__init__(fn, percent=percent, selection='random', allsbs=True, verbose=False, initialize=False, epochs=epochs, batch=batch, minConditionsPerRx=minConditionsPerRx, 
                          randfloat=randfloat)
        self.models = [ self.buildModel( len(self.X[0]) ) for i in range(numModels)]
        self.init_weights_list =[]
        if initialize:
            for modid,mod in enumerate(self.models):
                print(modid)
                bs = self.makeBootstrap() 
                self.trainModel(mod, trainIdx=bs, epochs=self.epochs[0], batch=self.batch )
                self.init_weights_list.append( mod.get_weights() )

    def makeBootstrap(self, ):
        return [ random.choice(self.idxes) for i in self.idxes]



    def retrain(self, epochs=None, restart=True):
        if not epochs:
            if restart:
                epochs = self.epochs[0]
            else:
                epochs = self.epochs[1]
        if restart:
            for i,model in enumerate(self.models):
                model.set_weights( self.init_weights_list[i] )
                self.trainModel(model=model, epochs=epochs)



class ensembleNN(multiModelBase):
    pass
    #def train(self):
    #    for model,num
    #    trainNN(self, model=None, epochs=None, batch=None, fulldata=None, xindex=None):


class GP(singleModelBase):
    def __init__(self, fn, percent=0.05, selection='random', allsbs=True, verbose=False, initialize=True, minConditionsPerRx=1, randfloat=None, normalized=False):
        self.allsbs = allsbs
        self.verbose = verbose
        self.normalized = normalized
        self.fulldata, self.inputDataNames = self.getSelectedData(fn, percent=percent, selection=selection, allsbs=allsbs, minConditionsPerRx=minConditionsPerRx, randfloat=randfloat)
        self.conds = self.trueYcond()
        inputDim = len(self.inputDataNames)
        print("INPUT DIM", inputDim)
        #old: l2=0.001, dropout=0.3 batch_size=32 hidden_n=56
        self.config = {'kernel_base': 'matern32', 'model_kind': 'vgp', 'mean_NN': False, 'noise':0.01, 'kernel_NN':True, 'dot_rbf': False, 'num_inducing_points': None,
                'kernel_cfg':{'embedding':100, 'l2':0.05, 'dropout':0.845918563446334, 'batch_size':128, 'hidden_n':200, 'hidden_l':1, 'output_dim':10, 'input_dim':(inputDim,), 
                              'output_act':'relu', 'input_act':'relu', 'input_bias':True},
                'train_cfg':{ 'iter_mean': None, 'iter': 10, 'iter_internal': 5, 'iter_post': 5}}
        self.model, self.init_weight = self.trainModel()



    def retrain(self, restart=True):
        #retrain(self, epochs=None, restart=False):
        #print("TRAIN")
        self.model =None
        self.model, init_weight = self.trainModel( weight=self.init_weight)
        return self.model



    def status(self):
        preds = self.predict()
        allconds = self.fulldata.conditions.unique().tolist()
        #print("FULLDA", len(self.fulldata), "P", len(preds['Ymean']), len(preds['Ystd']) )
        predictedBest = []
        av_unc = preds['Ystd'].mean()
        av_unc_train = preds['Ystd'][self.fulldata.training].mean()
        av_unc_test = preds['Ystd'][~self.fulldata.training].mean()
        for cond in allconds:
            pos = self.fulldata[ self.fulldata.conditions == cond].index.tolist()
            #print("POS", pos)
            yields = self.fulldata.loc[ pos]['yield']
            predY = preds['Ymean'][ pos]
            std = preds['Ystd'][ pos]
            training = self.fulldata.loc[ pos]['training'].unique()
            predictedBest.append( {'Ypred':numpy.mean(predY), 'Ytrue':numpy.mean(yields), 'unc':numpy.mean(std), 'max_unc':numpy.max(std), 'condtions':cond, 'training':training, 'av_unc_train':av_unc_train, 'av_unc_test':av_unc_test, 'total_av_unc':av_unc})

        predictBestSorted = sorted(predictedBest, reverse=True, key= lambda x: x['Ytrue'])
        for rank, dictio in enumerate( predictBestSorted):
            dictio['rankTrue']=rank
        return sorted(predictBestSorted, reverse=True, key= lambda x: x['Ypred'])


    def buildModel(self,):
        pass

    def trainModel(self, weight=None):
        kernel, weight = gp_module.make_kernel_from_config(self.config, weight=weight)
        adam_opt = tf.optimizers.Adam(0.001) #re-initialize in each fold
        #print("SHAPES", self.X.shape, self.Y.shape, "NEW IDXES", len(self.idxesNew) )
        train_mask = self.fulldata.training == True
        Xtrain = self.fulldata[ train_mask ][ self.inputDataNames].to_numpy()
        Ytrain = self.fulldata[ train_mask ]['yield'].to_numpy()
        if self.normalized:
           max_values = self.fulldata[train_mask].groupby('conditions')['yield'].transform('max').to_numpy()
           Ytrain/=np.where(max_values!=0, max_values,1)
        if self.perturb:
           Ytrain*=self.fulldata[train_mask]['perturb'].values
        Xtest = self.fulldata[ self.fulldata.training == False][ self.inputDataNames].to_numpy()
        Ytest = self.fulldata[ self.fulldata.training == False]['yield'].to_numpy()
        Ytrain = Ytrain.reshape(-1,1)
        #print("TRAINS", Xtrain.shape, Ytrain.shape)
        #m, training_loss = gp_module.make_GP_model_from_cfg(self.config, (self.X[self.idxesNew], self.Y[self.idxesNew]), kernel=kernel, support=None)
        m, training_loss = gp_module.make_GP_model_from_cfg(self.config, (Xtrain, Ytrain), kernel=kernel, support=None)
        iterations = ci_niter(self.config['train_cfg']['iter'])
        self.config['train_cfg']['iter'] = 5
        do_natural_grad = (self.config['model_kind'] in ['vgp', 'svgp']) # and not args.freeze_var
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

    def predict(self,):
        #evaluate status
        fullX = self.fulldata[ self.inputDataNames].to_numpy()
        predY, predY_var = self.model.predict_f(fullX)
        Ypred = predY.numpy()
        print("SHAPES pred in, out, out", fullX.shape, Ypred.shape, predY_var.shape,)
        Ystd = np.sqrt(predY_var.numpy())
        return {'Ymean':Ypred, 'Ystd':Ystd}




class plainGP(singleModelBase):
    def __init__(self, fn, percent=0.05, selection='random', allsbs=True, verbose=False, initialize=True, minConditionsPerRx=1, randfloat=None):
        self.allsbs = allsbs
        self.make_score = False
        self.verbose = verbose
        self.fulldata, self.inputDataNames = self.getSelectedData(fn, percent=percent, selection=selection, allsbs=allsbs, minConditionsPerRx=minConditionsPerRx, randfloat=randfloat)
        self.conds = self.trueYcond()
        inputDim = len(self.inputDataNames)
        print("INPUT DIM", inputDim)
        #old: l2=0.001, dropout=0.3 batch_size=32 hidden_n=56
        #self.config = {'kernel_base': 'matern32', 'model_kind': 'vgp', 'mean_NN': False, 'noise':0.01, 'kernel_NN':True, 'dot_rbf': False, 'num_inducing_points': None,
        #        'kernel_cfg':{'embedding':100, 'l2':0.05, 'dropout':0.845918563446334, 'batch_size':128, 'hidden_n':200, 'hidden_l':1, 'output_dim':10, 'input_dim':(inputDim,), 
        #                      'output_act':'relu', 'input_act':'relu', 'input_bias':True},
        #        'train_cfg':{ 'iter_mean': None, 'iter': 10, 'iter_internal': 5, 'iter_post': 5}}
        self.config = {'kernel_base': 'matern32', 'model_kind': 'gp', 'mean_NN': False, 'noise':0.01, 'kernel_NN':False, 'dot_rbf': False, 'num_inducing_points': None,
                'kernel_cfg':{},
                'train_cfg':{ 'iter_mean': None, 'iter': 10, 'iter_internal': 5, 'iter_post': 5}}
        self.model, self.init_weight = self.trainModel()



    def retrain(self, restart=True):
        #retrain(self, epochs=None, restart=False):
        #print("TRAIN")
        self.model =None
        self.model, init_weight = self.trainModel( weight=self.init_weight)
        return self.model



    def status(self):
        preds = self.predict()
        allconds = self.fulldata.conditions.unique().tolist()
        #print("FULLDA", len(self.fulldata), "P", len(preds['Ymean']), len(preds['Ystd']) )
        predictedBest = []
        for cond in allconds:
            pos = self.fulldata[ self.fulldata.conditions == cond].index.tolist()
            #print("POS", pos)
            yields = self.fulldata.loc[ pos]['yield']
            predY = preds['Ymean'][ pos]
            training = self.fulldata.loc[ pos]['training'].unique()
            predictedBest.append( {'Ypred':numpy.mean(predY), 'Ytrue':numpy.mean(yields), 'condtions':cond, 'training':training})

        predictBestSorted = sorted(predictedBest, reverse=True, key= lambda x: x['Ytrue'])
        for rank, dictio in enumerate( predictBestSorted):
            dictio['rankTrue']=rank
        return sorted(predictBestSorted, reverse=True, key= lambda x: x['Ypred'])


    def buildModel(self,):
        pass

    def trainModel(self, weight=None):
        kernel, weight = gp_module.make_kernel_from_config(self.config, weight=weight)
        adam_opt = tf.optimizers.Adam(0.001) #re-initialize in each fold
        #print("SHAPES", self.X.shape, self.Y.shape, "NEW IDXES", len(self.idxesNew) )
        Xtrain = self.fulldata[ self.fulldata.training == True][ self.inputDataNames].values
        Ytrain = self.fulldata[ self.fulldata.training == True]['yield'].values
        Xtest = self.fulldata[ self.fulldata.training == False][ self.inputDataNames].values
        Ytest = self.fulldata[ self.fulldata.training == False]['yield'].values
        Ytrain = Ytrain.reshape(-1,1)
        #print("TRAINS", Xtrain.shape, Ytrain.shape)
        #m, training_loss = gp_module.make_GP_model_from_cfg(self.config, (self.X[self.idxesNew], self.Y[self.idxesNew]), kernel=kernel, support=None)
        m, training_loss = gp_module.make_GP_model_from_cfg(self.config, (Xtrain, Ytrain), kernel=kernel, support=None)
        iterations = ci_niter(self.config['train_cfg']['iter'])
        self.config['train_cfg']['iter'] = 5
        do_natural_grad = (self.config['model_kind'] in ['vgp', 'svgp']) # and not args.freeze_var
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

    def predict(self,):
        #evaluate status
        fullX = self.fulldata[ self.inputDataNames].values
        predY, predY_var = self.model.predict_f(fullX)
        Ypred = predY.numpy()
        print("SHAPES pred in, out, out", fullX.shape, Ypred.shape, predY_var.shape,)
        Ystd = np.sqrt(predY_var.numpy())
        score = Ystd
        if self.make_score:
           train_mask = self.fulldata.training == True
           score = calc_plainGP_score(self.model, fullX, train_mask, self.fulldata)
           logging.info('plainGP score shape %s'%str(score.shape))
        return {'Ymean':Ypred, 'Ystd':Ystd, 'score':score}


class GPensemble(GP):
    def __init__(self, fn, percent=0.05, selection='random', allsbs=True, verbose=False, initialize=True, minConditionsPerRx=1, randfloat=None, numModels=100, inputOrder=None, normalized=False, perturbed=False, noise=0.1):
        self.normalized = normalized
        self.perturb = perturbed
        self.allsbs = allsbs
        self.make_score = False
        self.verbose = verbose
        self.fulldata, self.inputDataNames = self.getSelectedData(fn, percent=percent, selection=selection, allsbs=allsbs, minConditionsPerRx=minConditionsPerRx, 
                            randfloat=randfloat, inputOrder=inputOrder)
        if perturbed:
           lower = 1 - noise
           higher = noise + 1
           logging.info('GPE: perturbing with %.1f noise (vals from %.1f to %.1f'%(noise, lower, higher))  
           self.fulldata['perturb'] = self.fulldata.groupby('conditions')['yield'].transform(lambda x: np.random.random()*2*noise+1-noise)
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
        tf.keras.backend.clear_session()
        for i,m in enumerate(self.models):
            model, weight = self.trainModel( weight=self.init_weight_list[i])
            self.models[i]=model

    def predict(self, numSamples=100):
        #evaluate status
        fullX = self.fulldata[ self.inputDataNames].values
        samples = []
        for model in self.models:
            predYsamples = model.predict_f_samples(fullX, num_samples=numSamples)
            YpredSamples = predYsamples.numpy()
            YpredSamples = numpy.reshape( YpredSamples, YpredSamples.shape[0:2] )
            #print(YpredSamples.shape)
            samples.append(YpredSamples)
        full = numpy.concatenate( samples)
        mean = numpy.mean(full, axis=0)
        std = numpy.std(full, axis=0)
        if self.make_score:
           logging.info('computing score')
           train_mask = self.fulldata.training == True
           score = calc_GPE_score(self.models, fullX, train_mask, self.fulldata)
           logging.info('GPE score shape %s'%str(score.shape))
           print('GPE score shape %s'%str(score.shape))
        else:
           score = std
        return {'Ymean':mean, 'Ystd':std, 'score':score}







if __name__ == "__main__":
    #net = NN('inputFeatures.withinp', percent=0.1, allsbs=True, verbose=True)
    #hist = net.trainModel()
    #b=ensembleNN('inputFeatures.withinp', percent=0.1, allsbs=True, verbose=True, minConditionsPerRx=2)
    #b = GP('inputFeatures.withinp', percent=0.1, allsbs=True, verbose=True, minConditionsPerRx=2)
    b = GPensemble('inputFeatures.withinp', percent=0.1, allsbs=True, verbose=True, minConditionsPerRx=2, randfloat=0.34, numModels=5)
    #print("MODELS", b.models)
    pred = b.predict()
    #print( type(pred), pred.keys(), "SHAPES", pred )
    print("==",  dir(b.models[0]) )
