import numpy, random, os, sys, abc
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, regularizers, utils
from sklearn import preprocessing

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
import logging

######
from sklearn.model_selection import train_test_split
from vectorize import make_reagent_encoder, vectorize_unique_substrates, fgps_to_freq_vec
from loading import load_spreadsheet
#make_reagent_encoder(source='shared_file', df=None, cols=['ligand', 'solvent', 'base', 'temperature']):
#vectorize_unique_substrates(source='shared_file'):
#fgps_to_freq_vec(fgp_list, cutoff=0, set_place_for_unk=False, ranking=None):
######

class GP():
    def __init__(self, Xtrain, Ytrain):
        inputDim = Xtrain.shape[1]
        self.config = {'kernel_base': 'matern32', 'model_kind': 'vgp', 'mean_NN': False, 'noise':0.01, 'kernel_NN':True, 'dot_rbf': False, 'num_inducing_points': None,
                'kernel_cfg':{'embedding':100, 'l2':0.05, 'dropout':0.845918563446334, 'batch_size':128, 'hidden_n':200, 'hidden_l':1, 'output_dim':10, 'input_dim':(inputDim,), 
                              'output_act':'relu', 'input_act':'relu', 'input_bias':True},
                'train_cfg':{ 'iter_mean': None, 'iter': 10, 'iter_internal': 5, 'iter_post': 5}}
        self.model, self.init_weight = self.trainModel(Xtrain, Ytrain)


    def trainModel(self, Xtrain, Ytrain, weight=None):
        kernel, weight = gp_module.make_kernel_from_config(self.config, weight=weight)
        adam_opt = tf.optimizers.Adam(0.001) #re-initialize in each fold
        #print("SHAPES", self.X.shape, self.Y.shape, "NEW IDXES", len(self.idxesNew) )
        Ytrain = Ytrain.reshape(-1,1)
        #print("TRAINS", Xtrain.shape, Ytrain.shape)
        #m, training_loss = gp_module.make_GP_model_from_cfg(self.config, (self.X[self.idxesNew], self.Y[self.idxesNew]), kernel=kernel, support=None)
        #print('WTF??',np.mean(Ytrain))
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

    def predict(self, X):
        #evaluate status
        predY, predY_var = self.model.predict_f(X)
        Ypred = predY.numpy()
        print("SHAPES pred in, out, out", X.shape, Ypred.shape, predY_var.shape,)
        Ystd = np.sqrt(predY_var.numpy())
        return {'Ymean':Ypred, 'Ystd':Ystd}


class GPensemble(GP):
    def __init__(self, Xtrain, Ytrain, numModels=100):
        inputDim = Xtrain.shape[1]
        print("INPUT DIM", inputDim)
        #old: l2=0.001, dropout=0.3 batch_size=32 hidden_n=56
        dropOut = 0.845918563446334
        self.config = {'kernel_base': 'matern32', 'model_kind': 'vgp', 
                'mean_NN': False, 'noise':0.01, 'kernel_NN':True, 'dot_rbf': False, 'num_inducing_points': None,
        #self.config = {'kernel_base': 'matern32', 'model_kind': 'gp', 'mean_NN': False, 'noise':0.01, 'kernel_NN':False, 'dot_rbf': False, 'num_inducing_points': None,
                'kernel_cfg':{'embedding':100, 'l2':0.05, 'dropout':dropOut, 'batch_size':128, 'hidden_n':200, 'hidden_l':1, 'output_dim':10, 'input_dim':(inputDim,), 
                              'output_act':'relu', 'input_act':'relu', 'input_bias':True},
                'train_cfg':{ 'iter_mean': None, 'iter': 10, 'iter_internal': 5, 'iter_post': 5}}
        self.models = []
        self.init_weight_list = []
        for i in range(numModels):
            model, weight = self.trainModel(Xtrain, Ytrain)
            self.models.append(model)
            self.init_weight_list.append(weight)
            if (i+1)%10==0:
                logging.info('model done: %i/%i'%(i,numModels))

    def predict(self, fullX, numSamples=100):
        #evaluate status
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
        return {'Ymean':mean, 'Ystd':std}

class plainGP(GP):
    def __init__(self, Xtrain, Ytrain, numModels=1, kind='gp'):
        inputDim = Xtrain.shape[1]
        print("INPUT DIM", inputDim)
        #old: l2=0.001, dropout=0.3 batch_size=32 hidden_n=56
        dropOut = 0.845918563446334
        self.config = {'kernel_base': 'matern32', 'model_kind': kind, 'mean_NN': False, 'noise':0.01, 'kernel_NN':False, 'dot_rbf': False, 'num_inducing_points': None,
        #self.config = {'kernel_base': 'matern32', 'model_kind': 'gp', 'mean_NN': False, 'noise':0.01, 'kernel_NN':False, 'dot_rbf': False, 'num_inducing_points': None,
                'kernel_cfg':{'embedding':100, 'l2':0.05, 'dropout':dropOut, 'batch_size':128, 'hidden_n':200, 'hidden_l':1, 'output_dim':10, 'input_dim':(inputDim,), 
                              'output_act':'relu', 'input_act':'relu', 'input_bias':True},
                'train_cfg':{ 'iter_mean': None, 'iter': 10, 'iter_internal': 5, 'iter_post': 5}}
        self.models = []
        self.init_weight_list = []
        for i in range(numModels):
            model, weight = self.trainModel(Xtrain, Ytrain)
            self.models.append(model)
            self.init_weight_list.append(weight)
            if (i+1)%10==0:
                logging.info('model done: %i/%i'%(i,numModels))

    def predict(self, fullX, numSamples=100):
        #evaluate status
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
        return {'Ymean':mean, 'Ystd':std}

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['GPE','plainGP'])
    parser.add_argument('--reversed', action='store_true')
    parser.add_argument('--kind', type=str, choices=['gp', 'vgp'], default='gp')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(name)s:%(asctime)s- %(message)s')
    mida_col = 'boronate/boronic ester smiles'
    bromide_col = 'bromide smiles'
    columns = ["base", "solvent", "ligand", "temperature"]
    data, status = load_spreadsheet()
    data = data[~data['yield'].isna()]
    logging.info('loaded with status %s, records=%i'%(status, data.shape[0]))
    
    mida, bromide = vectorize_unique_substrates()
    mida_vecs, _ = fgps_to_freq_vec(mida[1])
    bromide_vecs, _ = fgps_to_freq_vec(bromide[1])
    mida_translate = dict(zip(mida[0], mida_vecs))
    bromide_translate = dict(zip(bromide[0], bromide_vecs))
    logging.info('unique substeates vectorized')
    
    vectors_m = [mida_translate[m] for m in data[mida_col].values]
    vectors_b = [bromide_translate[m] for m in data[bromide_col].values]
    
    conditions_encoder = make_reagent_encoder(cols=columns)
    conditions = conditions_encoder.transform(data[columns].values)
    Y = data['yield'].values.reshape(-1,1)
    uY, sY = Y.mean(), Y.std()
    Y = (Y-uY)/sY
    #sY = 1
    
    X = np.hstack([vectors_m, vectors_b, conditions])
    non_zero_idx = np.where(X.std(axis=0)>0)[0]
    logging.info('non zero idx: %i/%i'%(len(non_zero_idx), X.shape[1]))
    X = X[:,non_zero_idx]
    logging.info('arrays assembled')
    
    #X = X.astype(np.float32)
    #Y = Y.astype(np.float32)
    if args.reversed:
        tsize=0.8
    else:
        tsize=0.2
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=tsize)
    logging.info('train len: %i, test_len: %i'%(Xtrain.shape[0], Xtest.shape[0]))

    if args.model=='GPE':
        GPE = GPensemble(Xtrain, Ytrain) # predict returns {'Ymean':mean, 'Ystd':std}
    elif args.model=='plainGP':
        GPE = plainGP(Xtrain, Ytrain)
    else:
        raise ValueError('unknown model:%s'%args.model)

    logging.info('prediction:')
    pred_train = GPE.predict(Xtrain)
    pred_test = GPE.predict(Xtest)
    
    mae_train = abs(pred_train['Ymean'] - Ytrain.reshape(-1)).mean()*sY
    unc_train = pred_train['Ystd'].mean()*sY
    logging.info('MAE train: %.3f UNC train: %.3f '%(mae_train, unc_train))
    
    mae_test = abs(pred_test['Ymean'] - Ytest.reshape(-1)).mean()*sY
    unc_test = pred_test['Ystd'].mean()*sY
    logging.info('MAE test: %.3f UNC test: %.3f '%(mae_test, unc_test))
