import numpy as np
from typing import Dict, Optional, Tuple
import tensorflow as tf
import gpflow
from gpflow import set_trainable
from gpflow.mean_functions import Constant
from gpflow.inducing_variables import InducingPoints
from gpflow.utilities import positive, print_summary, to_default_float
from gpflow.utilities.ops import broadcasting_elementwise
import logging
import yaml
#tf.logging.set_verbosity(tf.logging.ERROR)


autotune = tf.data.experimental.AUTOTUNE

class DotProd(gpflow.kernels.Kernel):
        def __init__(self, ls=None):
            super().__init__()
            # We constrain the value of the kernel variance to be positive when it's being optimised
            self.variance = gpflow.Parameter(0.1, transform=positive())
            self.do_rbf = not isinstance(ls, type(None))
            if self.do_rbf:
               self.length_scale = gpflow.Parameter(1.0, transform=positive())

        def K(self, X, X2=None):
            """
            Compute the Dot product kernel matrix σ² * (<x, y>)

            :param X: N x D array
            :param X2: M x D array. If None, compute the N x N kernel matrix for X.
            :return: The kernel matrix of dimension N x M
            """
            if X2 is None:
                X2 = X
            X_ = tf.math.l2_normalize(X, axis=1)
            X2_ = tf.math.l2_normalize(X2, axis=1)
            outer_product = tf.tensordot(X_, X2_, [[-1], [-1]])  # outer product of the matrices X and X2

            if self.do_rbf:
               outer_product = tf.exp((outer_product-1)/self.length_scale)

            return self.variance * outer_product

        def K_diag(self, X):
            """
            Compute the diagonal of the N x N kernel matrix of X
            :param X: N x D array
            :return: N x 1 array
            """
            return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))


class ScaledDotProd(gpflow.kernels.Kernel):
        def __init__(self, lens, do_rbf=True):
            super().__init__()
            # We constrain the value of the kernel variance to be positive when it's being optimised
            self.variance = gpflow.Parameter(0.1, transform=positive())
            self.do_rbf = do_rbf
            mask = self.make_mask(lens)
            self.mask = gpflow.Parameter(mask, trainable=False, dtype=np.float64)             
            self.length_scales = gpflow.Parameter([[1.0] for _ in lens], transform=positive(), dtype=np.float64)


        def make_mask(self, lens):
            cumsum = np.cumsum(lens)
            mask = np.zeros((cumsum[-1], len(lens)))
            for i,x in enumerate(cumsum):
               prev = 0 if i==0 else cumsum[i-1]
               mask[prev:x,i]=1
            return mask
               

        def K(self, X, X2=None):
            """
            Compute the Dot product kernel matrix σ² * (<x, y>)

            :param X: N x D array
            :param X2: M x D array. If None, compute the N x N kernel matrix for X.
            :return: The kernel matrix of dimension N x M
            """
            if X2 is None:
                X2 = X
            mask = tf.squeeze(tf.matmul(self.mask, self.length_scales))
            Xscaled = X*mask
            outer_product = tf.tensordot(Xscaled, X2, [[-1], [-1]])  # outer product of the matrices X and X2

            sum_scales = tf.reduce_sum(self.length_scales)

            if self.do_rbf:
               outer_product = tf.exp(outer_product-sum_scales)
            else:
               outer_product = outer_product/sum_scales

            return self.variance * outer_product

        def K_diag(self, X):
            """
            Compute the diagonal of the N x N kernel matrix of X
            :param X: N x D array
            :return: N x 1 array
            """
            return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))


default_mlp_config = {'embedding':100, 'l2':0.05, 'dropout':0.3, 
                  'batch_size':32, 'hidden_n':56, 'hidden_l':1, 
                  'output_dim':10, 'input_dim':(512,), 
                  'output_act':'relu', 'input_act':'linear',
                  'input_bias':False}


def fill_missing_dict_keys(config, reference=default_mlp_config):
   for k in reference:
      if k not in config:
         val = config.setdefault(k, reference[k])
         logging.info('Missing key %s, updating with value %s'%(str(k), str(val)))


def make_mlp_model(config=default_mlp_config, weight=None, **kwargs):
   config2 = config.copy()
   config2.update(kwargs)
   fill_missing_dict_keys(config2)
   #logging.debug('Building MLP from config:\n' + yaml.dump(config2))

   inp = tf.keras.layers.Input(shape=config2['input_dim'], batch_size=config2['batch_size'])
   
   x = tf.keras.layers.Dense(config2['embedding'], activation=config2['input_act'],
         use_bias=config2['input_bias'], 
         kernel_regularizer=tf.keras.regularizers.l2(config2['l2']))(inp)
   x = tf.keras.layers.Dropout(config2['dropout'])(x)
   for _ in range(config2['hidden_l']):
      x = tf.keras.layers.Dense(config2['hidden_n'], activation="relu", 
         kernel_regularizer=tf.keras.regularizers.l2(config2['l2']), use_bias=True)(x)
      x = tf.keras.layers.Dropout(config2['dropout'])(x)
   x = tf.keras.layers.Dense(config2['output_dim'], activation=config2['output_act'],
      kernel_regularizer=tf.keras.regularizers.l2(config2['l2']), use_bias=True)(x)
   x = tf.keras.layers.Lambda(to_default_float)(x)
   model = tf.keras.Model(inputs=inp, outputs=x) 
   if weight:
      model.set_weights( weight)
   #print("MODelkernelWeight", model.get_weights() )
   return model, model.get_weights()


def make_keras_model(config=default_mlp_config, weight=None, **kwargs):
   config2 = config.copy()
   config2.update({'output_act':'relu', 'input_act':'linear', 'input_bias':False})
   return make_mlp_model(config2, weight=weight, **kwargs)


def make_keras_model_mean(config=default_mlp_config, **kwargs):
   config2 = config.copy()
   config2.update({'output_act':'linear', 'output_dim':1, 'input_act':'linear', 'input_bias':False})
   return make_mlp_model(config2, **kwargs)


class KernelWithNN(gpflow.kernels.Kernel):
    def __init__(
        self,
        image_shape: Tuple,
        output_dim: int,
        base_kernel: gpflow.kernels.Kernel,
        batch_size: Optional[int] = None,
        config: Dict={},
        weight = None,
    ):
        super().__init__()
        with self.name_scope:
            self.base_kernel = base_kernel
            input_size = int(tf.reduce_prod(image_shape))
            input_dim = (input_size,)
            other = dict(input_dim=input_dim, output_dim=output_dim, batch_size=batch_size)
            if config=={}:
               self.cnn, self.init_weight = make_keras_model(weight=weight,  **other)
            else:
               self.cnn, self.init_weight = make_keras_model(config=config, weight=weight, **other)

            self.cnn.build((None, input_size))

    def K(self, a_input: tf.Tensor, b_input: Optional[tf.Tensor] = None) -> tf.Tensor:
        transformed_a = self.cnn(a_input)
        transformed_b = self.cnn(b_input) if b_input is not None else b_input
        return self.base_kernel.K(transformed_a, transformed_b)

    def K_diag(self, a_input: tf.Tensor) -> tf.Tensor:
        transformed_a = self.cnn(a_input)
        return self.base_kernel.K_diag(transformed_a)


class KernelSpaceInducingPoints(gpflow.inducing_variables.InducingPoints):
    pass


@gpflow.covariances.Kuu.register(KernelSpaceInducingPoints, KernelWithNN)
def Kuu(inducing_variable, kernel, jitter=None):
    func = gpflow.covariances.Kuu.dispatch(
        gpflow.inducing_variables.InducingPoints, gpflow.kernels.Kernel
    )
    return func(inducing_variable, kernel.base_kernel, jitter=jitter)


@gpflow.covariances.Kuf.register(KernelSpaceInducingPoints, KernelWithNN, object)
def Kuf(inducing_variable, kernel, a_input):
    return kernel.base_kernel(inducing_variable.Z, kernel.cnn(a_input))


def make_kernel_from_config(config, weight=None):
     kernel_base = config['kernel_base']
     kernel_cfg = config.get('kernel_cfg', {})
     dot_rbf = config.get('dot_rbf', False)
     do_nn = config.get('kernel_NN', kernel_cfg!={})
     logging.debug('make_kernel_from_config: \n%s'%yaml.dump(config))
     logging.debug('make_kernel_from_config: do_nn=%s'%str(do_nn))
     # choose base kernel
     if kernel_base=='rbf':
        base = gpflow.kernels.SquaredExponential()
     elif kernel_base == 'dot':
        ls = 1.0 if dot_rbf else None
        base = DotProd(ls)
     elif kernel_base=='matern32':
        base = gpflow.kernels.Matern32()
     elif kernel_base=='matern52':
        base = gpflow.kernels.Matern52()
     else:
        raise KeyError('Base kernel should be defined')

     # make kernel
     # weight = None
     if not do_nn:
        k = base
        logging.debug('Fixed kernel')
     else:
        fill_missing_dict_keys(kernel_cfg)
        input_shape = kernel_cfg['input_dim']
        batch_size = kernel_cfg['batch_size']
        output_dim = kernel_cfg['output_dim']
        k = KernelWithNN( input_shape, output_dim, base, batch_size=batch_size, config=kernel_cfg, weight=weight)
        weight = k.init_weight
        logging.debug('NN kernel')
     return k, weight


def make_GP_model_from_cfg(config, data, kernel=None, support=None):
     '''Builds GP model and training_loss according to config'''
     logging.debug('Making GP model from config:\n'+yaml.dump(config))
     model_kind = config['model_kind']
     kernel_cfg = config.get('kernel_cfg', {})
     fill_missing_dict_keys(kernel_cfg)
     input_shape = kernel_cfg['input_dim']
     batch_size = kernel_cfg['batch_size']
     output_dim = kernel_cfg['output_dim']

     if kernel==None:
        k = make_kernel_from_config(config)
     else:
        k = kernel
        logging.debug('Using predefined kernel')
     X ,Y = data
     if model_kind=='svgp':
        #inducing_points
        assert not isinstance(support, type(None)),\
               'Support required to get inducing points'
        num_inducing_points = config['inducing_points']
        support_indices = np.random.choice(np.arange(support.shape[0]), 
                                           num_inducing_points,replace=False)
        inducing_variable_raw = support[support_indices]
        if kernel_cfg!={}:
           inducing_variable_raw = k.cnn(inducing_variable_raw)
        inducing_variable = KernelSpaceInducingPoints(inducing_variable_raw)
        
        # make model
        m = gpflow.models.SVGP(mean_function=Constant(Y.mean()), kernel=k, likelihood=gpflow.likelihoods.Gaussian(),
               inducing_variable=inducing_variable)
      
        # make data iterator
        data_minibatch = (
             tf.data.Dataset.from_tensor_slices((X, Y ))
             .prefetch(autotune)
             .repeat()
             .shuffle(X.shape[0])
             .batch(batch_size)
              )
        data_iterator = iter(data_minibatch)
        training_loss = m.training_loss_closure(data_iterator)
     elif model_kind=='vgp':
         m = gpflow.models.VGP(data=data, mean_function=Constant(np.mean(Y)), kernel=k, likelihood=gpflow.likelihoods.Gaussian(variance=0.01))
         training_loss = m.training_loss_closure()
     else:
         m = gpflow.models.GPR(data=data, mean_function=Constant(np.mean(Y)), kernel=k, noise_variance=config.get('noise',0.1))
         training_loss = m.training_loss_closure()
      
     return m, training_loss
