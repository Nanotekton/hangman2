import tensorflow as tf
import numpy as np
import logging

#@tf.function ?
def compute_values_and_gradients(model, test_points):
   all_grads = []
   all_ys = []

   for i in range(test_points.shape[0]):
      with tf.GradientTape() as tape:
         y = model(test_points[i:i+1])
         all_ys.append(y)
         grads = tape.gradient(y, model.trainable_variables)
         grads = tf.concat([tf.reshape(x,(1,-1)) for x in grads], axis=1)
         all_grads.append(grads)
   all_grads = tf.concat(all_grads, axis=0).numpy()
   all_ys = tf.concat(all_ys, axis=0).numpy()

   return all_ys, all_grads

def compute_values_and_gradients2(model, test_points):
   with tf.GradientTape() as tape:
      y = model(test_points)
   
   grads_raw = tape.jacobian(y, model.trainable_variables)
   flat = tf.keras.layers.Flatten()
   grads = tf.concat([flat(x) for x in grads_raw], axis=-1)

   return y.numpy(), grads.numpy()
   

def make_synthetic_data(): #(train_x, train_y), (test_x, test_y)
   N = 500
   train_x, test_x = [np.random.normal(scale=1, size=(N,2)) for i in range(2)]
   train_x+=np.array([2,1])
   test_y = -np.cos((test_x**2).sum(axis=1)**0.5) + np.random.normal(scale=0.1, size=(N,))
   train_y = -np.cos((train_x**2).sum(axis=1)**0.5) + np.random.normal(scale=0.1, size=(N,))
   return (train_x, train_y), (test_x, test_y)

def make_mlp():
   x = tf.keras.layers.Input(shape=(2,))
   h = tf.keras.layers.Dense(10, activation='relu')(x)
   h = tf.keras.layers.Dropout(0.2)(h)
   o = tf.keras.layers.Dense(1, activation='relu')(h)
   model = tf.keras.models.Model(inputs=x, outputs=o)
   model.compile(optimizer='adam', loss='mse', metrics=['mae'])
   return model

def compute_ensemble_scores(ensemble, test, optimum, mode):
   optimum_vals = []
   test_vals, F, Fyn, Fys, Fys_yn = [], [], [], [], []

   if mode=='grad':
      func = compute_values_and_gradients
   elif mode=='jacobian':
      func = compute_values_and_gradients2
   else:
      raise ValueError('unknown mode %s'%str(mode))

   for model in ensemble:
      opt_v, opt_grad = func(model, optimum)
      if opt_v.shape[0]>1:
         opt_v, opt_grad = opt_v.mean(axis=0), opt_grad.mean(axis=0)

      test_v, test_g = func(model, test) #shape n_test x n_param
      test_v = test_v.reshape(-1)
      opt_v = opt_v.reshape(-1)
      assert opt_v.shape[0]==1
      opt_v = opt_v[0]

      optimum_vals.append(opt_v)
      test_vals.append(test_v)

      logging.debug('Opt v, test_v: %s %s'%(str(opt_v.shape), str(test_v.shape)))
      logging.debug('Opt grad, test_grad: %s %s'%(str(opt_grad.shape), str(test_g.shape)))

      F_v = (test_g*opt_grad).sum(axis=1)
      logging.debug('F_V %s'%str(F_v.shape))
      F.append(F_v)
      
      Fyn_v = F_v*test_v
      logging.debug('Fyn_V %s'%str(Fyn_v.shape))
      Fyn.append(Fyn_v)

      Fys_v = F_v*opt_v
      logging.debug('Fys_V %s'%str(Fys_v.shape))
      Fys.append(Fys_v)

      Fys_yn_v = Fys_v * test_v
      logging.debug('Fys_yn_V %s'%str(Fys_yn_v.shape))
      Fys_yn.append(Fys_yn_v)

   optimum_av, optimum_std = np.mean(optimum_vals), np.std(optimum_vals)
   test_av, test_std = np.mean(test_vals, axis=0), np.std(test_vals, axis=0)

   F = np.mean(F, axis=0)
   Fyn = np.mean(Fyn, axis=0)
   Fys = np.mean(Fys, axis=0)
   Fys_yn = np.mean(Fys_yn, axis=0)

   scores = Fys_yn - Fys*test_av - Fyn*optimum_av + F*test_av*optimum_av

   return {'scores':scores, 'F':F, 'optimum_av':optimum_av, 'optimum_std':optimum_std, 'test_av':test_av, 'test_std':test_std}

if __name__=='__main__':
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--mode', type=str, choices=['grad', 'jacobian'], default='grad')
   parser.add_argument('--out', type=str, default='nne_pvr4.csv')
   args = parser.parse_args()
   

   logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(message)s')
   (train_x, train_y), (test_x, test_y) = make_synthetic_data()
   ensemble = [make_mlp() for _ in range(50)]
   train_preds = []
   test_preds = []
   for i, model in enumerate(ensemble):
      model.fit(train_x, train_y, batch_size=10, epochs=100, verbose=True)
      logging.info('Model %i fit'%i)
      train_preds.append(model.predict(train_x))
      test_preds.append(model.predict(test_x))

   train_unc = np.std(train_preds, axis=0)
   train_preds = np.mean(train_preds, axis=0)
   test_preds = np.mean(test_preds, axis=0)
   if train_preds.min()<test_preds.min():
      optimum_idx = train_preds.argmin()
      optimum = train_x[optimum_idx:optimum_idx+1]
   else:
      optimum_idx = test_preds.argmin()
      optimum = test_x[optimum_idx:optimum_idx+1]
   
   logging.info('computing scores for test, %s'%str(test_x.shape))
   result = compute_ensemble_scores(ensemble, test_x, optimum, args.mode)
   logging.info('computing scores for train, %s'%str(train_x.shape))
   result_train = compute_ensemble_scores(ensemble, train_x, optimum, args.mode)
   logging.info('writing output')

   with open(args.out, 'w') as f:
      f.write('optimum_loc:%s\n'%str(optimum))
      f.write('optimum_av:%f\n'%result['optimum_av'])
      f.write('optimum_unc:%f\n'%result['optimum_std'])
      f.write('x;y;F;score;unc;err;flag\n')
      
      for i,vec in enumerate(test_x):
         x,y=vec
         F = result['F'][i]
         score = result['scores'][i]
         unc = result['test_std'][i]
         err = test_preds[i]-test_y[i]
         f.write('%f;%f;%f;%f;%f;%f;test\n'%(x,y,F,score,unc,err))

      for i,vec in enumerate(train_x):
         x, y = vec
         F = result_train['F'][i]
         score = result_train['scores'][i]
         unc = train_unc[i]
         err = train_preds[i] - train_y[i]
         f.write('%f;%f;%f;%f;%f;%f;train\n'%(x,y,F,score,unc,err))
   
   logging.info('input echo  %s'%str(args.__dict__))
