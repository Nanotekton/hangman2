import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ground_truth', type=str, default='prediction_full_space_2021-Jul-22-02.49.28.csv')
parser.add_argument('--app', type=str, default='')
parser.add_argument('--totally_random', action='store_true')
parser.add_argument('--partially_random', action='store_true')
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--repetitions', type=int, default=100)
parser.add_argument('--pivot', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=72)
parser.add_argument('--batch_schedule', type=int, nargs='+', default=[])
parser.add_argument('--n_cond', type=int, default=18)
parser.add_argument('--max_iter', type=int, default=None)
parser.add_argument('--num_models', type=int, default=100)
args = parser.parse_args()

import numpy as np
import pandas as pd
from df_stats import calc_probability_of_improvement, get_top_from_full_space, HEADER, propose_batch
from vectorize import make_full_space_df, prepare_substrate_vect_dict, make_reagent_encoder, dataframe_to_encoded_array
import logging
from copy import deepcopy
from test_gpe import GPensemble, plainGP
import scipy
import datetime
import json
from os import popen
from os.path import isfile
import pickle
from tensorflow.keras.backend import clear_session
from multiprocessing import Queue, Process
now = datetime.datetime.now().strftime('%Y-%b-%d-%H.%M.%S')+args.app


#logging.basicConfig(level=logging.INFO, format='%(name)s:%(asctime)s-%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(filename='simulation_%s.log'%now)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(name)s:%(asctime)s-%(message)s')
for H in [fh, sh]:
        H.setLevel(logging.INFO)
        H.setFormatter(formatter)
        logger.addHandler(H)

#1. load the data
logger.info('input params: %s'%json.dumps(args.__dict__))
ground_truth = pd.read_csv(args.ground_truth, sep='\t')
N = ground_truth.shape[0]
logger.info('ground_truth data records: %i'%ground_truth.shape[0])
mask = ~ground_truth.training#ground_truth.Yexp.isna()
logger.info(f'missing {mask.mean()*100:.1f} % exp. records, filling with predictions')
#ground_truth.loc[mask, 'Yexp'] = ground_truth.loc[mask, 'Ypred']
ground_truth['Yexp'] = ground_truth['Ypred']

#1. Computer parameters (n steps etc)
if args.batch_schedule==[]:
   STEPS = int(N/args.batch_size)
   if N%args.batch_size!=0:
     STEPS +=1
   if args.max_iter!=None:
      STEPS = min([STEPS, args.max_iter])
   SCHEDULE = [args.batch_size for _ in range(STEPS)]
else:
   cumsum = np.cumsum(args.batch_schedule)
   STEPS = 0
   for i,x in enumerate(cumsum):
      if x<=N or cumsum[i-1]<N:
         STEPS+=1
   SCHEDULE = args.batch_schedule[:STEPS]
COND = args.n_cond

logging.info(f'using iterations={STEPS} with #conditions per iteration={COND}')

#2. prepare space; reagents get from the shared file, conditions - current_space_dict.json
substrates_cols = ['boronate/boronic ester smiles', 'bromide smiles', 'product_smiles']
mida_col, bromide_col = substrates_cols[:2]
conditions_cols = ['solvent', 'temperature', 'base', 'ligand']

#3. prepare substrate and conditions vectorizer
mida_dict, bromide_dict = prepare_substrate_vect_dict()
substrate_encoder_dicts = [(mida_col, mida_dict), (bromide_col, bromide_dict)]
reagent_encoder = make_reagent_encoder(source='space_dict.json', cols=conditions_cols)

#4. Make training vector_matrices
full_trainX = dataframe_to_encoded_array(ground_truth, substrate_encoder_dicts, reagent_encoder, conditions_cols)
idx = np.where(full_trainX.std(axis=0)>0)[0]
full_trainX = full_trainX[:,idx]

#5. mark conditions etc
all_cols = substrates_cols + conditions_cols
reaction_cols = conditions_cols + ['product_smiles']
ground_truth['condition_string'] = ground_truth[conditions_cols].agg(lambda x: ',,'.join(str(y) for y in x), axis=1)
ground_truth['reaction_string'] = ground_truth[reaction_cols].agg(lambda x: ',,'.join(str(y) for y in x), axis=1)

def get_best_conditions(df, group_key='condition_string', evaluation_key='Yexp', evaluation_mode='mean'):
   best = df.groupby(group_key).agg({evaluation_key:evaluation_mode})
   best.sort_values(evaluation_key, ascending=False, inplace=True)
   return best

def make_space_df(ground_truth):
   substrate_space_df = ground_truth[substrates_cols].drop_duplicates()
   space_df = make_full_space_df(substrate_space_df, 'current_space_dict.json', substrates_cols=substrates_cols)
   space_df['training'] = False
   space_df['condition_string'] = space_df[conditions_cols].agg(lambda x: ',,'.join(str(y) for y in x), axis=1)
   space_df['reaction_string'] = space_df[reaction_cols].agg(lambda x: ',,'.join(str(y) for y in x), axis=1)
   spaceX = dataframe_to_encoded_array(space_df, substrate_encoder_dicts, reagent_encoder, conditions_cols)
   spaceX = spaceX[:,idx]
   return space_df, spaceX

##### SIMULATION PART #######

def simulate(full_trainX, ground_truth, simulation_steps=7, repId=0, batch_size=72, batch_schedule=None, cond_per_batch=18, totally_random=False, partially_random=False, rng=None, normalize=False):
   if rng==None:
      rng = np.random.RandomState()
   all_indices = np.arange(ground_truth.shape[0])
   seen_indices = list(rng.choice(all_indices, batch_size if batch_size>1 else 36, replace=False))
   
   if batch_schedule==None:
      batch_schedule = [batch_size for _ in range(simulation_steps)]

   true_best_conditions = get_best_conditions(ground_truth).index[0]
   logging.info(f'Repetition ID {repId} : True best conditions: {true_best_conditions}')

   ranks = []
   space_df, spaceX = make_space_df(ground_truth)
   space_df.set_index('reaction_string', inplace=True)
   ground_truth.set_index('reaction_string', inplace=True)
   ground_truth['Yexp'] = ground_truth['Yexp'].clip(0,1)
   space_df['Yexp'] = ground_truth['Yexp']
   ground_truth.reset_index(inplace=True)
   space_df.reset_index(inplace=True)

   for iteration, batch_size in enumerate(batch_schedule):
      trainX = full_trainX[seen_indices]
      Y = ground_truth['Yexp'].values[seen_indices].reshape(-1,1)
      if normalize:
         logging.info(f'before max, max={Y.max()}')
         max_vals = ground_truth.iloc[seen_indices].groupby('condition_string')['Yexp'].transform(max).values.reshape(-1,1)
         max_vals = np.where(max_vals>0, max_vals,1)
         Y = Y/max_vals
         logging.info(f'max done, max={Y.max()}')
         check = ground_truth.iloc[seen_indices]
         check['Ycheck'] = Y.reshape(-1)
         check = check.groupby('condition_string').agg({'Ycheck':max}).values
         ncheck = len(check)
         non_zero = np.array([x for x in check if x!=0])
         #logging.info(f'check: {check}')
         logging.info(f'ncond={ncheck}, nonzero={non_zero.shape[0]}')
         logging.info(f'check: mean of cond={non_zero.mean():.1f}, std of cond={non_zero.std():.1f}')

      uY, sY = Y.mean(), Y.std()
      Y = (Y-uY)/sY
      GPE = GPensemble(trainX, Y, numModels=args.num_models) #smaller ensemble for tests
      logger.info(f'Repetition ID {repId}: iter {iteration} trained')

      #8. Predict
      pred = GPE.predict(spaceX)
      logger.info(f'Repetition ID {repId}: iter {iteration} predicted')

      space_df['Ypred'] = pred['Ymean']*sY + uY
      space_df['Yunc'] = pred['Ystd']*sY
      space_df.loc[space_df.reaction_string.isin(ground_truth.reaction_string[seen_indices]), 'training'] = True

      #8a. Control
      train_pred = GPE.predict(trainX)['Ymean'].reshape(-1)*sY + uY
      Ytrue = Y.reshape(-1)*sY+uY
      logger.info(f'Repetition ID {repId}: train mae: {abs(Ytrue-train_pred).mean():.3f}')

      check = space_df[space_df.training]
      mae_train = abs(check['Yexp'] - check['Ypred']).mean()
      logging.info(f'Repetition ID {repId}: mae train(space df): {mae_train:.3f}')
      logging.info(f'Repetition ID {repId}: average unc: {space_df["Yunc"].mean()}')
      sorted_unc = space_df.groupby('condition_string').agg({'Ypred':'mean', 'Yunc':'mean'}).sort_values('Ypred', ascending=False)
      sorted_unc = sorted_unc['Yunc'].values
      logging.info(f'Repetition ID {repId}: average unc(top-3): {sorted_unc[:3].mean():.5f} max unc(top-3): {sorted_unc[:3].max():.5f}')
      logging.info(f'Repetition ID {repId}: average unc(top-10): {sorted_unc[:10].mean():.5f} max unc(top-10): {sorted_unc[:10].max():.5f}')

      #9a. Get current best conditions
      condition_ranking = get_best_conditions(space_df, evaluation_key='Ypred')
      if true_best_conditions in condition_ranking.index:
         current_rank = (condition_ranking.index==true_best_conditions).argmax()
         ranks.append(current_rank)
         logging.info(f'Repetition ID {repId} iteration {iteration} - rank of the best: {current_rank}')
      else:
         logging.info(f'Repetition ID {repId} :WARNING: true best not in ranking')
         ranks.append(None)

      #9b. Sample. Strategy: 18 conds, 4 subs
      logging.info(f'Repetition ID {repId}: selecting {batch_size} rxs with {cond_per_batch} conditions')
      if totally_random:
         logging.info('totally random selection')
         pool = [x for x in all_indices if x not in seen_indices]
         if len(pool)>batch_size:
            to_add = list(rng.choice(pool, batch_size, replace=False))
         else:
            to_add = pool
      else:
         if partially_random:
            logging.info('partially random selection')
            space_df['Yunc'] = rng.random(space_df.shape[0])
            space_df['PI'] = space_df.groupby('condition_string')['Yunc'].transform(lambda x: rng.random())
         else:
            logging.info('PI/MaxUnc selection')
            calc_probability_of_improvement(space_df)
         experiments = propose_batch(space_df, Nbatch=batch_size, Ncond=cond_per_batch)
         logging.info(f'Repetition ID {repId}: new experiments: {experiments.shape[0]}, seen={100*experiments.shape[0]/ground_truth.shape[0]:.3f} %') 
         to_add = ground_truth.index[ground_truth.reaction_string.isin(experiments.reaction_string)]

      seen_indices.extend(to_add)
      clear_session()
         
   return ranks

all_ranks = []

from multiprocessing import Pool
SEEDS = np.random.SeedSequence(123456789).generate_state(args.repetitions)

def target(repId):
   try:
      rng = np.random.RandomState(SEEDS[repId])
      ranks = simulate(full_trainX, ground_truth, repId=repId, totally_random=args.totally_random, simulation_steps=STEPS,\
                    batch_schedule=SCHEDULE, cond_per_batch=COND, batch_size=args.batch_size, partially_random=args.partially_random, rng=rng, normalize=args.normalize)
   except:
      logger.exception(f'Encountered error in Rep #{repId}')
      raise
   return ranks

def worker(qin, qout):
   while True:
      repID = qin.get()
      if repID=='break':
         break
      else:
         qout.put(target(repID))

qin, qout = Queue(), Queue()
processes = []
for _ in range(22):
   p = Process(target=worker, args=(qin, qout))
   p.start()
   processes.append(p)

rep_batch = 25
all_ranks = []
for x in range(args.pivot, args.repetitions):
   qin.put(x)
for x in range(args.pivot, args.repetitions):
   all_ranks.append(qout.get())
#repetitions = list(range(args.repetitions))
#while repetitions!=[]:
#   [qin.put(x) for x in repetitions[:rep_batch]]
#   for i in range(rep_batch):
#      all_ranks += [qout.get()]
#   repetitions = repetitions[rep_batch:]
#   clear_session()
#
all_ranks = np.array(all_ranks).T
for p in processes:
   qin.put('break')
for p in processes:
   p.kill()
   p.join()

pd.DataFrame(all_ranks, columns = ['RepId:%i'%x for x in range(args.repetitions)]).to_csv(f'ranks_{now}.csv', sep=';', index=False)
