import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true')
parser.add_argument('--dump', type=str, default=None)
parser.add_argument('--update_gdrive', action='store_true')
parser.add_argument('--update_stats', action='store_true')
parser.add_argument('--num_models', type=int, default=100)
args = parser.parse_args()

import numpy as np
import pandas as pd
from loading import cache, cache_name, load_spreadsheet
from df_stats import calc_probability_of_improvement, get_top_from_full_space, HEADER, propose_batch
from vectorize import make_full_space_df, prepare_substrate_vect_dict, make_reagent_encoder, dataframe_to_encoded_array
import logging
from test_gpe import GPensemble, plainGP
import scipy
import datetime
import json
from os import popen
from os.path import isfile
import pickle
now = datetime.datetime.now().strftime('%Y-%b-%d-%H.%M.%S')


#logging.basicConfig(level=logging.INFO, format='%(name)s:%(asctime)s-%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(filename='prediction_%s.log'%now)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(name)s:%(asctime)s-%(message)s')
for H in [fh, sh]:
        H.setLevel(logging.INFO)
        H.setFormatter(formatter)
        logger.addHandler(H)

#1. load the shared data
if args.dump==None:
    df, status = load_spreadsheet()
    df = df[~df['yield'].isna()]
else:
    df = pd.read_csv(args.dump, sep='\t')
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    status = 'loaded from %s'%args.dump
N = df.shape[0]
logger.info('input params: %s'%json.dumps(args.__dict__))
logger.info('shared data status: %s'%status)
logger.info('shared data records: %i'%df.shape[0])


#2. prepare space; reagents get from the shared file, conditions - current_space_dict.json
substrates_cols = ['boronate/boronic ester smiles', 'bromide smiles', 'product_smiles']
mida_col, bromide_col = substrates_cols[:2]
conditions_cols = ['solvent', 'temperature', 'base', 'ligand']
substrate_space_df = df[substrates_cols].drop_duplicates()
space_df = make_full_space_df(substrate_space_df, 'current_space_dict.json', substrates_cols=substrates_cols)

#3. prepare substrate and conditions vectorizer
mida_dict, bromide_dict = prepare_substrate_vect_dict()
substrate_encoder_dicts = [(mida_col, mida_dict), (bromide_col, bromide_dict)]
reagent_encoder = make_reagent_encoder(source='space_dict.json', cols=conditions_cols)

#with open('cache_wtf.pkl','wb') as f:
#    pickle.dump((substrate_encoder_dicts, reagent_encoder), f)

#4. mark in space_df what is done
all_cols = substrates_cols + conditions_cols
reaction_cols = conditions_cols + ['product_smiles']
df['condition_string'] = df[conditions_cols].agg(lambda x: ',,'.join(str(y) for y in x), axis=1)
space_df['condition_string'] = space_df[conditions_cols].agg(lambda x: ',,'.join(str(y) for y in x), axis=1)

df_reaction_string = df[reaction_cols].agg(lambda x: ',,'.join(str(y) for y in x), axis=1)
space_df_reaction_string = space_df[reaction_cols].agg(lambda x: ',,'.join(str(y) for y in x), axis=1)

mask = space_df_reaction_string.isin(df_reaction_string)

space_df['training'] = mask

#5. Make training vector_matrices
trainX = dataframe_to_encoded_array(df, substrate_encoder_dicts, reagent_encoder, conditions_cols)
idx = np.where(trainX.std(axis=0)>0)[0]
trainX = trainX[:,idx]
#np.save('trainX.npy', trainX)
Y = df['yield'].values.reshape(-1,1)
#np.save('trainY.npy', Y)
#exit(0)
uY, sY = Y.mean(), Y.std()
Y = (Y-uY)/sY

#6. Make space vector_matrix
spaceX = dataframe_to_encoded_array(space_df, substrate_encoder_dicts, reagent_encoder, conditions_cols)
spaceX = spaceX[:,idx]
logger.info('matrices done')

#exit(0)

#7. Train main model
#TODO: add caching of weights 
if cache['last_prediction']!='' and status=='unchanged' and not args.force:
    logger.info('no sense to repeat, aborting')
    exit(0)
else:
    logger.info('cached info: \n' + '\n'.join('%s:%s'%(x,y) for x,y in cache.items()))

GPE = GPensemble(trainX, Y, numModels=args.num_models) #smaller ensemble for tests
logger.info('trained')

#8. Predict
pred = GPE.predict(spaceX)
logger.info('predicted')

space_df['Ypred'] = pred['Ymean']*sY + uY
space_df['Yunc'] = pred['Ystd']*sY

#8a. Control
train_pred = GPE.predict(trainX)['Ymean'].reshape(-1)*sY + uY
Ytrue = Y.reshape(-1)*sY+uY
logger.info('train mae: %.3f'%(abs(Ytrue-train_pred).mean()))

df['reaction_string'] = df[reaction_cols].agg(lambda x: ',,'.join(str(y) for y in x), axis=1)
space_df['reaction_string'] = space_df[reaction_cols].agg(lambda x: ',,'.join(str(y) for y in x), axis=1)

df.drop_duplicates('reaction_string', inplace=True)
df.set_index('reaction_string', inplace=True)
space_df.set_index('reaction_string', inplace=True)

space_df['Yexp'] = df['yield']
space_df.reset_index(inplace=True)
space_df.drop('reaction_string', inplace=True, axis=1)

check = space_df[space_df.training]
mae_train = abs(check['Yexp'] - check['Ypred']).mean()
logging.info('mae train(space df): %.3f'%mae_train)
#


#9. Sample. Strategy: 18 conds, 4 subs
calc_probability_of_improvement(space_df)
experiments = propose_batch(space_df)
experiments.to_csv('prediction_%s.csv'%now, sep='\t', index=False)
space_df.to_csv('prediction_full_space_%s.csv'%now, sep='\t', index=False)
logger.info('written')

#update statistics

groupped = space_df.groupby('condition_string').agg({'Ypred':'mean'})
max_ = groupped.Ypred.max()
argmax_ = groupped.Ypred.argmax()
max_cond = groupped.index[argmax_]
av_unc = space_df.Yunc.mean()
max_av_unc = space_df[space_df.condition_string==max_cond]['Yunc'].mean()
max_coverage = space_df[space_df.condition_string==max_cond]['training'].mean()
coverage = space_df['training'].mean()

progress_file = 'progress.csv'
if isfile(progress_file):
    with open(progress_file, 'r') as f:
        progress_lines=f.readlines()
else:
    progress_lines = [HEADER +'\n']

progress_lines.extend(get_top_from_full_space(space_df, N, newline=True, now=now))

if args.update_stats:
    with open(progress_file, 'w') as f:
        for x in progress_lines:
            f.write(x)

#=====================
cache['last_prediction'] = 'prediction_%s.csv'%now
with open(cache_name, 'w') as f: json.dump(cache, f)

if args.update_gdrive:
    print(popen('rclone copy prediction_%s.log remote:MADNESS/prediction_%s_files'%(now, now)).read())
    print(popen('rclone copy dump_%s.csv remote:MADNESS/prediction_%s_files'%(cache['last_dump_time'], now)).read())
    print(popen('rclone copy omitted_%s.csv remote:MADNESS/prediction_%s_files'%(cache['last_dump_time'], now)).read())
    print(popen('rclone copy prediction_full_space_%s.csv remote:MADNESS/prediction_%s_files'%(now, now)).read())
    print(popen('rclone copy prediction_%s.csv remote:MADNESS'%(now)).read())
    print(popen('rclone copy %s remote:MADNESS'%progress_file).read())

