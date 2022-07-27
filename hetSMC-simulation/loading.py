import numpy as np
import pandas as pd
import datetime
from os import popen
from os.path import isfile
import json
import logging
from rdkit.Chem import CanonSmiles

logger = logging.getLogger('loader_module')

#==========================
cache_name = 'cache.json'
if isfile(cache_name):
    with open(cache_name,'r') as f:
        cache = json.load(f)
else:
    cache = dict(last_dump_time='', last_dump_hash='', last_prediction='')

shared_url = cache['url']
#==========================

def common_correctors(df):
    df.replace('Sphos','SPhos', inplace=True)

def detect_smiles(sml):
    try:
        x = CanonSmiles(sml)
        return True
    except:
        return False

def get_hash(name):
    if isfile(name):
        result = popen('sha512sum %s'%name).read().split()[0]
    else:
        result = ''
    return result

def load_spreadsheet(url=shared_url, cache=cache, cache_name=cache_name):
    #shared spreadsheet
    data = pd.read_csv(shared_url)
    data.dropna(axis=0, how='all', inplace=True)
    data.dropna(axis=1, how='all', inplace=True)
    data = data[data['comment']!='wb_example'].reset_index()
    common_correctors(data)
    masks = []
    substrates_cols = ['boronate/boronic ester smiles', 'bromide smiles', 'product_smiles']
    for x in substrates_cols:
        masks.append(data[x].apply(detect_smiles))
    mask = masks[0] & masks[1] & masks[2]
    data['smiles_ok'] = mask
    
    now = datetime.datetime.now().strftime('%Y-%b-%d-%H.%M.%S')
    #1.a. remove incorrect smiles
    N = data.shape[0]
    bad = data[~data.smiles_ok]
    data = data[data.smiles_ok].reset_index()
    logger.info('dropped records: %i/%i'%(bad.shape[0], N))
    bad.to_csv('omitted_%s.csv'%now, sep='	', index=False)
    
    data.to_csv('dump_%s.csv'%now, sep='	', index=False)
    current_hash = get_hash('dump_%s.csv'%now)
    previous_hash =  cache['last_dump_hash']

    if previous_hash=='':
        status = 'initial_download'
    elif previous_hash==current_hash:
        status = 'unchanged'
        popen('rm dump_%s.csv'%now).read()
    else:
        status = 'changed'
        logger.info(('detected change between %s and %s'%(cache['last_dump_time'], now)).replace('.',':'))

    if status in ['changed','initial_download']:
        cache['last_dump_hash'] = current_hash
        cache['last_dump_time'] = now
        with open(cache_name, 'w') as f: json.dump(cache, f)
    
    return data, status

if __name__=='__main__':
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter('%(name)s:%(asctime)s- %(message)s'))
    logger.addHandler(h)
    data, status = load_spreadsheet()
    print(status, data.shape)
