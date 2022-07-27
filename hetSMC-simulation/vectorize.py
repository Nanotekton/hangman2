from rdkit import Chem
from rdkit.Chem import AllChem
from loading import load_spreadsheet
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import json

logger = logging.getLogger('vectorize_module')

def smiles_to_fgp(smiles=None, mol=None):
    try:
        if smiles==None and mol==None:
            return none
        elif mol==None:
            mol = Chem.MolFromSmiles(smiles)
    
        fgp = AllChem.GetMorganFingerprint(mol, 3, useFeatures=True, useCounts=True)
        
        return fgp
    except:
        print(smiles)
        raise


def vectorize_unique_substrates(source='shared_file'):
    if source=='shared_file':
        data, status = load_spreadsheet()
    elif type(source).__name__=='str':
        status = 'loaded_from_local_file'
        data = pd.read_csv(source, sep=';')
    else:
        status = 'assumed source to be DataFrame'
        data = source
    logger.info(status)

    mida_smiles = [x for x in data['boronate/boronic ester smiles'].unique() if type(x).__name__[:3]=='str']
    mida_fgps = [smiles_to_fgp(x) for x in mida_smiles]

    bromide_smiles = [x for x in data['bromide smiles'].unique() if type(x).__name__[:3]=='str']
    bromide_fgps = [smiles_to_fgp(x) for x in bromide_smiles]

    return (mida_smiles, mida_fgps), (bromide_smiles, bromide_fgps)


def make_reagent_encoder(source='shared_file', df=None, cols=['ligand', 'solvent', 'base', 'temperature']):
    '''
    Makes OneHotEncoder for reagents, inferring categories from different sources
    This may be a CSV file with proper columns, shared CSV or yaml with categories'''
    use_df = False
    if not (df is None):
        status = 'from_given_DataFrame'
        use_df = True
    elif source=='shared_file':
        df, status = load_spreadsheet()
        use_df = True
    elif '.csv' == source[-4:]:
        df = pd.read_csv(source, sep=';')
        status = 'from_local_CSV'
        use_df = True
    elif '.json' == source[-5:]:
        status = 'loaded_from_json'
        with open(source, 'r') as f:
            categories = json.load(f)
            if type(categories).__name__=='dict':
                category_names = list(categories.keys())
                logger.info('category names: %s'%str(category_names))
                assert set(category_names)==set(cols)
                category_names.sort(key=lambda x: cols.index(x))
                categories = [categories[x] for x in category_names]

    else:
        raise ValueError('unknown source of source format: %s'%source)

    if use_df:
        enc = OneHotEncoder(sparse=False)
        enc.fit(df[cols])
    else:
        enc = OneHotEncoder(sparse=False, categories=categories)
        Nmax = max(len(x) for x in categories)
        X = [[cat[min(i, len(cat)-1)] for cat in categories] for i in range(Nmax)]
        enc.fit(X)

    logger.info('encoder status: %s'%status)
    logger.info('columns: ' + str(cols))
    logger.info('categories: \n' + '\n'.join(str(x) for x in enc.categories_))

    return enc


def fgps_to_fixed_vec(fgp_list, size=2500):
    result = np.zeros((len(fgp_list), size))
    collisions = 0
    lens, collisions_per_record = [], []
    for i,f in enumerate(fgp_list):
        elements = f.GetNonzeroElements()
        lens.append(len(elements))
        this_collisions = 0
        for x in elements:
            j = x%size
            v = elements[x]
            if result[i,j]!=0:
                this_collisions += 1
            result[i,j] += v
        collisions += this_collisions
        collisions_per_record.append(this_collisions)
    logger.info('total %i collisions encountered, result shape= %i, %i'%(collisions, result.shape[0], result.shape[1]))
    averages = tuple(sum([[np.mean(x), np.std(x)] for x in [collisions_per_record, lens]],[]))
    rate = 100*averages[0]/averages[2] if averages[2]!=0 else -1
    averages += (rate,)
    logger.info('average per record: collisions= %.1f (%.1f), len= %.1f (%.1f), rate=%.1f %%'%averages)
    return result
    

def count_frequencies(fgp_list):
    all_ids={}
    
    for f in fgp_list:
        bits = f.GetNonzeroElements()
        for x in bits:
            all_ids[x] = all_ids.get(x,0) + bits[x]
    
    all_ids_keys = list(all_ids.keys())
    all_ids_keys.sort(key=lambda x:all_ids[x])

    return all_ids_keys, all_ids


def fgps_to_freq_vec(fgp_list, cutoff=0, set_place_for_unk=False, ranking=None):
    '''Sorts keys according to their frequency (ommitting keys less frequent than cutoff)'''
    
    if ranking is None:
        ranking, counts = count_frequencies(fgp_list)
        ranking = [x for x in ranking if counts[x]>=cutoff]

    dim = len(ranking) + int(set_place_for_unk)
    result = np.zeros((len(fgp_list), dim))
    for i, f in enumerate(fgp_list):
        elements = f.GetNonzeroElements()
        for x, v in elements.items():
            idx = ranking.index(x) if x in ranking else dim-1
            result[i, idx] += v

    return result, ranking


def unfold_conditions(condition_space_list):
    result = []
    for dim in condition_space_list:
        if result == []:
            result = [[x] for x in dim]
        else:
            new_result = []
            for x in dim:
                for y in result:
                    new_result.append(y+[x])
            result = new_result
    return result


def make_full_space_df(substrates_csv, reagent_span, substrates_cols=['boronate/boronic ester smiles', 'bromide smiles', 'product_smiles']):
    if type(reagent_span).__name__=='str':
        with open(reagent_span, 'r') as f:
            reagent_span = json.load(f)
    if type(substrates_csv).__name__=='str':
        substrate_csv = pd.read_csv(substrate_csv, sep=';')
    
    data = {}
    conditions_names = list(reagent_span.keys())
    conditions_space = unfold_conditions([reagent_span[x] for x in conditions_names])
    logger.info('reagent span:\n' + '\n'.join('%s:%s'%(x,str(y)) for x,y in reagent_span.items()))
    logger.info('reagent space size: %i'%len(conditions_space))

    for _, row in substrates_csv.iterrows():
        for cond in conditions_space:
            for col in substrates_cols:
                data[col] = data.get(col, []) + [row[col]]
            for i, name in enumerate(conditions_names):
                data[name] = data.get(name, []) + [cond[i]]
    
    data = pd.DataFrame(data)
    return data


def prepare_substrate_vect_dict():
    mida, bromide = vectorize_unique_substrates()
    logger.info('vectorization done')

    mida_ranking, mida_counts = count_frequencies(mida[1])
    bromide_ranking, bromide_counts = count_frequencies(bromide[1])
    logger.info('counts done')
    
    logger.info('len mida= %i, len bromide= %i'%(len(mida_ranking), len(bromide_ranking)))

    print('Mida top-10', [mida_counts[x] for x in mida_ranking[-10:]])
    print('Mida bottom-10', [mida_counts[x] for x in mida_ranking[:10]])
    print('Bromide top-10', [bromide_counts[x] for x in bromide_ranking[-10:]])
    print('Bromide bottom-10', [bromide_counts[x] for x in bromide_ranking[:10]])

    logger.info('vectorizing mida')
    mida_vec = fgps_to_freq_vec(mida[1])[0]
    #const_cols = np.where(mida_vec.std(axis=0)==0)[0]
    #logger.info('constant columns: %i/%i'%(len(const_cols), mida_vec.shape[1]))
    #logger.info('actual vars: %i'%(-len(const_cols) + mida_vec.shape[1]))

    logger.info('vectorizing bromide')
    bromide_vec = fgps_to_freq_vec(bromide[1])[0]

    #bconst_cols = np.where(bromide_vec.std(axis=0)==0)[0]
    #logger.info('constant columns: %i/%i'%(len(bconst_cols), bromide_vec.shape[1]))
    #logger.info('actual vars: %i'%(-len(bconst_cols) + bromide_vec.shape[1]))
    
    bromide_dict = dict(zip(bromide[0], bromide_vec))#[:, bconst_cols]))
    mida_dict = dict(zip(mida[0], mida_vec))#[:, const_cols]))
    
    return mida_dict, bromide_dict


def dataframe_to_encoded_array(df, substrate_encoders, conditions_encoder, conditions_fields):
    vectors = []
    for field_name, encoder_dict in substrate_encoders:
        vecs = [encoder_dict[m] for m in df[field_name].values]
        vectors.append(vecs)
    conditions = df[conditions_fields].values
    vectors.append(conditions_encoder.transform(conditions))
    vectors = np.hstack(vectors)
    return vectors


if __name__=='__main__':
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter('%(name)s:%(asctime)s- %(message)s'))
    logger.addHandler(h)

    logger.info('start')
    mida, bromide = vectorize_unique_substrates()
    logger.info('vectorization done')

    mida_ranking, mida_counts = count_frequencies(mida[1])
    bromide_ranking, bromide_counts = count_frequencies(bromide[1])
    logger.info('counts done')
    
    logger.info('len mida= %i, len bromide= %i'%(len(mida_ranking), len(bromide_ranking)))

    print('Mida top-10', [mida_counts[x] for x in mida_ranking[-10:]])
    print('Mida bottom-10', [mida_counts[x] for x in mida_ranking[:10]])
    print('Bromide top-10', [bromide_counts[x] for x in bromide_ranking[-10:]])
    print('Bromide bottom-10', [bromide_counts[x] for x in bromide_ranking[:10]])

    logger.info('vectorizing mida')
    mida_vec = fgps_to_freq_vec(mida[1])[0]
    const_cols = np.where(mida_vec.std(axis=0)==0)[0]
    logger.info('constant columns: %i/%i'%(len(const_cols), mida_vec.shape[1]))
    logger.info('actual vars: %i'%(-len(const_cols) + mida_vec.shape[1]))

    logger.info('vectorizing bromide')
    bromide_vec = fgps_to_freq_vec(bromide[1])[0]
    const_cols = np.where(bromide_vec.std(axis=0)==0)[0]
    logger.info('constant columns: %i/%i'%(len(const_cols), bromide_vec.shape[1]))
    logger.info('actual vars: %i'%(-len(const_cols) + bromide_vec.shape[1]))

    logger.info('Conditions:')
    enc = make_reagent_encoder()
    df, _ = load_spreadsheet()
    logger.info('data: %i records'%df.shape[0])
    columns = ['ligand', 'solvent', 'base', 'temperature']
    x = enc.transform(df[columns])
    print(df[columns][:5])
    print(x[:5])

    df2 = df[['boronate/boronic ester smiles', 'bromide smiles', 'product_smiles']].drop_duplicates()
    logger.info('reagent_space: %i'%df2.shape[0])
    df2 = make_full_space_df(df2, 'space_dict.json')
    logger.info('full space: %i'%df2.shape[0])
    total_dim = 1
    for col in df2.columns:
        un = df2[col].unique()
        print(col, len(un), un)
        if 'product' in col or not ('smiles' in col):
            total_dim *= len(un)
    logger.info('total dim: %i, %s'%(total_dim, total_dim==df2.shape[0]))

