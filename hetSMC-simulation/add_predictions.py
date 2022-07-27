import pandas as pd
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--inp', type=str)
parser.add_argument('--out', type=str)
args = parser.parse_args()

rounds = ['2021-May-17-12.11.45', '2021-May-25-10.29.18', '2021-Jun-07-03.19.12', '2021-Jun-21-00.04.56']

data = pd.read_csv(args.inp, sep='\t')

cols = ['product_smiles', 'solvent', 'base', 'ligand', 'temperature']
ypred, yunc = [], []
for _, row in data.iterrows():
    if row.round_id==1:
        ypred.append(None)
        yunc.append(None)
        continue
    stamp = rounds[int(row.round_id)-2]
    full = pd.read_csv(f'prediction_full_space_{stamp}.csv', sep='\t')
    mask = full[cols[0]]==row[cols[0]]
    for x in cols[1:]:
        mask = mask & (full[x]==row[x])
    pred = full[mask]
    try:
        ypred.append(np.clip(pred.Ypred.values[0], 0, 1))
        yunc.append(pred.Yunc.values[0])
    except:
        print(row)
        for x in cols:
            print(x, sum(full[x]==row[x]), full[x].unique())
        raise

data['predicted_yield_prior_proposal'] = ypred
data['prediction_uncertainty'] = yunc
data.to_csv(args.out, sep='\t', index=False)
