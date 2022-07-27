import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--radius', default=3, type=int)
parser.add_argument('--class_importances', type=str, default='')
parser.add_argument('-N', default=50, type=int)
parser.add_argument('--mark', action='store_true')
parser.add_argument('--boron', action='store_true')
parser.add_argument('--smiles_column', type=str, default='block')
#parser.add_argument('--within_class', action='store_true')
parser.add_argument('--sep', type=str, default=';')
parser.add_argument('--mtx', type=str, default='')
parser.add_argument('--out_core', type=str, default='')
parser.add_argument('filename', type=str)
args=parser.parse_args()
#args.radius=3
#args.th=0.4
#args.mark=True
app='marked' if args.mark else ''
output_name = '_'.join([args.out_core,'clusters','r%i'%args.radius, app]).strip('_')

from assign_class import *
from rdkit import Chem
from rdkit.Chem import AllChem
import gzip
import yaml
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
import logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
import pandas as pd
import numpy as np 
import sys

if args.boron:
    patt = Chem.MolFromSmarts('[$(cB)]')
else:    
    patt = Chem.MolFromSmarts('[$(c[Br,I])]')

def make_logger(logname=None, level=logging.INFO, logger = logging.getLogger()):
   formatter=logging.Formatter('%(asctime)s: %(levelname)s): %(message)s')
   handlers= [logging.StreamHandler(sys.stdout)]
   if logname!=None:
       handlers.append(logging.FileHandler(logname))
   for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)
   logger.setLevel(level)
   return logger

def mark_ring(mol):
   matches = sum(mol.GetSubstructMatches(patt), ())
   atom_rings = mol.GetRingInfo().AtomRings()
   for ring in atom_rings:
      if any([x in matches for x in ring]):
         for x in ring:
            mol.GetAtomWithIdx(x).SetIsotope(42)

def ClusterFps(fps, cutoff=0.4, dists=None, return_dists=False):
   # first generate the distance matrix:
   nfps = len(fps)
   if isinstance(dists, type(None)):
    dists = []
    for i in range(1,nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1-x for x in sims])
   else:
       if len(np.shape(dists))==2:
           assert dists.shape[0]==dists.shape[1]
           dists=mtx_to_lower(dists)
   # now cluster the data:
   cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
   if return_dists:
      return cs, dists
   else:
      return cs

def lower_to_mtx(vect, N, diag=0):
   result=np.identity(N)*diag
   beg=0
   for i in range(1,N):
      row = vect[beg:beg+i]
      result[i,:i]=row
      result[:i,i]=row
      beg+=i
   return result

def mtx_to_lower(mtx):
    dst=[]
    N=len(mtx)
    for i in range(1,N):
        dst.extend(mtx[i,:i])
    return dst

def av_nn(mtx, idx):
   tmp = mtx[idx,:][:,idx]
   N=len(tmp)
   tmp+=100*np.identity(N)
   tmp=tmp.min(axis=1)
   return tmp.mean()

def diversity(mtx,idx):
   tmp = mtx[idx,:][:,idx]
   N=len(tmp)
   return tmp.sum()/(N*N-N)
   
logger = make_logger(output_name+'.log')
logging.info('START')

config_str = yaml.dump(args.__dict__)
logging.info('Configuration:\n'+config_str)

#with open('mono_halides_unique.smiles', 'r') as f: smiles_list = f.readlines()
data = pd.read_csv(args.filename, sep=args.sep)
data['clean_smiles'] = data[args.smiles_column].str.replace('Xe','Br')
if args.boron:
    data['clean_smiles'] = data[args.smiles_column].str.replace('\[Be\]','B(OC(=O)C)(OC(=O)C)')
data['ring_class'] = data['clean_smiles'].apply(get_class, args=(args.boron,))

if args.class_importances!='':
    class_importances = pd.read_csv(args.class_importances, sep=';')
    class_importances = class_importances[class_importances['class_name'].isin(data['ring_class'])]
else:
    uniq_class = data['ring_class'].unique()
    class_importances = pd.DataFrame({'class_name':uniq_class, 'freq':np.ones(len(uniq_class))})

class_importances['expected_count'] = class_importances['freq']*args.N/class_importances['freq'].sum()
logging.info('class_importances prepared')

mark=args.mark
fps=[]
for x in data['clean_smiles']:
   mol = Chem.MolFromSmiles(x)
   if mark: mark_ring(mol)
   fps.append(AllChem.GetMorganFingerprint(mol, args.radius, useCounts=True))

logging.info('FGP done, nMol=%s'%len(fps))

output_name_back = output_name

if args.mtx=='':
    _, dst = ClusterFps(fps,cutoff=0.4, return_dists=True)
    dst = lower_to_mtx(dst, len(fps))
    with gzip.open(output_name+'_dst.npz' ,'wb') as f:
        np.save(f, dst)
    logging.info('DST saved')
        ##else:
            ##clusters = ClusterFps(fps, cutoff=th, dists=dst, return_dists=False)
else:
    with gzip.open(args.mtx, 'rb') as f:
        dst = np.load(f)
        logging.info('DST LOADED')

fps = np.array(fps)
all_indices = np.arange(len(fps))
output_name = output_name_back
clusters=[]

ths = {}
for class_name in class_importances['class_name']:
    idx = data['ring_class']==class_name
    fps_to_use = list(fps[idx])
    indices_used=all_indices[idx]
    dists_to_use = dst[idx,:][:,idx]
    prev_clusters = None
    expected = class_importances[class_importances['class_name']==class_name]['expected_count'].values[0]
    
    #chose th for each class
    for th in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        logging.info('CLASS: %s TH=%.4f'%(class_name, th))
    
        clusters_internal = ClusterFps(fps_to_use, cutoff=th, dists=dists_to_use, return_dists=False)
        N_non_singleton = len([x for x in clusters_internal if len(x)>1]) #!!!
        Nclust = len(clusters_internal)
        ths[class_name]=th
        
        if isinstance(prev_clusters, type(None)):
            prev_clusters = clusters_internal
        
        if Nclust<expected:
            break
        
        prev_clusters = clusters_internal
    
    for clust in prev_clusters:
        clust = [indices_used[idx] for idx in clust]
        clusters.append(clust)

    logging.info('clusters done, Ncluster=%i, N_non-singleton_clusters=%i'%(len(prev_clusters), len([x for x in clusters_internal if len(x)>1])))

logging.info('ALL clusters done, Ncluster=%i, N_non-singleton_clusters=%i'%(len(clusters), len([x for x in clusters if len(x)>1])))
    
is_centroid=np.zeros(len(fps))
cluster_ids=np.zeros(len(fps))

for clust_id, cluster in enumerate(clusters):
  for idx in cluster:
    centroid_flag = 1 if idx==cluster[0] else 0
    is_centroid[idx]=centroid_flag
    cluster_ids[idx]=clust_id

# collect all data
data['is_centroid'] = is_centroid
data['clusterID'] = cluster_ids
data['is_singleton'] = [(data['clusterID']==x).sum()==1 for x in data['clusterID']]
data['cluster_size'] = [(data['clusterID']==x).sum() for x in data['clusterID']]
data['th'] = [ths[x] for x in data['ring_class']]

classes_counts = data.groupby('ring_class').agg({'is_singleton':'sum','is_centroid':'sum'})
classes_counts['only_singletons'] = classes_counts['is_singleton']==classes_counts['is_centroid']

data['select'] = ((data['is_centroid']==1) & (~data['is_singleton'])) | data['ring_class'].apply(lambda x: classes_counts['only_singletons'][x])

logging.info('Data transformed')
data.to_csv(output_name+'.csv', sep=';', index=False)

idx_centroid = data.is_centroid==1
centroids = data[data.is_centroid==1]
centroids.to_csv(output_name+'_centroids.csv', sep=';', index=False)

data[data['select']].to_csv(output_name+'_selected.csv', sep=';', index=False)
logging.info('Selected: %i'%(data['select'].sum()))

idx_cheapest=data.groupby('clusterID')['lowest_price'].idxmin()
cheapest = data.loc[idx_cheapest]
cheapest.to_csv(output_name+'_cheapest.csv', sep=';', index=False)

idx_ring_cls=data[data['is_centroid']==1].groupby('ring_class')['cluster_size'].idxmax()
ring_cls_centroids=data.loc[idx_ring_cls]
logging.info('Ring class Diversity: %8.4f'%diversity(dst, np.where(idx_ring_cls.values)[0]))
logging.info('Ring class Av NN Dst: %8.4f'%av_nn(dst, np.where(idx_ring_cls.values)[0]))
ring_cls_centroids.to_csv(output_name+'_ring_class_centroids.csv', sep=';')

logging.info('Data written')
all_idx = np.arange(len(fps))
logging.info('Input Diversity: %8.4f'%diversity(dst, all_idx))
logging.info('Input Av NN Dst: %8.4f'%av_nn(dst,all_idx))
logging.info('Cheapest Diversity: %8.4f'%diversity(dst, idx_cheapest.values))
logging.info('Cheapest Av NN Dst: %8.4f'%av_nn(dst, idx_cheapest.values))
logging.info('Centroids Diversity: %8.4f'%diversity(dst, np.where(idx_centroid.values)[0]))
logging.info('Centroids Av NN Dst: %8.4f'%av_nn(dst, np.where(idx_centroid.values)[0]))

