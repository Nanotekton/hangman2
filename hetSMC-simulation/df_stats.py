import numpy as np
import scipy
import pandas as pd
import logging

def calc_probability_of_improvement(df, epsilon=0.01, group='condition_string'):
    df['av_group_Y'] = df.groupby(group)['Ypred'].transform('mean')
    df['Yvar'] = df['Yunc'].apply(lambda x: x*x)
    df['av_group_unc'] = df.groupby(group)['Yvar'].transform('sum').apply(lambda x: np.sqrt(x))
    df['group_size'] = df.groupby(group)['Yvar'].transform('count')
    df['av_group_unc'] = df['av_group_unc']/df['group_size']
    Ymax = df['av_group_Y'].max()
    notSeenData = df[ df.training ==False]
    Z = (notSeenData['av_group_Y'] - Ymax - epsilon) / notSeenData['av_group_unc']
    #Z = (fulldata['YpredMean'] - Ymax - epsilon)/fulldata['YpredStd']
    PI = scipy.stats.norm.cdf(Z)
    df['PI']=0
    df.loc[df.training ==False, 'PI'] = PI
    #return proposeNextPoint(fulldata, allsbs), fulldata #modifies df

HEADER = 'time_stamp\tbest_conditions_solvent\tbest_conditions_temperature\tbest_conditions_base\tbest_conditions_ligand\tconditions_average_yield_predicted\tconditions_average_uncertainty\tconditions_coverage\ttotal_average_uncertainty\ttraining_set_average_uncertainty\tunseen_set_average_uncertainty\tspace_coverage\treactions_in_training_set\ttopK\tcomment'

def get_top_from_full_space(space_df, N, K=3, now='', groupby='condition_string', pred_cols=('Ypred','Yunc'), group_sep=',,', newline=False):
    pred, unc = pred_cols
    groupped = space_df.groupby(groupby).agg({pred:'mean', unc:'mean'})
    conditions = groupped.index
    order = np.argsort(groupped[pred])

    lines = []
    av_unc = space_df.Yunc.mean()
    train_av_unc = space_df[space_df.training].Yunc.mean()
    test_av_unc = space_df[~space_df.training].Yunc.mean()
    coverage = space_df['training'].mean()
    for topK in range(1,K+1):
        idx = order[-topK]
        max_av = groupped[pred].values[idx]
        max_unc = groupped[unc].values[idx]
        max_cond = conditions[idx]
        max_cond_coverage = space_df[space_df[groupby]==max_cond]['training'].mean()
        new_line = f'{now}|{max_cond.replace(group_sep,"|")}|{max_av}|{max_unc}|{max_cond_coverage}|{av_unc}|{train_av_unc}|{test_av_unc}|{coverage}|{N}|{topK}|'
        new_line = new_line.replace('|','\t') 
        if newline: new_line+='\n'
        lines.append(new_line)

    return lines

def propose_batch(space_df, Ncond=18, Nbatch=72):
    not_seen = space_df[~space_df.training].reset_index()
    not_seen.sort_values(['PI', 'Yunc'], inplace=True, ascending=False)
    
    new_conditions = list(not_seen.condition_string.unique())
    N_new_cond = len(new_conditions)
    N_new = not_seen.shape[0]
    
    if N_new<=Nbatch:
        logging.info('very last iteration')
        experiments = not_seen
    else:
        idx_to_take = []
        idx_pool = list(not_seen.index)
        while(len(idx_to_take)<Nbatch):
            to_remove = []
            for cond in new_conditions[:Ncond]:
                view = not_seen.loc[idx_pool]
                if cond in view.condition_string.values:
                    idx = view[view.condition_string==cond].index[0]
                    idx_to_take.append(idx)
                    idx_pool.remove(idx)
                else:
                    to_remove.append(cond)
            for cond in to_remove:
                new_conditions.remove(cond)
    
        experiments = not_seen.loc[idx_to_take]
    
    #del experiments['condition_string']
    #del experiments['Yvar']
    #del experiments['training']
    
    experiments.sort_values(['PI','solvent','temperature','base','ligand','Yunc'], inplace=True, ascending=False)

    return experiments

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--now', type=str, default='')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--batch_p', type=int, nargs=2, default=(18,72))
    parser.add_argument('dump', type=str, default=None)
    args = parser.parse_args()
    
    space_df = pd.read_csv(args.dump, sep='\t')
    if args.batch:
        ncond, nbatch = args.batch_p
        exp = propose_batch(space_df, Ncond=ncond, Nbatch=nbatch)
        exp.to_csv('prediction_%s.csv'%args.now, sep='\t')
    else:
        N = int(space_df['training'].sum())*2
        lines = get_top_from_full_space(space_df, N, now=args.now)
        print(HEADER)
        for line in lines:print(line)
