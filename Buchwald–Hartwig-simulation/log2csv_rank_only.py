import numpy as np 
import pickle
import gzip
import pandas as pd

def condition_entry_to_data(entry_line): # repetition, iteration, rank_of_actual, best_yield, unc_stats #best_unc, max_unc, av_unc, std_unc
   head, rest = entry_line.split('::::')
   repetition, iteration = [int(x) for x in head.split(':')[-1].split()[1:]]
   data = eval(rest.replace('array', ''))
   ranks = [d['rankTrue'] for d in data]
   best_yield = data[0]['Ypred']
   rank_of_actual = ranks.index(0)
   return repetition, iteration, rank_of_actual, best_yield


def update_dict_of_lists(dc, key, value):
   dc[key] = dc.get(key, []) + [value]


def process_lines(line_iterator):
   performance_metrics = {'rank':{}, 'best_yield':{}}

   for line in line_iterator:
      if 'NEXTCOND' in line:
         repID, iterID, rank_of_actual, best_yield = condition_entry_to_data(line)
         update_dict_of_lists(performance_metrics['rank'], iterID, rank_of_actual)
         update_dict_of_lists(performance_metrics['best_yield'], iterID, best_yield)

   return performance_metrics


modes = {'av':np.mean, 'std':np.std, 'med':np.median, 'interq':lambda x: np.quantile(x, 0.75) - np.quantile(x, 0.25), 'len':len, 'q1':lambda x:np.quantile(x, 0.25), 'q3':lambda x:np.quantile(x, 0.75)}


def make_basic_stats(dict_with_data):
   '''Dict has to be nested three times'''
   series = list(dict_with_data.keys())
   result = {}
   iterIDs = set()
   for key in series:
      iterIDs.update(dict_with_data[key].keys())
      for mode in modes:
         new_key = '%s_%s'%(key, mode)
         result[new_key] = {}
         for iterID in dict_with_data[key]:
            result[new_key][iterID] = modes[mode](dict_with_data[key][iterID])
    
   #flatten
   iterIDs = list(iterIDs)
   iterIDs.sort()
   for new_series in result:
      result[new_series] = [result[new_series].get(iID,None) for iID in iterIDs]
   result['iterID']=iterIDs
     
   return result


if __name__=='__main__':
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--output_core', type=str, default='results')
   parser.add_argument('logname', type=str)
   args = parser.parse_args()

   with open(args.logname, 'r') as f:
      performance = process_lines(f)
   print('data read')

   with gzip.open('%s_raw_data.pkz'%args.output_core, 'wb') as f:
      pickle.dump(performance, f)
   print('raw saved')

   performance = pd.DataFrame(make_basic_stats(performance))
   performance.to_csv('%s_stats.csv'%args.output_core, sep=';', index=False)

   


