import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='extracted_ranks.csv')
parser.add_argument('logfile', type=str)
args = parser.parse_args()

import pandas as pd

#Repetition ID 24 iteration 0 - rank of the best: 10
result_dc = {}

#1. extract
max_iter = 0
with open(args.logfile, 'r') as f:
   for line in f:
      if 'rank of the best:' in line:
         line = line.split()
         rank = int(line[-1])
         iteration = int(line[-7])
         repID = 'RepId:' + line[-9]
         if repID not in result_dc:
            result_dc[repID] = {}
         result_dc[repID][iteration] = rank
         if iteration>max_iter:
            max_iter=iteration

#2. fill blanks
for repID in result_dc:
   ranks = []
   for i in range(max_iter+1):
      ranks.append(result_dc[repID].get(i, None))
   result_dc[repID] = ranks

#3. save
df = pd.DataFrame(result_dc)
df.to_csv(args.output, sep=';', index=False)
