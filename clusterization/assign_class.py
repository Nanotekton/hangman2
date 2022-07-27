from rdkit import Chem
import pandas as pd

class_patterns = pd.read_csv('roszak_classes.csv', sep=';')
class_patterns['class_smarts_boron'] = class_patterns['class_smarts'].str.replace('Br','B').str.replace('I','B').apply(lambda x: Chem.MolFromSmarts(x))
class_patterns['class_smarts'] = class_patterns['class_smarts'].apply(lambda x:Chem.MolFromSmarts(x))

def get_class(x, boron=False):
   m = Chem.MolFromSmiles(x)
   if boron:
       iterable = class_patterns.class_smarts_boron
   else:
       iterable = class_patterns.class_smarts
   for i,p in enumerate(iterable):
      if m.HasSubstructMatch(p): return class_patterns.class_name.values[i]
   return 'none'

if __name__=='__main__':
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--inp', type=str)
   parser.add_argument('--out', type=str)
   parser.add_argument('--smiles_column', type=str, default='smiles')
   args = parser.parse_args()
   
   
   inp = pd.read_csv(args.inp, sep=';')
   inp = inp.assign(halogen_class=inp[args.smiles_column].apply(get_class))
   
   inp.to_csv(args.out, sep=';', index=False)
