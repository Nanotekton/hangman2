#12 {'', 'dtbpf', 'Xantphos', 'XPhos', 'P(o-Tol)3', 'SPhos', 'P(Cy)3', 'CataCXium A', 'dppf', 'P(Ph)3', 'P(tBu)3', 'AmPhos'}
#7 {'', 'NaOH', 'NaHCO3', 'KOH', 'K3PO4', 'Et3N', 'CsF'}
#4 {'DMF', 'MeCN', 'THF', 'MeOH'}
import sys

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

def data():
    yields = dict()
    for line in open('cond.hyp').readlines()[3:]:
        yld, lig,base, solv = line.split('\t')
        cond=(lig,base,solv.strip())
        yields[cond]=float(yld)
    return yields

def training():
    import logging
    logging.basicConfig(filename='hyperOpt_Yields.log',level=logging.INFO)
    lig = {{choice(['', 'dtbpf', 'Xantphos', 'XPhos', 'P(o-Tol)3', 'SPhos', 'P(Cy)3', 'CataCXium A', 'dppf', 'P(Ph)3', 'P(tBu)3', 'AmPhos'])}}
    base = {{choice(['', 'NaOH', 'NaHCO3', 'KOH', 'K3PO4', 'Et3N', 'CsF'])}}
    solv = {{choice(['DMF', 'MeCN', 'THF', 'MeOH'])}}
    cond = (lig,base,solv)
    try:
        negYld = -yields[cond]
    except:
        #print("K", yields.keys() )
        print("cond", cond, "not found")
        m1 = [ k for k in yields.keys() if k[0] == cond[0] ]
        print(m1)
        raise
        negYld=0
    logging.info("MAE: "+str( negYld).strip()+ " SPACE:"+ str(space) )
    print("MAE: "+str( -negYld).strip(), file=sys.stderr) 
    return {'loss': negYld, 'status': STATUS_OK, 'model': None}




if __name__ == "__main__":
    best_run, best_model = optim.minimize(model=training, data=data, algo=tpe.suggest, max_evals=550, trials=Trials())
    print("============================================", file=sys.stderr)