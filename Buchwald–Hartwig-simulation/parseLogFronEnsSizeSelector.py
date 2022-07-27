#INFO:root:FC ['DBU;Xphos', 'MTBD;Xphos',
#INFO:root:full: EI 1350 P2Et;Xphos PI: 1347 BEMP;Xphos
#INFO:root:PI 1 [627,
import sys
fullList = []
best = dict()
results = dict()
for line in open(sys.argv[1]):
    if line.startswith('INFO:root:FC '):
        fullList = eval(line[12:])
    elif line.startswith('INFO:root:full: '):
            line = line.split()
            best = {'EI':(int(line[2]), line[3]) , 'PI':(int(line[5]), line[6]) }
    elif line.startswith('INFO:root:PI'):
        line = line.split()
        enslen = int(line[1])
        ensres = eval( ' '.join(line[2:]) )
        ensresCond = [ fullList[x] for x in ensres]
        pi = ensres.count( best['PI'][0])/len(ensres) 
        piCond = ensresCond.count( best['PI'][1])/len(ensres) 
        #print("PI", best['PI'][1], ensresCond)
        if not enslen in results:
            results[enslen] = {'PI':-1, 'PIcond':-1, 'EI':-1, 'EIcond':-1}
        results[enslen]['PI'] = pi
        results[enslen]['PIcond'] = piCond
    elif line.startswith('INFO:root:EI'):
        line = line.split()
        enslen = int(line[1])
        ensres = eval( ' '.join(line[2:]) )
        ensresCond = [ fullList[x] for x in ensres]
        ei =  ensres.count( best['EI'][0])/len(ensres) 
        eiCond =  ensresCond.count( best['EI'][1])/len(ensres) 
        if not enslen in results:
            results[enslen] = {'PI':-1, 'PIcond':-1, 'EI':-1, 'EIcond':-1}
        results[enslen]['EI'] = ei
        results[enslen]['EIcond'] = eiCond

for i in sorted(results):
    print(i, results[i]['EI'], results[i]['EIcond'], results[i]['PI'], results[i]['PIcond'])
