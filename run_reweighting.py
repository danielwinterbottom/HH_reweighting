import imp
import math

rw=imp.load_module('allmatrix2py', *imp.find_module('allmatrix2py'))
rw.get_pdg_order()
rw.initialise('MadLoop5_resources/param_card.dat') 
rw.initialise('MadLoop5_resources/param_card_1.dat') 
rw.set_madloop_path('MadLoop5_resources')

def invert_momenta(p):
    """ fortran/C-python do not order table in the same order"""
    return [[p[j][i] for j in range(len(p))] for i in range(len(p[0]))]

pdgs = [21, 21, 25, 25]

def zboost(part, pboost=[]):
        """Both momenta should be in the same frame.
           The boost perform correspond to the boost required to set pboost at
           rest (only z boost applied).
        """
        E = pboost[0]
        pz = pboost[3]

        #beta = pz/E
        gamma = E / math.sqrt(E**2-pz**2)
        gammabeta = pz  / math.sqrt(E**2-pz**2)

        out =  [gamma * part[0] - gammabeta * part[3],
                            part[1],
                            part[2],
                            gamma * part[3] - gammabeta * part[0]]

        if abs(out[3]) < 1e-6 * out[0]:
            out[3] = 0
        return out


parts = [
  [1.4861514428e+03, 0.0000000000e+00, 0.0000000000e+00, 1.4861514428e+03],
  [4.1482521106e+01, 0.0000000000e+00, 0.0000000000e+00, -4.1482521106e+01],
  [4.2120989418e+02, 1.3960712742e+02, -1.1253971723e+02, 3.6005199204e+02],
  [1.1064240698e+03, -1.3960712742e+02, 1.1253971723e+02, 1.0846169297e+03]
]

## in gg rest frame
##[[248.29278803997386, 0.0, 0.0, 248.29278803997386], 
##[248.29278803997408, 0.0, 0.0, -248.29278803997408], 
##[248.2927880461964, 139.60712742, -112.53971723, -117.76659378872637], 
##[248.2927881275723, -139.60712742, 112.53971723, 117.7665937149527]]

print parts

#pboost = [parts[0][i] + parts[1][i] for i in range(4)] 
pboost = [parts[2][i] + parts[3][i] for i in range(4)] 

print pboost

parts_2 = []
for part in parts:
    parts_2.append(zboost(part, pboost))

print parts_2

parts = parts_2

# to be checked but this will be the idea:
# if information for 2 incoming gluons is available and hh pair have no pT then input this information (i.e we have LO)
# if no information for 2 gluon provided OR if information for 2 incoming gluons is available but hh pair have pT>0 then this is NLO(-like) in this case we boost to hh rest frame and then we set the incoming gluon's E equal to higgs E in rest frame and pz will equal +E and -E 

parts = invert_momenta(parts)

alphas = 1.01467400e-01
#alphas = 0.137
scale2 = 0.
nhel = -1 # can get this from LHE file, -1 will sum over all helicities 

val = rw.smatrixhel(pdgs,parts,alphas,scale2,nhel)
print val 
