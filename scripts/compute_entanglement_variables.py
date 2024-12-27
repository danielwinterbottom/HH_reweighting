import argparse
import ROOT
import numpy as np
from entanglement_funcs import EntanglementVariables
from array import array

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Input ROOT file.")
parser.add_argument('--n_events', '-n', help='Maximum number of events to process', default=-1, type=int)
parser.add_argument('--n_skip', '-s', help='skip n_events*n_skip', default=0, type=int)

args = parser.parse_args()

f = ROOT.TFile(args.input)

tree = f.Get('new_tree')

sum_cosncosn = 0.
sum_cosrcosr = 0.
sum_coskcosk = 0.

sum_cosncosr = 0.
sum_cosncosk = 0.
sum_cosrcosk = 0.

sum_cosrcosn = 0.
sum_coskcosn = 0.
sum_coskcosr = 0.

array_cosncosn = array([],'f')

print(array_cosncosn)

#Cij: i = tau+ = row, j = tau- column 
count=0
for i in range(1,tree.GetEntries()+1):
    tree.GetEntry(i)
    sum_cosncosn += tree.cosn_plus*tree.cosn_minus 
    sum_cosrcosr += tree.cosr_plus*tree.cosr_minus
    sum_coskcosk += tree.cosk_plus*tree.cosk_minus
    
    sum_cosncosr += tree.cosn_plus*tree.cosr_minus
    sum_cosncosk += tree.cosn_plus*tree.cosk_minus
    sum_cosrcosk += tree.cosr_plus*tree.cosk_minus
    
    sum_cosrcosn += tree.cosr_plus*tree.cosn_minus
    sum_coskcosn += tree.cosk_plus*tree.cosn_minus
    sum_coskcosr += tree.cosk_plus*tree.cosr_minus
    count += 1

# note currently not sure where the minus signs come from below but they are needed to get the correct matrix, although it doesn't change the entanglement variables at all anyway...
C11 = -sum_cosncosn/count*9
C22 = -sum_cosrcosr/count*9
C33 = -sum_coskcosk/count*9
C12 = -sum_cosncosr/count*9
C13 = -sum_cosncosk/count*9
C23 = -sum_cosrcosk/count*9
C21 = -sum_cosrcosn/count*9
C31 = -sum_coskcosn/count*9
C32 = -sum_coskcosr/count*9

C = np.array([[C11, C12, C13],
              [C21, C22, C23],
              [C31, C32, C33]])

print('C = ')
print(C)

con, m12 = EntanglementVariables(C)

print('concurrence = %.4f' % con)
print('m12 = %.3f' % m12)
