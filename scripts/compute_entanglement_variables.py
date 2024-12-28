import argparse
import ROOT
import numpy as np
import uproot
import pandas as pd
from entanglement_funcs import EntanglementVariables
from array import array


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Input ROOT file.")
parser.add_argument('--n_events', '-n', help='Maximum number of events to process', default=-1, type=int)
parser.add_argument('--n_skip', '-s', help='skip n_events*n_skip', default=0, type=int)

args = parser.parse_args()

with uproot.open(args.input) as file:
    tree = file["new_tree"]

    total_entries = tree.num_entries
    entry_start = args.n_skip * args.n_events if args.n_events != -1 else 0
    entry_stop = total_entries if args.n_events == -1 else min(entry_start + args.n_events, total_entries)

    df = tree.arrays(entry_start=entry_start, entry_stop=entry_stop, library="pd")

    # example below applys a cut on cosTheta
    #df = df[abs(df['cosTheta']) < 0.5]

df["cosncosn"] = df["cosn_plus"]*df["cosn_minus"]
df["cosrcosr"] = df["cosr_plus"]*df["cosr_minus"]
df["coskcosk"] = df["cosk_plus"]*df["cosk_minus"]
df["cosncosr"] = df["cosn_plus"]*df["cosr_minus"]
df["cosncosk"] = df["cosn_plus"]*df["cosk_minus"]
df["cosrcosk"] = df["cosr_plus"]*df["cosk_minus"]
df["cosrcosn"] = df["cosr_plus"]*df["cosn_minus"]
df["coskcosn"] = df["cosk_plus"]*df["cosn_minus"]
df["coskcosr"] = df["cosk_plus"]*df["cosr_minus"]


def ComputeEntanglementVariables(df, verbose=False):

    # note currently not sure where the minus signs come from below but they are needed to get the correct matrix, although it doesn't change the entanglement variables at all anyway...
    C11 = -df["cosncosn"].mean()*9
    C22 = -df["cosrcosr"].mean()*9
    C33 = -df["coskcosk"].mean()*9
    C12 = -df["cosncosr"].mean()*9
    C13 = -df["cosncosk"].mean()*9
    C23 = -df["cosrcosk"].mean()*9
    C21 = -df["cosrcosn"].mean()*9
    C31 = -df["coskcosn"].mean()*9
    C32 = -df["coskcosr"].mean()*9
    
    C = np.array([[C11, C12, C13],
                  [C21, C22, C23],
                  [C31, C32, C33]])
    
    
    con, m12 = EntanglementVariables(C)
    
    if verbose:
        print('C = ')
        print(C)
        print('concurrence = %.4f' % con)
        print('m12 = %.3f' % m12)
    return(con, m12)

con, m12 = ComputeEntanglementVariables(df, True)

N = 100  # Number of bootstrap samples to generate
bootstrap_samples = []

bs_con_vals = array('d')
bs_m12_vals = array('d')

# Generate N bootstrap samples
for i in range(N):
    sample = df.sample(n=len(df), replace=True)
    bootstrap_samples.append(sample)
    sample_con, sample_m12 = ComputeEntanglementVariables(sample)

    bs_con_vals.append(sample_con)
    bs_m12_vals.append(sample_m12)

print('\nconcurrence = %.4f +/- %.4f' %(con,np.std(bs_con_vals)))
print('m12 = %.4f +/- %.4f' %(m12,np.std(bs_m12_vals)))


