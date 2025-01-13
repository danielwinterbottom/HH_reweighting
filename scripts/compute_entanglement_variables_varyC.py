import argparse
import ROOT
import numpy as np
import uproot
import pandas as pd
from entanglement_funcs import EntanglementVariables
from array import array
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input_full", required=True, help="Input ROOT file.")
parser.add_argument("--input_noC", required=True, help="Input ROOT file.")
parser.add_argument('--n_events', '-n', help='Maximum number of events to process', default=-1, type=int)
parser.add_argument('--n_skip', '-s', help='skip n_events*n_skip', default=0, type=int)
parser.add_argument('--f', '-f', help='f param to control amount of entanglement', default=1., type=float)

args = parser.parse_args()

def LoadFile(file_name):

    with uproot.open(file_name) as file:
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
    
    return df

def GetBinnedMean(df1, df2, f, var, binrange=(-1,1), Nbins=100, plot_hists=False):
    bins = np.linspace(binrange[0], binrange[1], Nbins)
    hist1, bin_edges = np.histogram(df1[var], bins=bins)
    hist2, _ = np.histogram(df2[var], bins=bins)

    # hist1 = full model
    # hist2 = no entanglement model
    summed_hist = f*hist1 + (1. - f)*hist2

    # Calculate bin midpoints
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    ## Print bin contents
    #print("Bin Midpoints and Corresponding Frequencies:")
    #for midpoint, frequency in zip(bin_midpoints, summed_hist):
    #    print(f"Bin {midpoint:.2f}: Frequency {frequency:.2f}")

    # do some checks to make sure resulting histogram is physical
    # Calculate integrals
    total_integral = np.sum(summed_hist)
    above_zero_integral = np.sum(summed_hist[bin_midpoints > 0])
    below_zero_integral = np.sum(summed_hist[bin_midpoints < 0])
    
    # Ensure total integral is not negative and integrals for +ve and -ve values are also non regative
    if total_integral <= 0:
        raise ValueError("The total integral of the histogram is not > 0.")
    if above_zero_integral <= 0:
        raise ValueError("The integral of bins above 0 is not > 0.")
    if below_zero_integral <= 0:
        raise ValueError("The integral of bins below 0 is not > 0.")

    ## Check that all bins are > 0
    #if np.any(summed_hist <= 0):
    #    raise ValueError("Not all bins in the resulting histogram have values > 0.")

    # Calculate mean of the summed histogram
    mean = np.average(bin_midpoints, weights=summed_hist)

    if plot_hists:
        #plt.hist(df1[var], bins=bins, alpha=0.5, label=f'df1 (scaled by {f:.2f})', weights=f * np.ones(len(df1)))
        #plt.hist(df2[var], bins=bins, alpha=0.5, label=f'df2 (scaled by {1-f:.2f})', weights=(1 - f) * np.ones(len(df2)))
        plt.step(bin_midpoints, summed_hist, where='mid', label='Summed Histogram (scaled)', color='red')
        plt.axvline(mean, color='black', linestyle='--', label=f'Mean: {mean:.2f}')
        plt.legend()
        plt.xlabel(var)
        plt.ylabel('N')
        plt.savefig("histogram_output_F%s_%s.pdf" % (('%g' % f).replace('.','p'), var), format="pdf")

    return mean

def ComputeEntanglementVariables(df1, df2, f, verbose=False):

    # note currently not sure where the minus signs come from below but they are needed to get the correct matrix, although it doesn't change the entanglement variables at all anyway...
    C11 = -GetBinnedMean(df1, df2, f, "cosncosn", plot_hists=True)*9
    C22 = -GetBinnedMean(df1, df2, f, "cosrcosr", plot_hists=True)*9
    C33 = -GetBinnedMean(df1, df2, f, "coskcosk", plot_hists=True)*9
    C12 = -GetBinnedMean(df1, df2, f, "cosncosr", plot_hists=True)*9
    C13 = -GetBinnedMean(df1, df2, f, "cosncosk", plot_hists=True)*9
    C23 = -GetBinnedMean(df1, df2, f, "cosrcosk", plot_hists=True)*9
    C21 = -GetBinnedMean(df1, df2, f, "cosrcosn", plot_hists=True)*9
    C31 = -GetBinnedMean(df1, df2, f, "coskcosn", plot_hists=True)*9
    C32 = -GetBinnedMean(df1, df2, f, "coskcosr", plot_hists=True)*9
    
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

df1 = LoadFile(args.input_full)
df2 = LoadFile(args.input_noC)

rows_df1 = len(df1)
rows_df2 = len(df2)

if rows_df1 != rows_df2:
    # Find the smaller size
    min_rows = min(rows_df1, rows_df2)

    # Trim both DataFrames to match the smaller size
    df1 = df1.iloc[:min_rows]
    df2 = df2.iloc[:min_rows]

# the f parameter controls the amount of entanglement
# f = 0 corresponds to no entanglement
# f = 1 corresponds to predicted entanglement
f = args.f 

con, m12 = ComputeEntanglementVariables(df1, df2, f=f, verbose=True)

#N = 100  # Number of bootstrap samples to generate
#bootstrap_samples = []
#
#bs_con_vals = array('d')
#bs_m12_vals = array('d')
#
## Generate N bootstrap samples
#for i in range(N):
#    sample = df.sample(n=len(df), replace=True)
#    bootstrap_samples.append(sample)
#    sample_con, sample_m12 = ComputeEntanglementVariables(sample)
#
#    bs_con_vals.append(sample_con)
#    bs_m12_vals.append(sample_m12)
#
##bs_con_vals = np.random.normal(loc=0, scale=1, size=100)
#
#print('\nconcurrence = %.4f +/- %.4f' %(con,np.std(bs_con_vals)))
#print('m12 = %.4f +/- %.4f' %(m12,np.std(bs_m12_vals)))
#
#bs_con_vals = np.array(bs_con_vals)
#bs_m12_vals = np.array(bs_m12_vals)
#
## get assymetric errors
#mean_con = np.mean(bs_con_vals)
#con_hi = np.sqrt(np.mean((bs_con_vals[bs_con_vals >= mean_con] - mean_con)**2))
#con_lo = np.sqrt(np.mean((mean_con - bs_con_vals[bs_con_vals < mean_con])**2))
#
#mean_m12 = np.mean(bs_m12_vals)
#m12_hi = np.sqrt(np.mean((bs_m12_vals[bs_m12_vals >= mean_m12] - mean_m12)**2))
#m12_lo = np.sqrt(np.mean((mean_m12 - bs_m12_vals[bs_m12_vals < mean_m12])**2))
#
#print('\nconcurrence = %.4f +/- +%.4f/%.4f' %(con,con_hi,con_lo))
#print('m12 = %.4f +/- +%.4f/%.4f' %(m12,m12_hi,m12_lo))
#
## use percentiles instead to get error
#con_perc_lo = np.percentile(bs_con_vals, 16)
#con_perc_hi = np.percentile(bs_con_vals, 84)
#m12_perc_lo = np.percentile(bs_m12_vals, 16)
#m12_perc_hi = np.percentile(bs_m12_vals, 84)
#print('\nconcurrence = %.4f +/- +%.4f/%.4f' %(con,con_perc_hi-con,con_perc_lo-con))
#print('m12 = %.4f +/- +%.4f/%.4f' %(m12,m12_perc_hi-m12,m12_perc_lo-m12))
