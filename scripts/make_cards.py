import numpy as np
from numpy import arange
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mass', '-m', help= 'Mass to use for the heavy Higgs in the nominal sample', default='600')
parser.add_argument('--masses', help= 'Masses to use for the heavy Higgs in the reweighting', default='')
parser.add_argument('--output', '-o', help= 'Name of output directory',default='MCCards')
args = parser.parse_args()

mass=float(args.mass)

masses = args.masses.split(',')
masses = [float(i) for i in masses]

if mass not in masses: masses.append(mass)

reweight_out_string='\
launch --rwgt_name=box\n\
  set bsm 6 0.785398\n\
  set bsm 15 0.000000e+00\n\
  set bsm 16 0.000000e+00\n\
\n\
launch --rwgt_name=box_and_schannel_h_1\n\
  set bsm 6 0.785398\n\
  set bsm 15 31.803878252\n\
  set bsm 16 0.000000e+00\n\
\n\
launch --rwgt_name=box_and_schannel_h_2\n\
  set bsm 6 0.785398\n\
  set bsm 15 318.03878252\n\
  set bsm 16 0.000000e+00\n\
\n\
launch --rwgt_name=all\n\
  set bsm 6 0.785398\n\
  set bsm 15 31.803878252\n\
  set bsm 16 31.803878252\n\
\n\
launch --rwgt_name=schannel_H\n\
  set bsm 6 1.570796\n\
  set bsm 15 0.000000e+00\n\
  set bsm 16 31.803878252\n\
\n\
launch --rwgt_name=box_and_schannel_H_1\n\
  set bsm 6 0.785398\n\
  set bsm 15 0.000000e+00\n\
  set bsm 16 31.803878252\n\
\n\
launch --rwgt_name=box_and_schannel_H_2\n\
  set bsm 6 0.785398\n\
  set bsm 15 0.000000e+00\n\
  set bsm 16 318.03878252\n\n'

mass_and_width_dep_string='\
launch --rwgt_name=all_$postfix\n\
  set bsm 6 0.785398\n\
  set bsm 15 31.803878252\n\
  set bsm 16 31.803878252\n\
  set mass 99925 $M\n\
  set DECAY 99925 $W\n\
\n\
launch --rwgt_name=schannel_H_$postfix\n\
  set bsm 6 1.570796\n\
  set bsm 15 0.000000e+00\n\
  set bsm 16 31.803878252\n\
  set mass 99925 $M\n\
  set DECAY 99925 $W\n\
\n\
launch --rwgt_name=box_and_schannel_H_1_$postfix\n\
  set bsm 6 0.785398\n\
  set bsm 15 0.000000e+00\n\
  set bsm 16 31.803878252\n\
  set mass 99925 $M\n\
  set DECAY 99925 $W\n\
\n\
launch --rwgt_name=box_and_schannel_H_2_$postfix\n\
  set bsm 6 0.785398\n\
  set bsm 15 0.000000e+00\n\
  set bsm 16 318.03878252\n\
  set mass 99925 $M\n\
  set DECAY 99925 $W\n\n'

param_base_file=open("Cards/param_card_BSM.dat","r")

# for generating samples scale the s-channel H production to ensure we get plenty of events near resonance peak but still a good number of events in the non-resonant part
# this method is quite approximate and could be improved
kap_sf = 1.5/300.*(mass-300.) + 0.5
kap112_param = 31.80387825*kap_sf

# set width of generated sample to largest width
param_out_string = param_base_file.read().replace('$W','%g' % (0.05*mass)).replace('$M','%g' % mass).replace('$k','%g' % kap112_param)

with open("%s/param_card.dat" % args.output, "w") as param_file:
    param_file.write(param_out_string)

for m in masses:

    # widths go from 0.5%-5% in 1% intervals
    sep=0.01
    widths = arange(sep,0.05+sep,sep)*m
    
    # append exact widths for BM scenario
    if m==600:
        widths = np.append(widths,[4.979180])
    elif m==300:
        widths = np.append(widths,[0.5406704])
    
    # Add 0.5 GeV width as the smallest width if a smaller width was not already used 
    if 0.5 not in widths and widths[0]>0.5:
        widths  = np.insert(widths,0,0.5)
    
    for width in widths:
        reweight_out_string+=mass_and_width_dep_string.replace('$postfix',('Mass%g_Width%g' % (m,width)).replace('.','p')).replace('$W','%g' % width).replace('$M','%g' % m)
    
    with open("%s/reweight_card.dat" % args.output, "w") as reweight_file:
        reweight_file.write(reweight_out_string)
    
