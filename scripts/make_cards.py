import numpy as np
from numpy import arange
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mass', '-m', help= 'Mass to use for the heavy Higgs in the nominal sample', default='600')
parser.add_argument('--masses', help= 'Masses to use for the heavy Higgs in the reweighting', default='')
parser.add_argument('--output', '-o', help= 'Name of output directory',default='MCCards')
parser.add_argument('--width', '-w', help= 'Fractional width to use for the nominal events', type=float, default=0.01)
parser.add_argument('--noH', help= 'If this option is specified then the heavy scalar H is excluded from the generation', action='store_true')
args = parser.parse_args()

mass=float(args.mass)

masses = args.masses.split(',')
masses = [float(i) for i in masses]

if mass not in masses: masses.append(mass)

reweight_out_string='\
change process p p > h h [QCD] / iota0\n\
\n\
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

param_base_file=open("../Cards/param_card_BSM.dat","r")

nominal_frac_width = args.width

## for generating samples scale the s-channel H production to ensure we get plenty of events near resonance peak but still a good number of events in the non-resonant part
## this method is quite approximate and could be improved
##kap_sf = (1.5/300.*(mass-300.) + 0.5)*nominal_frac_width/0.05
##kap112_param = 31.80387825*kap_sf

kap112_param = 31.80387825

if args.noH: kap112_param = 0.

# set width of generated sample to largest width
param_out_string = param_base_file.read().replace('$W','%g' % (nominal_frac_width*mass)).replace('$M','%g' % mass).replace('$k','%g' % kap112_param)

with open("%s/param_card.dat" % args.output, "w") as param_file:
    param_file.write(param_out_string)

for m in masses:

    widths = np.array([0.001,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.011,0.012,0.013,0.014,0.15])*m 

    # append exact widths for BM scenarios
    if m==600:
        widths = np.append(widths,[4.979180/m])
    elif m==300:
        widths = np.append(widths,[0.5406704/m])
    
    for width in widths:
        if len(masses)==1 and float(masses[0]) == float(mass):
            reweight_out_string+=mass_and_width_dep_string.replace('$postfix',('RelativeWidt%g' % (width)).replace('.','p')).replace('$W','%g' % width*m).replace('$M','%g' % m)
 
        else: reweight_out_string+=mass_and_width_dep_string.replace('$postfix',('Mass%g_RelativeWidth%g' % (m,width)).replace('.','p')).replace('$W','%g' % width*m).replace('$M','%g' % m)
    
    with open("%s/reweight_card.dat" % args.output, "w") as reweight_file:
        reweight_file.write(reweight_out_string)
    
