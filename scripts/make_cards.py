from numpy import arange
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mass', '-m', help= 'Mass to use for the heavy Higgs', type=float, default=600.)
parser.add_argument('--output', '-o', help= 'Name of output directory',default='MCCards')
args = parser.parse_args()

mass=args.mass

# width go from 0.2%-2% in 0.2% intervals
widths = arange(0.002,0.022,0.002)*mass

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
  set bsm 16 0.000000e+00\n\n'

width_dep_string='\
launch --rwgt_name=all_$postfix\n\
  set bsm 6 0.785398\n\
  set bsm 15 31.803878252\n\
  set bsm 16 31.803878252\n\
  set DECAY 99925 $W\n\
\n\
launch --rwgt_name=schannel_H_$postfix\n\
  set bsm 6 1.570796\n\
  set bsm 15 0.000000e+00\n\
  set bsm 16 31.803878252\n\
  set DECAY 99925 $W\n\
\n\
launch --rwgt_name=box_and_schannel_H_1_$postfix\n\
  set bsm 6 0.785398\n\
  set bsm 15 0.000000e+00\n\
  set bsm 16 31.803878252\n\
  set DECAY 99925 $W\n\
\n\
launch --rwgt_name=box_and_schannel_H_2_$postfix\n\
  set bsm 6 0.785398\n\
  set bsm 15 0.000000e+00\n\
  set bsm 16 318.03878252\n\
  set DECAY 99925 $W\n\n'

for width in widths:
    reweight_out_string+=width_dep_string.replace('$postfix',('Width%g' % width).replace('.','p')).replace('$W','%g' % width)

param_base_file=open("Cards/param_card_BSM.dat","r")

param_out_string = param_base_file.read().replace('$W','%g' % width).replace('$M','%g' % mass)

with open("%s/reweight_card.dat" % args.output, "w") as reweight_file:
    reweight_file.write(reweight_out_string)

with open("%s/param_card.dat" % args.output, "w") as param_file:
    param_file.write(param_out_string)

