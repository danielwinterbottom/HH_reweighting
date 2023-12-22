import os

def WriteReweightCard(out_name, masses=[600,650], widths=[0.001,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15]):
    reweight_out_string='\
change rwgt_dir ./rwgt\n\
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
\n'

    mass_and_width_dep_string = '\
launch --rwgt_name=all_$postfix\n\
  set bsm 6 0.785398\n\
  set bsm 15 31.803878252\n\
  set bsm 16 31.803878252\n\
  set mass 99925 $M\n\
  set decay 99925 $W\n\
\n\
launch --rwgt_name=schannel_H_$postfix\n\
  set bsm 6 1.570796\n\
  set bsm 15 0.000000e+00\n\
  set bsm 16 31.803878252\n\
  set mass 99925 $M\n\
  set decay 99925 $W\n\
\n\
launch --rwgt_name=box_and_schannel_H_1_$postfix\n\
  set bsm 6 0.785398\n\
  set bsm 15 0.000000e+00\n\
  set bsm 16 31.803878252\n\
  set mass 99925 $M\n\
  set decay 99925 $W\n\
\n\
launch --rwgt_name=box_and_schannel_H_2_$postfix\n\
  set bsm 6 0.785398\n\
  set bsm 15 0.000000e+00\n\
  set bsm 16 318.03878252\n\
  set mass 99925 $M\n\
  set decay 99925 $W\n\n'

    for mass in masses:
        for w in widths:
            postfix = ('Mass_%g_RelWidth_%g' % (mass,w)).replace('.','p')
            reweight_out_string += mass_and_width_dep_string.replace('$M', '%g' % float(mass)).replace('$W', '%g' % (float(mass)*w)).replace('$postfix', postfix)

    with open(out_name, "w") as reweightcard_file:
        reweightcard_file.write(reweight_out_string)

card_name = 'reweight_card.dat'
WriteReweightCard(card_name, masses=[600],widths=[0.01])
