import ROOT
import plotting
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--reco_cuts', help= 'apply reco pT and eta cuts', action='store_true')
parser.add_argument('--bm', help= 'benchmark to use for plots comparing w/o inteference', default='singlet_M600')
args = parser.parse_args()

bm_name = args.bm

f1 = ROOT.TFile('outputs_4b_Feb13/output_powheg_pythia_sm.root')
f2 = ROOT.TFile('outputs_4b_Feb13/output_powheg_pythia_box.root')
f3 = ROOT.TFile('outputs_4b_Feb13/output_powheg_pythia_chhh10.root')

f4 = ROOT.TFile('outputs_4b_Feb13/output_powheg_pythia_from_single_H_width_5GeV.root')
f5 = ROOT.TFile('outputs_4b_Feb13/output_powheg_pythia_from_single_H_width_12GeV.root')
#f6 = ROOT.TFile('outputs_4b_Feb27/output_mg_pythia_sm_new_reweighted.root')
f6 = ROOT.TFile('outputs_4b_Mar12_v2/output_mg_pythia_sm_reweighted.root')
#if bm_name == 'singlet_M600':
#  f7 = ROOT.TFile('outputs_4b_Feb13/output_mg_pythia_width_5GeV.root')
if 'M260' in bm_name:
  f7 = ROOT.TFile('outputs_4b_Mar12_v2/output_mg_pythia_mass_260GeV_relWidth0p001_reweighted.root')  
elif 'M380' in bm_name:
  f7 = ROOT.TFile('outputs_4b_Mar12_v2/output_mg_pythia_mass_380GeV_relWidth0p002_reweighted.root')
elif 'M440' in bm_name:
  f7 = ROOT.TFile('outputs_4b_Mar12_v2/output_mg_pythia_mass_440GeV_relWidth0p003_reweighted.root')  
elif 'M500' in bm_name:
  f7 = ROOT.TFile('outputs_4b_Mar12_v2/output_mg_pythia_mass_500GeV_relWidth0p003_reweighted.root')
elif 'M560' in bm_name:
  f7 = ROOT.TFile('outputs_4b_Mar12_v2/output_mg_pythia_mass_560GeV_relWidth0p005_reweighted.root') 
elif 'M600' in bm_name:
  #f7 = ROOT.TFile('outputs_4b_Mar12_v2/output_mg_pythia_mass_600GeV_relWidth0p008333_reweighted.root')
  f7 = ROOT.TFile('outputs_4b_Mar12_v2/output_mg_pythia_mass_600GeV_relWidth0p02_reweighted.root')
elif 'M620' in bm_name:
  f7 = ROOT.TFile('outputs_4b_Mar12_v2/output_mg_pythia_mass_620GeV_relWidth0p007_reweighted.root') 
  #f7 = ROOT.TFile('outputs_4b_Mar11/output_mg_pythia_mass_620GeV_relWidth0p007_10k_reweighted.root')  
elif 'M680' in bm_name:
  f7 = ROOT.TFile('outputs_4b_Mar12_v2/output_mg_pythia_mass_680GeV_relWidth0p009_reweighted.root')     
elif 'M800' in bm_name:
  f7 = ROOT.TFile('outputs_4b_Mar12_v2/output_mg_pythia_mass_800GeV_relWidth0p01_reweighted.root')
elif 'M870' in bm_name:
  f7 = ROOT.TFile('outputs_4b_Mar12_v2/output_mg_pythia_mass_870GeV_relWidth0p01_reweighted.root')  
else:
  f7 = ROOT.TFile('outputs_4b_Feb13/output_mg_pythia_width_5GeV.root')  
f8 = ROOT.TFile('outputs_4b_Feb13/output_mg_pythia_BM.root')

benchmarks = {}

benchmarks['singlet_M600'] = {
#  # sina=0.17, tanb=1.5, M=600
  'kappa_h_t' : 0.9854491056576354,
  'kappa_H_t' : 0.16997076265807162,
  'kappa_h_lam' : 0.9491226120544515,
  'kappa_H_lam' : 5.266738184342865,
  'width' : 4.98,
  'rel_width' : 0.008333,
  'mass': 600.,
  'label': 'sin#alpha=0.17, tan#beta=1.5, m_{H}=600 GeV, #Gamma_{H}=5.0 GeV'
}

benchmarks['singlet_M600_new'] = {
#  # sina=0.17, tanb=1.5, M=600
  'kappa_h_t' : 0.9854491056576354,
  'kappa_H_t' : 0.16997076265807162,
  'kappa_h_lam' : 0.9491226120544515,
  'kappa_H_lam' : 5.266738184342865,
  'width' : 4.98,
  'rel_width' : 0.008333,
  'mass': 600.,
  'label': 'sin#alpha=0.17, tan#beta=1.5, m_{H}=600 GeV, #Gamma_{H}=5.0 GeV'
}

#benchmarks['template'] = {
#  # sina=, tanb=, M=
#  'kappa_h_t' : ,
#  'kappa_H_t' : ,
#  'kappa_h_lam' : ,
#  'kappa_H_lam' : ,
#  'width' : ,
#  'rel_width': ,
#  'mass': ,
#  'label': 'sin#alpha=, tan#beta=, m_{H}= GeV, #Gamma_{H}= GeV'
#}

benchmarks['singlet_M260'] = {
  # sina=0.24, tanb=3.5, M=260
  # maximum deviation of lambda_hhh from SM value
  'kappa_h_t' : 0.97188150278,
  'kappa_H_t' : 0.24,
  'kappa_h_lam' : 0.866472369614,
  'kappa_H_lam' : 2.66900576224,
  'width' : 0.5718486648000001,
  'rel_width': 0.002199,
  'mass': 260,
  'label': 'sin#alpha=0.24, tan#beta=3.5, m_{H}=260 GeV, #Gamma_{H}=0.6 GeV',
  #'xs_SH_nnlo' : 11.76,
  #'xs_SH_lo':  3.8599
}

benchmarks['singlet_M380'] = {
  # sina=0.16, tanb=0.5, M=380
  # maximum split inteference effects (above and below mH) when effect on total cross-section is < 1%
  'kappa_h_t' : 0.987335676014,
  'kappa_H_t' : 0.16,
  'kappa_h_lam' : 0.959798818773,
  'kappa_H_lam' : 1.89464912164,
  'width' : 0.8433558311999999,
  'rel_width': 0.002219,
  'mass': 380.,
  'label': 'sin#alpha=0.16, tan#beta=0.5, m_{H}=380 GeV, #Gamma_{H}=0.8 GeV',
  'xs_SH_nnlo' : 10.4,
  'xs_SH_lo': 3.5208
}

benchmarks['singlet_M440'] = {
  # sina=0.16, tanb=0.5, M=440
  # maximum split inteference effects (above and below mH)
  'kappa_h_t' : 0.987335676014,
  'kappa_H_t' : 0.16,
  'kappa_h_lam' : 0.959798818773,
  'kappa_H_lam' : 2.42534503274,
  'width' : 1.4619768815999998,
  'rel_width': 0.003323,
  'mass': 440,
  'label': 'sin#alpha=0.16, tan#beta=0.5, m_{H}=440 GeV, #Gamma_{H}=1.5 GeV',
  'xs_SH_nnlo' : 7.301,
  'xs_SH_lo': 2.56210 
}

benchmarks['singlet_M500'] = {
  # sina=0.08, tanb=0.5, M=500
  # largest fraction of non-resonant HH within +/-10% of mH for HL-LHC-like sensitivity
  'kappa_h_t' : 0.996808519881,
  'kappa_H_t' : 0.08,
  'kappa_h_lam' : 0.990159376423,
  'kappa_H_lam' : 1.48819938414,
  'width' : 0.5576455704000001,
  'rel_width': 0.001115,
  'mass': 500.,
  'label': 'sin#alpha=0.08, tan#beta=0.5, m_{H}=500 GeV, #Gamma_{H}=0.6 GeV',
  'xs_SH_nnlo' : 4.538,
  'xs_SH_lo': 1.6224  
}

benchmarks['singlet_M560'] = {
  # sina=-0.16, tanb=0.6, M=560
  # largest fraction of non-resonant HH within +/-10% of mH for Run-3 like sensitivity
  'kappa_h_t' : 0.987335676014,
  'kappa_H_t' : -0.16,
  'kappa_h_lam' : 0.963894818773,
  'kappa_H_lam' : -3.16200197948,
  'width' : 2.9865183695999997,
  'rel_width': 0.005333,
  'mass': 560.,
  'label': 'sin#alpha=-0.16, tan#beta=0.5, m_{H}=560 GeV, #Gamma_{H}=3.0 GeV',
  'xs_SH_nnlo' : 2.8076,
  'xs_SH_lo': 1.0046 
}

benchmarks['singlet_M620'] = {
  # sina=0.16, tanb=1.0, M=620
  # largest cross-section difference due to inteference
  'kappa_h_t' : 0.987335676014,
  'kappa_H_t' : 0.16,
  'kappa_h_lam' : 0.957750818773,
  'kappa_H_lam' : 4.8195234808,
  'width' : 4.6014512208000005,
  'rel_width': 0.007422,
  'mass': 620,
  'label': 'sin#alpha=0.16, tan#beta=1.0, m_{H}=620 GeV, #Gamma_{H}=4.6 GeV',
  'xs_SH_nnlo' : 1.7444,
  'xs_SH_lo': 0.62273, 
}

benchmarks['singlet_M680'] = {
  # sina=0.16, tanb=1.0, M=680
  # largest cross-section difference due to inteference
  'kappa_h_t' : 0.987335676014,
  'kappa_H_t' : 0.16,
  'kappa_h_lam' : 0.957750818773,
  'kappa_H_lam' : 5.72394506507,
  'width' : 6.069265895999999,
  'rel_width': 0.008925,
  'mass': 680,
  'label': 'sin#alpha=0.16, tan#beta=1.0, m_{H}=680 GeV, #Gamma_{H}=6.1 GeV',
  'xs_SH_nnlo' : 1.0949,
  'xs_SH_lo':  0.39366
}

benchmarks['singlet_M800'] = {
  # sina=0.16, tanb=1.0, M=800
  'kappa_h_t' : 0.987335676014,
  'kappa_H_t' : 0.16,
  'kappa_h_lam' : 0.957750818773,
  'kappa_H_lam' : 7.78324344156,
  'width' : 9.8111474352,
  'rel_width': 0.01226,
  'mass': 800.,
  'label': 'sin#alpha=0.16, tan#beta=1.0, m_{H}=800 GeV, #Gamma_{H}=9.8 GeV'
}
#870.0 0.149999996647 1.10000001639
benchmarks['singlet_M870'] = {
  # sina=0.15, tanb=1.1, M=870
  'kappa_h_t' : 0.98885488629,
  'kappa_H_t' : 0.15,
  'kappa_h_lam' : 0.962728061739,
  'kappa_H_lam' : 8.63030446334,
  'width' : 9.544587,
  'rel_width': 0.01097,
  'mass': 870.,
  'label': 'sin#alpha=0.15, tan#beta=1.1, m_{H}=870 GeV, #Gamma_{H}=9.5 GeV',
  'xs_SH_nnlo' : 0.28374,
  'xs_SH_lo':  0.1027600000000024
}




######### old BMs below #########
#
#benchmarks['singlet_M380_1'] = {
#  # sina=0.16, tanb=2.5, M=380
#  'kappa_h_t' : 0.987335676014,
#  'kappa_H_t' : 0.16,
#  'kappa_h_lam' : 0.951606818773,
#  'kappa_H_lam' : 2.46280398262,
#  'width' : 1.0402459311999999,
#  'rel_width': 0.002737,
#  'mass': 380.,
#  'label': 'sin#alpha=0.16, tan#beta=2.5, m_{H}=380 GeV, #Gamma_{H}=1.0 GeV'
#}
#
#benchmarks['singlet_M380_2'] = {
#  # sina=0.08, tanb=2.5, M=380
#  'kappa_h_t' : 0.996808519881,
#  'kappa_H_t' : 0.08,
#  'kappa_h_lam' : 0.989135376423,
#  'kappa_H_lam' : 1.07286140701,
#  'width' : 0.2310074328,
#  'rel_width': 0.0006079,
#  'mass': 380.,
#  'label': 'sin#alpha=0.16, tan#beta=2.5, m_{H}=380 GeV, #Gamma_{H}=1.0 GeV'
#}
#
#benchmarks['singlet_M260_1'] = {
#  # sina=0.24, tanb=3.5, M=260
#  'kappa_h_t' : 0.97188150278,
#  'kappa_H_t' : 0.24,
#  'kappa_h_lam' : 0.866472369614,
#  'kappa_H_lam' : 2.66900576224,
#  'width' : 0.5718486648000001,
#  'rel_width': 0.002199,
#  'mass': 260.,
#  'label': 'sin#alpha=0.24, tan#beta=3.5, m_{H}=260 GeV, #Gamma_{H}=0.6 GeV'
#}
#
#benchmarks['singlet_M260_2'] = {
#  # sina=0.16, tanb=3.5, M=260
#  'kappa_h_t' : 0.987335676014,
#  'kappa_H_t' : 0.16,
#  'kappa_h_lam' : 0.947510818773,
#  'kappa_H_lam' : 1.54585384393,
#  'width' : 0.2214987288,
#  'rel_width': 0.0008519,
#  'mass': 260.,
#  'label': 'sin#alpha=0.16, tan#beta=3.5, m_{H}=260 GeV, #Gamma_{H}=0.2 GeV'
#}
#
#benchmarks['singlet_M260_3'] = {
#  # sina=-0.16, tanb=3.5, M=260
#  'kappa_h_t' : 0.987335676014,
#  'kappa_H_t' : -0.16,
#  'kappa_h_lam' : 0.976182818773,
#  'kappa_H_lam' : -0.426768287275,
#  'width' : 0.1286585288,
#  'rel_width': 0.0004948,
#  'mass': 260.,
#  'label': 'sin#alpha=-0.16, tan#beta=3.5, m_{H}=260 GeV, #Gamma_{H}=0.1 GeV'
#}
#
#benchmarks['singlet_M500_1'] = {
#  # sina=0.16, tanb=1.5, M=500
#  'kappa_h_t' : 0.987335676014,
#  'kappa_H_t' : 0.16,
#  'kappa_h_lam' : 0.955702818773,
#  'kappa_H_lam' : 3.48856728031,
#  'width' : 2.4607704816,
#  'rel_width': 0.0049212,
#  'mass': 500.,
#  'label': 'sin#alpha=0.16, tan#beta=1.5, m_{H}=500 GeV, #Gamma_{H}=2.5 GeV'
#}
#
#benchmarks['singlet_M500_2'] = {
#  # sina=0.08, tanb=1.5, M=500
#  'kappa_h_t' : 0.996808519881,
#  'kappa_H_t' : 0.08,
#  'kappa_h_lam' : 0.989647376423,
#  'kappa_H_lam' : 1.60303015242,
#  'width' : 0.5823255704,
#  'rel_width': 0.001164,
#  'mass': 500.,
#  'label': 'sin#alpha=0.08, tan#beta=1.5, m_{H}=500 GeV, #Gamma_{H}=0.6 GeV'
#}
#
#benchmarks['singlet_M800_1'] = {
#  # sina=0.16, tanb=1.0, M=800
#  'kappa_h_t' : 0.987335676014,
#  'kappa_H_t' : 0.16,
#  'kappa_h_lam' : 0.957750818773,
#  'kappa_H_lam' : 7.78324344156,
#  'width' : 9.8111474352,
#  'rel_width': 0.01226,
#  'mass': 800.,
#  'label': 'sin#alpha=0.16, tan#beta=1.0, m_{H}=800 GeV, #Gamma_{H}=9.8 GeV'
#}
#
#benchmarks['singlet_M800_2'] = {
#  # sina=0.08, tanb=1.0, M=800
#  'kappa_h_t' : 0.996808519881,
#  'kappa_H_t' : 0.08,
#  'kappa_h_lam' : 0.989903376423,
#  'kappa_H_lam' : 3.68886724696,
#  'width' : 2.3795691088,
#  'rel_width': 0.00297,
#  'mass': 800.,
#  'label': 'sin#alpha=0.08, tan#beta=1.0, m_{H}=800 GeV, #Gamma_{H}=2.4 GeV'
#}

bm = benchmarks[bm_name]


kappa_h_t = bm['kappa_h_t']
kappa_H_t = bm['kappa_H_t']
kappa_h_lam = bm['kappa_h_lam']
kappa_H_lam = bm['kappa_H_lam']

rel_width_str = str(bm['rel_width']).replace('.','p')
mass_str = '%.0f' % bm['mass']

norm_hists=False

partial_width = 0.06098 # partial width for kap112=1 and M=600 GeV

xs_box_nnlo = 70.3874
xs_Sh_nnlo = 11.0595
xs_box_Sh_int_nnlo = -50.4111

xs_box_lo = 27.4968
xs_Sh_lo = 3.6962
xs_box_Sh_int_lo = -18.0793

# for now the nlo SH xs are all based on M=600!

xs_box_nlo = 60.3702386403
xs_Sh_nlo = 9.16625205944
xs_box_Sh_int_nlo = -42.3503809411

xs_box_nlo_rw = 58.4478649041
xs_Sh_nlo_rw = 8.30809509287
xs_box_Sh_int_nlo_rw = -39.5673159149

# for kfactors no numbers produced yet for M!=600 so for now we are forcing the same kfactors as we get for M=600
xs_SH_nnlo = 2.006
xs_SH_lo = 0.7309
xs_SH_nlo = 1.5129402
xs_SH_nlo_rw = 1.5395984 

k_box_nlo = xs_box_nnlo/xs_box_nlo
k_sh_nlo = xs_Sh_nnlo/xs_Sh_nlo
k_box_sh_int_nlo = xs_box_Sh_int_nnlo/xs_box_Sh_int_nlo
k_sH_nlo = xs_SH_nnlo/xs_SH_nlo
k_sH_box_int_nlo = (k_box_nlo*k_sH_nlo)**.5
k_sH_sh_int_nlo = (k_sh_nlo*k_sH_nlo)**.5

k_box_nlo_rw = xs_box_nnlo/xs_box_nlo_rw
k_sh_nlo_rw = xs_Sh_nnlo/xs_Sh_nlo_rw
k_box_sh_int_nlo_rw = xs_box_Sh_int_nnlo/xs_box_Sh_int_nlo_rw
k_sH_nlo_rw = xs_SH_nnlo/xs_SH_nlo_rw
k_sH_box_int_nlo_rw = (k_box_nlo_rw*k_sH_nlo_rw)**.5
k_sH_sh_int_nlo_rw = (k_sh_nlo_rw*k_sH_nlo_rw)**.5

print('\n***************')
print('K-factors (LO->NNLO):')

k_box_lo = xs_box_nnlo/xs_box_lo
k_sh_lo = xs_Sh_nnlo/xs_Sh_lo
k_box_sh_int_lo = xs_box_Sh_int_nnlo/xs_box_Sh_int_lo
# for BM have specific k-factor for each mass point if specified
if 'xs_SH_nnlo' in bm and 'xs_SH_lo' in bm: 
    xs_SH_nnlo_ = bm['xs_SH_nnlo']
    xs_SH_lo_ = bm['xs_SH_lo']
else: 
    xs_SH_nnlo_ = xs_SH_nnlo
    xs_SH_lo_ = xs_SH_lo    
k_sH_lo = xs_SH_nnlo_/xs_SH_lo_
k_sH_box_int_lo = (k_box_lo*k_sH_lo)**.5
k_sH_sh_int_lo = (k_sh_lo*k_sH_lo)**.5


print('k_box =', k_box_lo)
print('k_sh =', k_sh_lo)
print('k_box_sh_int =', k_box_sh_int_lo)
print('k_sH =', k_sH_lo)
print('k_sH_box_int =', k_sH_box_int_lo)
print('k_sH_sh_int =', k_sH_sh_int_lo)

print('\nK-factors (NLO->NNLO):')

print('k_box =', k_box_nlo)
print('k_sh =', k_sh_nlo)
print('k_box_sh_int =', k_box_sh_int_nlo)
print('k_sH =', k_sH_nlo)
print('k_sH_box_int =', k_sH_box_int_nlo)
print('k_sH_sh_int =', k_sH_sh_int_nlo)

print('\nK-factors (NLOApprox->NNLO):')

print('k_box =', k_box_nlo_rw)
print('k_sh =', k_sh_nlo_rw)
print('k_box_sh_int =', k_box_sh_int_nlo_rw)
print('k_sH =', k_sH_nlo_rw)
print('k_sH_box_int =', k_sH_box_int_nlo_rw)
print('k_sH_sh_int =', k_sH_sh_int_nlo_rw)

print('***************\n')

def DrawHist(f, h, plot,wt_extra='1',sep_file=False):
  t = f.Get('ntuple')
  if sep_file:
      N = abs(t.GetEntries('wt_nom>0')-t.GetEntries('wt_nom<0'))
  else: N = t.GetEntries()
  h_name = str(h.GetName())
  if hh_mass_optimistic_str in plot: 
      var = '('+plot.split('(')[1]
      bins = '('+plot.split('(')[-1] 
  elif 'b1_pT+' in plot or 'b1_pT_smear+' in plot:  
      var = '('.join(plot.split('(')[0:2])
      bins = '('+plot.split('(')[2]    
  elif 'abs(' not in plot: 
      var = plot.split('(')[0]
      bins = '('+plot.split('(')[1]      
  else:   
      var = '('.join(plot.split('(')[0:2])
      bins = '('+plot.split('(')[2] 

  if args.reco_cuts:
    if 'b1_pT' in plot or 'b4_pT' in plot: t.Draw('%(var)s>>%(h_name)s%(bins)s'  % vars(),'(fabs(b4_eta_smear)<2.5&&fabs(b3_eta_smear)<2.5&&fabs(b2_eta_smear)<2.5&&fabs(b1_eta_smear)<2.5)*(wt_nom)*'+wt_extra, 'goff')
    else: t.Draw('%(var)s>>%(h_name)s%(bins)s'  % vars(),'((b1_pT_smear+b2_pT_smear+b3_pT_smear+b4_pT_smear)>280&&b4_pT_smear>30&&fabs(b4_eta_smear)<2.5&&fabs(b3_eta_smear)<2.5&&fabs(b2_eta_smear)<2.5&&fabs(b1_eta_smear)<2.5)*(wt_nom)*'+wt_extra, 'goff')
  else: t.Draw('%(var)s>>%(h_name)s%(bins)s'  % vars(),'(wt_nom)*'+wt_extra, 'goff')


  h = t.GetHistogram()
  h.Scale(1000./N) # units from pb to fb
  return h

#(hh_mass_smear_improved-hh_mass)/2 + hh_mass
#= (hh_mass_smear_improved+hh_mass)/2
hh_mass_optimistic_str = '(hh_mass_smear_improved_2+hh_mass)/2'

if bm_name == 'singlet_M600':
   plots = ['hh_mass_fine(200,250,1000)', 'hh_mass(75,250,1000)', 'hh_mass_smear(75,250,1000)', 'hh_mass_smear_improved(75,250,1000)']   
elif 'M260' in bm_name:
   plots = ['hh_mass_fine(200,200,500)', 'hh_mass(100,200,500)', 'hh_mass_smear(100,200,500)', 'hh_mass_smear_improved(100,200,500)','hh_mass_smear_improved_2(100,200,500)', 'hh_mass_smear_bbgg_improved(100,200,500)','%s(100,200,500)' % hh_mass_optimistic_str ]
elif 'M380' in bm_name:
   plots = ['hh_mass_fine(100,250,750)', 'hh_mass(100,250,750)', 'hh_mass_smear(100,250,750)', 'hh_mass_smear_improved(100,250,750)','hh_mass_smear_improved_2(100,250,750)', 'hh_mass_smear_bbgg_improved(100,250,750)','%s(100,250,750)' % hh_mass_optimistic_str]
elif 'M680' in bm_name:
   plots = ['hh_mass_fine(200,250,1200)', 'hh_mass(75,250,1200)', 'hh_mass_smear(75,250,1200)', 'hh_mass_smear_improved(75,250,1200)','hh_mass_smear_improved_2(75,250,1200)', 'hh_mass_smear_bbgg_improved(75,250,1200)','%s(100,250,1200)' % hh_mass_optimistic_str]   
elif 'M800' in bm_name or 'M870' in bm_name:
   plots = ['hh_mass_fine(200,250,1300)', 'hh_mass(75,250,1300)', 'hh_mass_smear(75,250,1300)', 'hh_mass_smear_improved(75,250,1300)','hh_mass_smear_improved_2(75,250,1300)', 'hh_mass_smear_bbgg_improved(75,250,1300)','%s(100,250,1300)' % hh_mass_optimistic_str] 
else:
   plots = ['hh_mass_fine(200,250,1000)', 'hh_mass(75,250,1000)', 'hh_mass_smear(75,250,1000)', 'hh_mass_smear_improved(75,250,1000)','hh_mass_smear_improved_2(75,250,1000)', 'hh_mass_smear_bbgg_improved(75,250,1000)','%s(100,250,1000)' % hh_mass_optimistic_str]

plots += ['hh_pT(75,0,300)', 'hh_pT_smear(75,0,300)', 'hh_pT_smear_improved(75,0,300)',
        'h1_pT(50,0,600)','h2_pT(50,0,600)','h1_pT_smear(50,0,600)','h2_pT_smear(50,0,600)', 'h1_pT_smear_improved(50,0,600)','h2_pT_smear_improved(50,0,600)',
        'h1_eta(100,-7,7)','h2_eta(100,-7,7)','h1_eta_smear(100,-7,7)','h2_eta_smear(100,-7,7)',
        'hh_dR(100,0,10)', 'hh_dR_smear(100,0,10)',
        'hh_dphi(100,0,7)', 'hh_dphi_smear(100,0,7)',
        'fabs(h1_eta-h2_eta)(100,0,7)', 'fabs(h1_eta_smear-h2_eta_smear)(100,0,7)',
        'hh_eta(100,-9,9)', 'hh_eta_smear(100,-9,9)',
        'b4_pT(50,0,300)', 'b4_pT_smear(50,0,300)',
        'b1_pT(50,0,1000)', 'b1_pT_smear(50,0,1000)',
        '(b1_pT+b2_pT+b3_pT+b4_pT)(50,0,1500)', '(b1_pT_smear+b2_pT_smear+b3_pT_smear+b4_pT_smear)(50,0,1500)',
        ]

plots = plots[:7]
#plots = plots[-4:]

for plot in plots:

  fineBins = False
  if '_fine' in plot:
    fineBins = True
    plot = plot.replace('_fine','')


  chhh10_wt = '(wt_box + 100*wt_schannel_h + 10*wt_box_and_schannel_h_i)' 
  box_wt = '(wt_box)'
  sh_wt = '(wt_schannel_h)'
  int_wt = '(wt_box_and_schannel_h_i)'

  h_sm = ROOT.TH1D()
  h_sm.SetName('sm')

  h_box = ROOT.TH1D()
  h_box.SetName('box')

  h_chhh10 = ROOT.TH1D()
  h_chhh10.SetName('chhh10')

  h_box_weighted = ROOT.TH1D()
  h_box_weighted.SetName('box_weighted')

  h_chhh10_weighted = ROOT.TH1D()
  h_chhh10_weighted.SetName('chhh10_weighted')

  h_sh_weighted = ROOT.TH1D()
  h_sh_weighted.SetName('sh_weighted')

  h_int_weighted = ROOT.TH1D()
  h_int_weighted.SetName('int_weighted')

  h_sm_lo = ROOT.TH1D() 
  h_sm_lo.SetName('sm_lo')
  h_box_lo = ROOT.TH1D()
  h_box_lo.SetName('box_lo')
  h_sh_lo = ROOT.TH1D() 
  h_sh_lo.SetName('sh_lo')
  h_int_lo = ROOT.TH1D()
  h_int_lo.SetName('int_lo')
  h_sH_box_lo = ROOT.TH1D()
  h_sH_box_lo.SetName('sH_box_lo')
  h_sH_sh_lo = ROOT.TH1D()
  h_sH_sh_lo.SetName('sH_sh_lo')

  h_sH_lo = ROOT.TH1D()
  h_sH_lo.SetName('sH_lo')
  h_sH_lo_v2 = ROOT.TH1D()
  h_sH_lo_v2.SetName('sH_lo_v2')
  h_sH = ROOT.TH1D()
  h_sH.SetName('sH')
  h_sH_weighted = ROOT.TH1D()
  h_sH_weighted.SetName('sH_weighted')
  h_sH_weighted_v2 = ROOT.TH1D()
  h_sH_weighted_v2.SetName('sH_weighted_v2')
  h_sH_box_weighted = ROOT.TH1D()
  h_sH_box_weighted.SetName('sH_box_weighted')
  h_sH_sh_weighted = ROOT.TH1D()
  h_sH_sh_weighted.SetName('sH_sh_weighted')
  h_sH_0p02_lo = ROOT.TH1D()
  h_sH_0p02_lo.SetName('sH_0p02_lo')
  h_sH_box_0p02_lo = ROOT.TH1D()
  h_sH_box_0p02_lo.SetName('sH_box_0p02_lo')
  h_sH_sh_0p02_lo = ROOT.TH1D()
  h_sH_sh_0p02_lo.SetName('sH_sh_0p02_lo')

  if bm_name == 'singlet_M600':
      h_sm = DrawHist(f1,h_sm,plot)
      h_box = DrawHist(f2,h_box,plot)
      h_chhh10 = DrawHist(f3,h_chhh10,plot)

  plot_mod = plot
  if 'hh_mass' in plot and 'hh_mass_smear_improved' not in plot: plot_mod = 'hh_mass(50,500,700)'

  if bm_name == 'singlet_M600':

      h_sH = DrawHist(f4,h_sH,plot_mod)
      h_sH.Scale(partial_width/5.)
    
      h_sH_weighted = DrawHist(f5,h_sH_weighted,plot_mod,'(wt_schannel_H_Mass_%(mass_str)s_RelWidth_%(rel_width_str)s)' % vars())
      h_sH_weighted.Scale(partial_width/12.)
    
      h_sH_box_weighted = DrawHist(f1,h_sH_box_weighted,plot,'(wt_box_and_schannel_H_i_Mass_%(mass_str)s_RelWidth_%(rel_width_str)s)' % vars())
    
      h_sH_sh_weighted = DrawHist(f1,h_sH_sh_weighted,plot,'(wt_schannel_H_and_schannel_h_i_Mass_%(mass_str)s_RelWidth_%(rel_width_str)s)' % vars())
    
      h_sh = h_chhh10.Clone()
      h_sh.Add(h_box,9)
      h_sh.Add(h_sm,-10)
      h_sh.Scale(1./90)
    
      h_int = h_sm.Clone()
      h_int.Scale(10./9.)
      h_int.Add(h_box, -11./10.)
      h_int.Add(h_chhh10,-1./90.)
    
    
      h_box_weighted = DrawHist(f1,h_box_weighted,plot, box_wt)
      h_chhh10_weighted = DrawHist(f1,h_chhh10_weighted,plot, chhh10_wt)
    
      h_sh_weighted = DrawHist(f1,h_sh_weighted,plot, sh_wt)
      h_int_weighted = DrawHist(f1,h_int_weighted,plot, int_wt)


  # get LO distributions

  h_sH_lo = DrawHist(f7,h_sH_lo,plot_mod, '(wt_schannel_H_Mass_%(mass_str)s_RelWidth_%(rel_width_str)s)' % vars())
  if bm_name == 'singlet_M600':
      h_sH_box_lo = DrawHist(f6,h_sH_box_lo,plot,'(wt_box_and_schannel_H_i_Mass_%(mass_str)s_RelWidth_%(rel_width_str)s)' % vars())
      h_sH_sh_lo = DrawHist(f6,h_sH_sh_lo,plot,'(wt_schannel_H_and_schannel_h_i_Mass_%(mass_str)s_RelWidth_%(rel_width_str)s)' % vars())

      h_sm_lo = DrawHist(f6, h_sm_lo, plot)
      h_box_lo = DrawHist(f6, h_box_lo, plot, box_wt)
      h_sh_lo = DrawHist(f6, h_sh_lo, plot, sh_wt)
      h_int_lo = DrawHist(f6, h_int_lo, plot, int_wt)

  # 2% width histograms:
  h_sH_0p02_lo = DrawHist(f7,h_sH_0p02_lo,plot_mod,'(wt_schannel_H_Mass_600_RelWidth_0p02)')
  if bm_name == 'singlet_M600':
      h_sH_box_0p02_lo = DrawHist(f6,h_sH_box_0p02_lo,plot,'(wt_box_and_schannel_H_i_Mass_600_RelWidth_0p02)')
      h_sH_sh_0p02_lo = DrawHist(f6,h_sH_sh_0p02_lo,plot,'(wt_schannel_H_and_schannel_h_i_Mass_600_RelWidth_0p02)')

  if 'hh_mass(' in plot: 
    x_title="m_{hh} (GeV)"
    y_title="d#sigma/dm_{hh} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_mass'

  elif 'hh_mass_smear(' in plot: 
    x_title="m_{hh} (GeV)"
    y_title="d#sigma/dm_{hh} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_mass_smear'

  elif 'hh_mass_smear_improved(' in plot:
    x_title="m_{hh} (GeV)"
    y_title="d#sigma/dm_{hh} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_mass_smear_improved'

  elif 'hh_mass_smear_improved_2(' in plot:
    x_title="m_{hh} (GeV)"
    y_title="d#sigma/dm_{hh} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_mass_smear_improved_paired'    
    
  elif 'hh_mass_smear_bbgg_improved(' in plot:
    x_title="m_{hh} (GeV)"
    y_title="d#sigma/dm_{hh} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_mass_smear_bbgg_improved'

  elif hh_mass_optimistic_str in plot:
    x_title="m_{hh} (GeV)"
    y_title="d#sigma/dm_{hh} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_mass_smear_improved_optimistic'    

  elif 'hh_pT(' in plot:
    x_title="p_{T}^{hh} (GeV)"
    y_title="d#sigma/dp_{T}^{hh} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_pT'

  elif 'hh_pT_smear(' in plot:
    x_title="p_{T}^{hh} (GeV)"
    y_title="d#sigma/dp_{T}^{hh} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_pT_smear'

  elif 'hh_pT_smear_improved(' in plot:
    x_title="p_{T}^{hh} (GeV)"
    y_title="d#sigma/dp_{T}^{hh} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_pT_smear_improved'    

  elif 'h1_pT(' in plot:
    x_title="p_{T}^{h_{1}} (GeV)"
    y_title="d#sigma/dp_{T}^{h_{1}} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_h1_pT'

  elif 'h1_pT_smear(' in plot:
    x_title="p_{T}^{h_{1}} (GeV)"
    y_title="d#sigma/dp_{T}^{h_{1}} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_h1_pT_smear'

  elif 'h1_pT_smear_improved(' in plot:
    x_title="p_{T}^{h_{1}} (GeV)"
    y_title="d#sigma/dp_{T}^{h_{1}} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_h1_pT_smear_improved'    

  elif 'h2_pT(' in plot:
    x_title="p_{T}^{h_{2}} (GeV)"
    y_title="d#sigma/dp_{T}^{h_{2}} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_h2_pT'

  elif 'h2_pT_smear(' in plot:
    x_title="p_{T}^{h_{2}} (GeV)"
    y_title="d#sigma/dp_{T}^{h_{2}} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_h2_pT_smear'

  elif 'h2_pT_smear_improved(' in plot:
    x_title="p_{T}^{h_{2}} (GeV)"
    y_title="d#sigma/dp_{T}^{h_{2}} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_h2_pT_smear_improved' 


  elif 'b4_pT(' in plot:
    x_title="p_{T}^{b_{4}} (GeV)"
    y_title="d#sigma/dp_{T}^{b_{4}} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_b4_pT' 

  elif 'b4_pT_smear(' in plot:
    x_title="p_{T}^{b_{4}} (GeV)"
    y_title="d#sigma/dp_{T}^{b_{4}} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_b4_pT_smear'  

  elif 'b1_pT(' in plot:
    x_title="p_{T}^{b_{1}} (GeV)"
    y_title="d#sigma/dp_{T}^{b_{1}} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_b1_pT' 

  elif 'b1_pT_smear(' in plot:
    x_title="p_{T}^{b_{1}} (GeV)"
    y_title="d#sigma/dp_{T}^{b_{1}} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_b1_pT_smear'    

  elif 'h1_eta(' in plot:
    x_title="#eta^{h_{1}}"
    y_title="d#sigma/d#eta^{h_{1}} (fb)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_h1_eta'

  elif 'h1_eta_smear(' in plot:
    x_title="#eta^{h_{1}}"
    y_title="d#sigma/d#eta^{h_{1}} (fb)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_h1_eta_smear' 

  elif 'h2_eta(' in plot:
    x_title="#eta^{h_{2}}"
    y_title="d#sigma/d#eta^{h_{2}} (fb)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_h2_eta'

  elif 'h2_eta_smear(' in plot:
    x_title="#eta^{h_{2}}"
    y_title="d#sigma/d#eta^{h_{2}} (fb)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_h2_eta_smear'  

  elif 'hh_eta(' in plot:
    x_title="#eta^{hh}"
    y_title="d#sigma/d#eta^{hh} (fb)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_eta'

  elif 'hh_eta_smear(' in plot:
    x_title="#eta^{hh}"
    y_title="d#sigma/d#eta^{hh} (fb)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_eta_smear'  

  elif 'hh_dR(' in plot:
    x_title="#Delta R(h_{1},h_{2})"
    y_title="d#sigma/d#Delta R(h_{1},h_{2}) (fb)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_dR'

  elif 'hh_dR_smear(' in plot:
    x_title="#Delta R(h_{1},h_{2})"
    y_title="d#sigma/d#Delta R(h_{1},h_{2}) (fb)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_dR_smear' 

  elif 'hh_dphi(' in plot:
    x_title="#Delta#phi(h_{1},h_{2})"
    y_title="d#sigma/d#Delta#phi(h_{1},h_{2}) (fb)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_dphi'

  elif 'hh_dphi_smear(' in plot:
    x_title="#Delta#phi(h_{1},h_{2})"
    y_title="d#sigma/d#Delta#phi(h_{1},h_{2}) (fb)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_dphi_smear' 

  elif 'fabs(h1_eta-h2_eta)(' in plot:
    x_title="#Delta#eta(h_{1},h_{2})"
    y_title="d#sigma/d#Delta#eta(h_{1},h_{2}) (fb)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_deta' 

  elif 'fabs(h1_eta_smear-h2_eta_smear)(' in plot:
    x_title="#Delta#eta(h_{1},h_{2})"
    y_title="d#sigma/d#Delta#eta(h_{1},h_{2}) (fb)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_deta_smear'

  elif '(b1_pT+b2_pT+b3_pT+b4_pT)(' in plot:
    x_title="H_{T} (GeV)"
    y_title="d#sigma/dH_{T} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_HT'

  elif '(b1_pT_smear+b2_pT_smear+b3_pT_smear+b4_pT_smear)(' in plot:
    x_title="H_{T} (GeV)"
    y_title="d#sigma/dH_{T} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_HT_smear'  

  else: 
      raise Exception('invalid plot: %s' % plot) 

  if fineBins: plot_name = plot_name+'_fineBins'     


  if args.reco_cuts: plot_name = plot_name.replace('plots_NLO', 'plots_NLO_recocuts')

  if bm_name == 'singlet_M600_new':
      # plot comparing inteference for different widths 

      h_box_SH_lo_0p01 = ROOT.TH1D()
      h_box_SH_lo_0p02 = ROOT.TH1D()
      h_box_SH_lo_0p05 = ROOT.TH1D()
      h_box_SH_lo_0p10 = ROOT.TH1D()

      h_box_SH_lo_0p01.SetName('box_SH_lo_0p01')
      h_box_SH_lo_0p02.SetName('box_SH_lo_0p02')
      h_box_SH_lo_0p05.SetName('box_SH_lo_0p05')
      h_box_SH_lo_0p10.SetName('box_SH_lo_0p10')

      h_box_SH_lo_0p01 = DrawHist(f6,h_box_SH_lo_0p01,plot,'(wt_box_and_schannel_H_i_Mass_600_RelWidth_0p01)' % vars())
      h_box_SH_lo_0p02 = DrawHist(f6,h_box_SH_lo_0p02,plot,'(wt_box_and_schannel_H_i_Mass_600_RelWidth_0p02)' % vars())
      h_box_SH_lo_0p05 = DrawHist(f6,h_box_SH_lo_0p05,plot,'(wt_box_and_schannel_H_i_Mass_600_RelWidth_0p05)' % vars())
      h_box_SH_lo_0p10 = DrawHist(f6,h_box_SH_lo_0p10,plot,'(wt_box_and_schannel_H_i_Mass_600_RelWidth_0p1)' % vars())


      plotting.CompareHists(hists=[h_box_SH_lo_0p01.Clone(), h_box_SH_lo_0p02.Clone(), h_box_SH_lo_0p05.Clone(), h_box_SH_lo_0p10.Clone()],
             legend_titles=['#Gamma_{H}=6 GeV', '#Gamma_{H}=12 GeV', '#Gamma_{H}=30 GeV', '#Gamma_{H}=60 GeV'],
             title="S_{H}-#Box contribution, LO+PS",
             ratio=True,
             log_y=False,
             log_x=False,
             ratio_range="-0.2,1.2",
             custom_x_range=False,
             x_axis_max=1000,
             x_axis_min=250,
             custom_y_range=False,
             y_axis_max=4000,
             y_axis_min=0,
             x_title=x_title,
             y_title=y_title,
             extra_pad=0,
             norm_hists=norm_hists,
             plot_name=plot_name.replace('NLO_Validation','LO_WidthComp_box_SH'),
             label="m_{H} = 600 GeV",
             norm_bins=True,
             wideLeg=True) 

  if bm_name == 'singlet_M600':

      plotting.CompareHists(
                   hists=[h_sh_lo.Clone(), h_box_lo.Clone(), h_int_lo.Clone(), h_sH_0p02_lo.Clone(), h_sH_box_0p02_lo.Clone(), h_sH_sh_0p02_lo.Clone()],
                   legend_titles=['S_{h}', '#Box', 'S_{h}-#Box', 'S_{H}', 'S_{H}-#Box', 'S_{H}-S_{h}'],
                   title="LO+PS" % vars(),
                   ratio=False,
                   log_y=False,
                   log_x=False,
                   ratio_range="0.,2.0",
                   custom_x_range=True,
                   x_axis_max=1000,
                   x_axis_min=250,
                   custom_y_range=False,
                   y_axis_max=4000,
                   y_axis_min=0,
                   x_title=x_title,
                   y_title=y_title,
                   extra_pad=0,
                   norm_hists=False,
                   plot_name=plot_name.replace('NLO_Validation','LO_contributions'),
                   label="m_{H} = 600 GeV,  #Gamma_{H} = 12 GeV",
                   norm_bins=True)  

    
      plotting.CompareHists(hists=[h_box.Clone(), h_box_weighted.Clone()],
                   legend_titles=['Generated','Reweighted'],
                   title="",
                   ratio=True,
                   log_y=False,
                   log_x=False,
                   ratio_range="0.5,1.5",
                   custom_x_range=False,
                   x_axis_max=1000,
                   x_axis_min=250,
                   custom_y_range=False,
                   y_axis_max=4000,
                   y_axis_min=0,
                   x_title=x_title,
                   y_title=y_title,
                   extra_pad=0,
                   norm_hists=norm_hists,
                   plot_name=plot_name+'_box',
                   label='',
                   norm_bins=True,
                   IncErrors=True)

      plotting.CompareHists(hists=[h_chhh10.Clone(), h_chhh10_weighted.Clone()],
                   legend_titles=['Generated','Reweighted'],
                   title="",
                   ratio=True,
                   log_y=False,
                   log_x=False,
                   ratio_range="0.5,1.5",
                   custom_x_range=False,
                   x_axis_max=1000,
                   x_axis_min=250,
                   custom_y_range=False,
                   y_axis_max=4000,
                   y_axis_min=0,
                   x_title=x_title,
                   y_title=y_title,
                   extra_pad=0,
                   norm_hists=norm_hists,
                   plot_name=plot_name+'_chhh10',
                   label='',
                   norm_bins=True,
                   IncErrors=True)

      plotting.CompareHists(hists=[h_sh.Clone(), h_sh_weighted.Clone()],
                   legend_titles=['Generated','Reweighted'],
                   title="",
                   ratio=True,
                   log_y=False,
                   log_x=False,
                   ratio_range="0.5,1.5",
                   custom_x_range=False,
                   x_axis_max=1000,
                   x_axis_min=250,
                   custom_y_range=False,
                   y_axis_max=4000,
                   y_axis_min=0,
                   x_title=x_title,
                   y_title=y_title,
                   extra_pad=0,
                   norm_hists=norm_hists,
                   plot_name=plot_name+'_Sh',
                   label='',
                   norm_bins=True,
                   IncErrors=True)

      plotting.CompareHists(hists=[h_int.Clone(), h_int_weighted.Clone()],
                   legend_titles=['Generated','Reweighted'],
                   title="",
                   ratio=True,
                   log_y=False,
                   log_x=False,
                   ratio_range="0.5,1.5",
                   custom_x_range=False,
                   x_axis_max=1000,
                   x_axis_min=250,
                   custom_y_range=False,
                   y_axis_max=4000,
                   y_axis_min=0,
                   x_title=x_title,
                   y_title=y_title,
                   extra_pad=0,
                   norm_hists=norm_hists,
                   plot_name=plot_name+'_box_Sh_int',
                   label='',
                   norm_bins=True,
                   IncErrors=True)


    # make plots comparing LO to NLO to NLO-approx+PS

    #kaps_lab = '#kappa_{t}^{h} = #kappa_{t}^{H} = #kappa_{\lambda_{hhh}} = #kappa_{\lambda_{Hhh}} = 1, m_{H} = 600 GeV, #Gamma_{H} = 5 GeV'
  kaps_lab = 'm_{H} = 600 GeV, #Gamma_{H} = 5 GeV'

  for x in ['','_inc_kfactors']:

    if bm_name == 'singlet_M600':

        plotting.CompareHists(hists=[h_box.Clone(), h_box_weighted.Clone(), h_box_lo.Clone()],
                   legend_titles=['NLO+PS','NLO-approx+PS','LO+PS'] + (['Inc. K-factor scaling'] if x == '_inc_kfactors' else []),
                   scale_factors = [k_box_nlo, k_box_nlo_rw, k_box_lo] if x == '_inc_kfactors' else None,
                   title="#Box contribution" % vars(),
                   ratio=True,
                   log_y=False,
                   log_x=False,
                   ratio_range="0.,2.0",
                   custom_x_range=False,
                   x_axis_max=1000,
                   x_axis_min=250,
                   custom_y_range=False,
                   y_axis_max=4000,
                   y_axis_min=0,
                   x_title=x_title,
                   y_title=y_title,
                   extra_pad=0,
                   norm_hists=norm_hists,
                   plot_name=plot_name.replace('NLO_Validation','NLO_vsLO'+x)+'_box',
                   label=kaps_lab,
                   norm_bins=True,
                   IncErrors=True,
                   wideLeg=True)

        plotting.CompareHists(hists=[h_sh.Clone(), h_sh_weighted.Clone(), h_sh_lo.Clone()],
                     legend_titles=['NLO+PS','NLO-approx+PS','LO+PS'] + (['Inc. K-factor scaling'] if x == '_inc_kfactors' else []),
                     scale_factors = [k_sh_nlo, k_sh_nlo_rw, k_sh_lo] if x == '_inc_kfactors' else None,
                     title="S_{h} contribution" % vars(),
                     ratio=True,
                     log_y=False,
                     log_x=False,
                     ratio_range="0.,2.",
                     custom_x_range=False,
                     x_axis_max=1000,
                     x_axis_min=250,
                     custom_y_range=False,
                     y_axis_max=4000,
                     y_axis_min=0,
                     x_title=x_title,
                     y_title=y_title,
                     extra_pad=0,
                     norm_hists=norm_hists,
                     plot_name=plot_name.replace('NLO_Validation','NLO_vsLO'+x)+'_Sh',
                     label=kaps_lab,
                     norm_bins=True,
                     IncErrors=True,
                     wideLeg=True)

        plotting.CompareHists(hists=[h_int.Clone(), h_int_weighted.Clone(), h_int_lo.Clone()],
                     legend_titles=['NLO+PS','NLO-approx+PS','LO+PS'] + (['Inc. K-factor scaling'] if x == '_inc_kfactors' else []),
                     scale_factors = [k_box_sh_int_nlo, k_box_sh_int_nlo_rw, k_box_sh_int_lo] if x == '_inc_kfactors' else None,
                     title="S_{h}-#Box contribution" % vars(),
                     ratio=True,
                     log_y=False,
                     log_x=False,
                     ratio_range="0.,2.",
                     custom_x_range=False,
                     x_axis_max=1000,
                     x_axis_min=250,
                     custom_y_range=False,
                     y_axis_max=4000,
                     y_axis_min=0,
                     x_title=x_title,
                     y_title=y_title,
                     extra_pad=0,
                     norm_hists=norm_hists,
                     plot_name=plot_name.replace('NLO_Validation','NLO_vsLO'+x)+'_box_Sh_int',
                     label=kaps_lab,
                     norm_bins=True,
                     IncErrors=True,
                     lowerLeg=True,
                     wideLeg=True)

        # do same for H in s-channel

        plotting.CompareHists(hists=[h_sH.Clone(), h_sH_weighted.Clone(), h_sH_lo.Clone()],
                     legend_titles=['NLO+PS','NLO-approx+PS','LO+PS'] + (['Inc. K-factor scaling'] if x == '_inc_kfactors' else []),
                     scale_factors = [k_sH_nlo, k_sH_nlo_rw, k_sH_lo] if x == '_inc_kfactors' else None,
                     title="S_{H} contribution" % vars(),
                     ratio=True,
                     log_y=False,
                     log_x=False,
                     ratio_range="0.,2.",
                     custom_x_range=False,
                     x_axis_max=1000,
                     x_axis_min=250,
                     custom_y_range=False,
                     y_axis_max=4000,
                     y_axis_min=0,
                     x_title=x_title,
                     y_title=y_title,
                     extra_pad=0,
                     norm_hists=norm_hists,
                     plot_name=plot_name.replace('NLO_Validation','NLO_vsLO'+x)+'_SP',
                     label=kaps_lab,
                     norm_bins=True,
                     IncErrors=True,
                     wideLeg=True)

       # now makes plots of inteferences without the NLO exact (as it does not exist)


        ##print h_sH_lo.Integral(-1,-1), h_sH_box_weighted.Integral(-1,-1), h_sH_box_lo.Integral(-1,-1)
        ##exit() 

        plotting.CompareHists(hists=[h_sH_box_weighted.Clone(), h_sH_box_lo.Clone()],
                     legend_titles=['NLO-approx+PS','LO+PS'] + (['Inc. K-factor scaling'] if x == '_inc_kfactors' else []),
                     scale_factors = [k_sH_box_int_nlo_rw, k_sH_box_int_lo] if x == '_inc_kfactors' else None,
                     title="S_{H}-#Box contribution" % vars(),
                     ratio=True,
                     log_y=False,
                     log_x=False,
                     ratio_range="0.,2.",
                     custom_x_range=False,
                     x_axis_max=1000,
                     x_axis_min=250,
                     custom_y_range=False,
                     y_axis_max=4000,
                     y_axis_min=0,
                     x_title=x_title,
                     y_title=y_title,
                     extra_pad=0,
                     norm_hists=norm_hists,
                     plot_name=plot_name.replace('NLO_Validation','NLO_vsLO'+x)+'_box_SP',
                     label=kaps_lab,
                     norm_bins=True,
                     IncErrors=True,
                     skipCols=1,
                     wideLeg=True)

        plotting.CompareHists(hists=[h_sH_sh_weighted.Clone(), h_sH_sh_lo.Clone()],
                     legend_titles= ['NLO-approx+PS','LO+PS'] + (['Inc. K-factor scaling'] if x == '_inc_kfactors' else []),
                     scale_factors = [k_sH_sh_int_nlo_rw, k_sH_sh_int_lo] if x == '_inc_kfactors' else None,
                     title="S_{H}-S_{h} contribution" % vars(),
                     ratio=True,
                     log_y=False,
                     log_x=False,
                     ratio_range="0.,2.",
                     custom_x_range=False,
                     x_axis_max=1000,
                     x_axis_min=250,
                     custom_y_range=False,
                     y_axis_max=4000,
                     y_axis_min=0,
                     x_title=x_title,
                     y_title=y_title,
                     extra_pad=0,
                     norm_hists=norm_hists,
                     plot_name=plot_name.replace('NLO_Validation','NLO_vsLO'+x)+'_SP_Sh',
                     label=kaps_lab,
                     norm_bins=True,
                     IncErrors=True,
                     skipCols=1,
                     lowerLeg=True,
                     wideLeg=True)

    # make plots comparing benchmarks

    box_SF                         = kappa_h_t**4
    schannel_H_SF                  = kappa_H_t**2*kappa_H_lam**2
    schannel_h_SF                  = kappa_h_t**2*kappa_h_lam**2
    box_and_schannel_h_i_SF        = kappa_h_t**3*kappa_h_lam
    box_and_schannel_H_i_SF        = kappa_h_t**2*kappa_H_t*kappa_H_lam
    schannel_H_and_schannel_h_i_SF = kappa_H_t*kappa_H_lam*kappa_h_t*kappa_h_lam

    # make sH histograms again so that they do not include the zoomed binning for mass, and so that 5 GeV width samples are used for both NLO and LO
    if bm_name == 'singlet_M600': h_sH_weighted_v2 = DrawHist(f4,h_sH_weighted_v2,plot)
    h_sH_weighted_v2.Scale(partial_width/5.)
    h_sH_lo_v2 = DrawHist(f7,h_sH_lo_v2,plot, '(wt_schannel_H_Mass_%(mass_str)s_RelWidth_%(rel_width_str)s)' % vars())


    wt_BM = '(wt_box*%(box_SF)g + wt_schannel_h*%(schannel_h_SF)g + wt_box_and_schannel_h_i*%(box_and_schannel_h_i_SF)g + wt_box_and_schannel_H_i_Mass_%(mass_str)s_RelWidth_%(rel_width_str)s*%(box_and_schannel_H_i_SF)g + wt_schannel_H_and_schannel_h_i_Mass_%(mass_str)s_RelWidth_%(rel_width_str)s*%(schannel_H_and_schannel_h_i_SF)g)' % vars()
    #wt_BM = '(wt_box*%(box_SF)g + wt_schannel_h*%(schannel_h_SF)g + wt_box_and_schannel_h_i*%(box_and_schannel_h_i_SF)g + wt_box_and_schannel_H_i_Mass_%(mass_str)s_RelWidth_%(rel_width_str)s*%(box_and_schannel_H_i_SF)g + wt_schannel_H_and_schannel_h_i_Mass_%(mass_str)s_RelWidth_%(rel_width_str)s*%(schannel_H_and_schannel_h_i_SF)g + wt_schannel_H_Mass_%(mass_str)s_RelWidth_%(rel_width_str)s*%(schannel_H_SF)s)' % vars() # including s-channel in plot as well



    wt_BM_kfacts_nlo_rw = '(wt_box*%(box_SF)g*%(k_box_nlo_rw)g + wt_schannel_h*%(schannel_h_SF)g*%(k_sh_nlo_rw)g + wt_box_and_schannel_h_i*%(box_and_schannel_h_i_SF)g*%(k_box_sh_int_nlo_rw)g + wt_box_and_schannel_H_i_Mass_%(mass_str)s_RelWidth_%(rel_width_str)s*%(box_and_schannel_H_i_SF)g*%(k_sH_box_int_nlo_rw)g + wt_schannel_H_and_schannel_h_i_Mass_%(mass_str)s_RelWidth_%(rel_width_str)s*%(schannel_H_and_schannel_h_i_SF)g*%(k_sH_sh_int_nlo_rw)g)' % vars()

    wt_BM_kfacts_lo = '(wt_box*%(box_SF)g*%(k_box_lo)g + wt_schannel_h*%(schannel_h_SF)g*%(k_sh_lo)g + wt_box_and_schannel_h_i*%(box_and_schannel_h_i_SF)g*%(k_box_sh_int_lo)g + wt_box_and_schannel_H_i_Mass_%(mass_str)s_RelWidth_%(rel_width_str)s*%(box_and_schannel_H_i_SF)g*%(k_sH_box_int_lo)g + wt_schannel_H_and_schannel_h_i_Mass_%(mass_str)s_RelWidth_%(rel_width_str)s*%(schannel_H_and_schannel_h_i_SF)g*%(k_sH_sh_int_lo)g)' % vars()

    wt_BM_nonres_lo = '(wt_box*%(box_SF)g + wt_schannel_h*%(schannel_h_SF)g + wt_box_and_schannel_h_i*%(box_and_schannel_h_i_SF)g)' % vars()
    wt_BM_nonres_kfacts_lo = '(wt_box*%(box_SF)g*%(k_box_lo)g + wt_schannel_h*%(schannel_h_SF)g*%(k_sh_lo)g + wt_box_and_schannel_h_i*%(box_and_schannel_h_i_SF)g*%(k_box_sh_int_lo)g )' % vars()

    wt_BM_kapt1_nonres_lo = '(wt_box + wt_schannel_h*%(kappa_h_lam)g*%(kappa_h_lam)g + wt_box_and_schannel_h_i*%(kappa_h_lam)g)' % vars()
    wt_BM_kapt1_nonres_kfacts_lo = '(wt_box*%(k_box_lo)g + wt_schannel_h*%(kappa_h_lam)g*%(kappa_h_lam)g*%(k_sh_lo)g + wt_box_and_schannel_h_i*%(kappa_h_lam)g*%(k_box_sh_int_lo)g )' % vars()

    wt_BM_noint_kfacts_lo = '(wt_box*%(box_SF)g*%(k_box_lo)g + wt_schannel_h*%(schannel_h_SF)g*%(k_sh_lo)g + wt_box_and_schannel_h_i*%(box_and_schannel_h_i_SF)g*%(k_box_sh_int_lo)g)' % vars()

    wt_BM_noint = '(wt_box*%(box_SF)g + wt_schannel_h*%(schannel_h_SF)g + wt_box_and_schannel_h_i*%(box_and_schannel_h_i_SF)g)' % vars()

    wt_BM_noint_kfacts_nlo_rw = '(wt_box*%(box_SF)g*%(k_box_nlo_rw)g + wt_schannel_h*%(schannel_h_SF)g*%(k_sh_nlo_rw)g + wt_box_and_schannel_h_i*%(box_and_schannel_h_i_SF)g*%(k_box_sh_int_nlo_rw)g)' % vars()

    if bm_name == 'singlet_M600':

        h_BM_weighted = ROOT.TH1D()
        h_BM_weighted.SetName('BM_weighted')
        h_BM_weighted = DrawHist(f1,h_BM_weighted,plot, wt_BM)
        h_BM_weighted.Add(h_sH_weighted_v2,schannel_H_SF)

        h_BM_weighted_kfacts = ROOT.TH1D()
        h_BM_weighted_kfacts.SetName('BM_weighted_kfacts')
        h_BM_weighted_kfacts = DrawHist(f1,h_BM_weighted_kfacts,plot, wt_BM_kfacts_nlo_rw)
        h_BM_weighted_kfacts.Add(h_sH_weighted_v2,schannel_H_SF*k_sH_nlo_rw)

        h_BM_weighted_noint = ROOT.TH1D()
        h_BM_weighted_noint.SetName('BM_weighted')
        h_BM_weighted_noint = DrawHist(f1,h_BM_weighted,plot, wt_BM_noint)
        h_BM_weighted_noint.Add(h_sH_weighted_v2,schannel_H_SF)

        h_BM_weighted_noint_kfacts = ROOT.TH1D()
        h_BM_weighted_noint_kfacts.SetName('BM_weighted_kfacts')
        h_BM_weighted_noint_kfacts = DrawHist(f1,h_BM_weighted_kfacts,plot, wt_BM_noint_kfacts_nlo_rw)
        h_BM_weighted_noint_kfacts.Add(h_sH_weighted_v2,schannel_H_SF*k_sH_nlo_rw)


    h_BM_lo = ROOT.TH1D()
    h_BM_lo.SetName('BM_lo')
    h_BM_lo = DrawHist(f6,h_BM_lo,plot, wt_BM)
    h_BM_lo.Add(h_sH_lo_v2,schannel_H_SF)

    h_sH_before_lo = ROOT.TH1D()
    h_sH_before_lo.SetName('sH_before_lo')

    h_sH_before_lo = DrawHist(f7,h_sH_before_lo,plot, '1')
    h_before_lo = ROOT.TH1D()
    h_before_lo.SetName('BM_before_lo')
    h_before_lo = DrawHist(f6,h_before_lo,plot, '1')
    h_before_lo.Add(h_sH_before_lo)

    if bm_name == 'singlet_M600':
        # get seperatly generated hist here:

        h_BM_sep_generated = ROOT.TH1D()
        h_BM_sep_generated.SetName('BM_sep_lo')

        sep_files = [
            (schannel_H_SF,'outputs_4b_Feb13/output_mg_pythia_width_5GeV.root'),
            (box_SF,'outputs_4b_Feb13/output_mg_pythia_box.root'),
            (schannel_h_SF,'outputs_4b_Feb13/output_mg_pythia_sh.root'),
            (box_and_schannel_H_i_SF*2.**1.5,'outputs_4b_Feb13/output_mg_pythia_width_5GeV_box_SH.root'),
            (box_and_schannel_h_i_SF,'outputs_4b_Feb13/output_mg_pythia_box_sh.root'),
            (schannel_H_and_schannel_h_i_SF*2.,'outputs_4b_Feb13/output_mg_pythia_width_5GeV_Sh_SH.root'),
          ]
        count = 0
        for sep_file in sep_files:
            f_sep = ROOT.TFile(sep_file[1])
            if count == 0:
                h_BM_sep_generated = DrawHist(f_sep,h_BM_sep_generated,plot,'%g' % sep_file[0], True)
            else:
                h_temp = h_BM_sep_generated.Clone()
                h_temp = DrawHist(f_sep,h_temp,plot,'%g' % sep_file[0], True)
                h_BM_sep_generated.Add(h_temp)
            count+=1


        h_BM_lo_generated = ROOT.TH1D()
        h_BM_lo_generated.SetName('BM_generated')
        h_BM_lo_generated = DrawHist(f8,h_BM_lo_generated,plot)

    h_BM_lo_noint = ROOT.TH1D()
    h_BM_lo_noint.SetName('BM_lo_noint')
    h_BM_lo_noint = DrawHist(f6,h_BM_lo_noint,plot, wt_BM_noint)
    h_BM_lo_noint.Add(h_sH_lo_v2,schannel_H_SF)

    #wt_BM_nonres_lo
    h_BM_nonres_lo = ROOT.TH1D()
    h_BM_nonres_lo.SetName('BM_nonres_lo')
    h_BM_nonres_lo = DrawHist(f6,h_BM_nonres_lo,'hh_mass(100,200,800)', wt_BM_nonres_lo)

    h_BM_nonres_kfacts_lo = ROOT.TH1D()
    h_BM_nonres_kfacts_lo.SetName('BM_nonres_kfacts_lo')
    h_BM_nonres_kfacts_lo = DrawHist(f6,h_BM_nonres_kfacts_lo,'hh_mass(100,200,800)', wt_BM_nonres_kfacts_lo)

    h_BM_kapt1_nonres_lo = ROOT.TH1D()
    h_BM_kapt1_nonres_lo.SetName('BM_nonres_lo')
    h_BM_kapt1_nonres_lo = DrawHist(f6,h_BM_kapt1_nonres_lo,'hh_mass(100,200,800)', wt_BM_kapt1_nonres_lo)

    h_BM_kapt1_nonres_kfacts_lo = ROOT.TH1D()
    h_BM_kapt1_nonres_kfacts_lo.SetName('BM_nonres_kfacts_lo')
    h_BM_kapt1_nonres_kfacts_lo = DrawHist(f6,h_BM_kapt1_nonres_kfacts_lo,'hh_mass(100,200,800)', wt_BM_kapt1_nonres_kfacts_lo)

    h_BM_lo_kfacts = ROOT.TH1D()
    h_BM_lo_kfacts.SetName('BM_lo_kfacts')
    h_BM_lo_kfacts = DrawHist(f6,h_BM_lo_kfacts,plot, wt_BM_kfacts_lo)
    h_BM_lo_kfacts.Add(h_sH_lo_v2,schannel_H_SF*k_sH_lo)

    h_BM_lo_noint_kfacts = ROOT.TH1D()
    h_BM_lo_noint_kfacts.SetName('BM_lo_kfacts')
    h_BM_lo_noint_kfacts = DrawHist(f6,h_BM_lo_kfacts,plot, wt_BM_noint_kfacts_lo)
    h_BM_lo_noint_kfacts.Add(h_sH_lo_v2,schannel_H_SF*k_sH_lo)    

    sH_lo = h_sH_lo_v2.Clone()
    sH_lo.Scale(schannel_H_SF)
    sH_lo_kfacts = h_sH_lo_v2.Clone()
    sH_lo_kfacts.Scale(schannel_H_SF*k_sH_lo)

    wt_SM = '(wt_box+wt_schannel_h+wt_box_and_schannel_h_i)'
    wt_SM_kfacts_lo = '(wt_box*%(k_box_lo)g + wt_schannel_h*%(k_sh_lo)g + wt_box_and_schannel_h_i*%(k_box_sh_int_lo)g)' % vars()

    h_SM_lo = ROOT.TH1D()
    h_SM_lo.SetName('SM')
    h_SM_lo = DrawHist(f6,h_SM_lo,plot, wt_SM)

    h_SM_kfacts_lo = ROOT.TH1D()
    h_SM_kfacts_lo.SetName('SM_kfacts')
    h_SM_kfacts_lo = DrawHist(f6,h_SM_kfacts_lo,plot, wt_SM_kfacts_lo)

    h_SM_newbins_lo = ROOT.TH1D()
    h_SM_newbins_lo.SetName('SM_newbins')
    h_SM_newbins_lo = DrawHist(f6,h_SM_newbins_lo,'hh_mass(100,200,800)', wt_SM)  

    h_SM_newbins_kfacts_lo = ROOT.TH1D()
    h_SM_newbins_kfacts_lo.SetName('SM_newbins_kfacts')
    h_SM_newbins_kfacts_lo = DrawHist(f6,h_SM_newbins_kfacts_lo,'hh_mass(100,200,800)', wt_SM_kfacts_lo)    

    h_BM_approx_lo = h_SM_lo.Clone()
    h_BM_approx_lo.Add(sH_lo)

    h_BM_approx_kfacts_lo = h_SM_kfacts_lo.Clone()
    h_BM_approx_kfacts_lo.Add(sH_lo_kfacts)


    if bm_name == 'singlet_M600':
        plotting.CompareHists(hists=[h_BM_lo_generated.Clone(), h_BM_sep_generated.Clone(), h_BM_lo.Clone()],
                   legend_titles=['Directly generated', 'Separately generated', 'Reweighted'],
                   title="LO+PS" % vars(),
                   ratio=True,
                   log_y=False,
                   log_x=False,
                   ratio_range="0.,2.0",
                   custom_x_range=True,
                   x_axis_max=1000,
                   x_axis_min=250,
                   custom_y_range=False,
                   y_axis_max=4000,
                   y_axis_min=0,
                   x_title=x_title,
                   y_title=y_title,
                   extra_pad=0,
                   norm_hists=False,
                   plot_name=plot_name.replace('NLO_Validation','ReweightValidation_old'),
                   label=bm['label'],
                   norm_bins=True,
                   IncErrors=True,
                   wideLeg=True)


        plotting.CompareHists(hists=[h_BM_lo_generated.Clone(), h_BM_lo.Clone(),h_before_lo.Clone()],
                   legend_titles=['Directly generated', 'Reweighted', 'Before reweighting'],
                   title="LO+PS" % vars(),
                   ratio=True,
                   log_y=False,
                   log_x=False,
                   ratio_range="0.,2.0",
                   custom_x_range=True,
                   x_axis_max=1000,
                   x_axis_min=250,
                   custom_y_range=False,
                   y_axis_max=4000,
                   y_axis_min=0,
                   x_title=x_title,
                   y_title=y_title,
                   extra_pad=0,
                   norm_hists=False,
                   plot_name=plot_name.replace('NLO_Validation','ReweightValidation'),
                   label=bm['label'],
                   norm_bins=True,
                   IncErrors=True,
                   wideLeg=True)


        plotting.CompareHists(hists=[h_BM_lo_generated.Clone(), h_BM_lo.Clone()],
                   legend_titles=['Directly generated', 'Reweighted'],
                   title="LO+PS" % vars(),
                   ratio=True,
                   log_y=False,
                   log_x=False,
                   ratio_range="0.,2.0",
                   custom_x_range=True,
                   x_axis_max=1000,
                   x_axis_min=250,
                   custom_y_range=False,
                   y_axis_max=4000,
                   y_axis_min=0,
                   x_title=x_title,
                   y_title=y_title,
                   extra_pad=0,
                   norm_hists=False,
                   plot_name=plot_name.replace('NLO_Validation','ReweightValidation_noBefore'),
                   label=bm['label'],
                   norm_bins=True,
                   IncErrors=True,
                   wideLeg=True)


        plotting.CompareHists(hists=[h_BM_weighted.Clone() if x != '_inc_kfactors' else h_BM_weighted_kfacts.Clone(), h_BM_lo.Clone() if x != '_inc_kfactors' else h_BM_lo_kfacts.Clone()],
                     legend_titles=['NLO-approx+PS','LO+PS'],
                     title="",
                     ratio=True,
                     log_y=False,
                     log_x=False,
                     ratio_range="0.,2.",
                     custom_x_range=False,
                     x_axis_max=1000,
                     x_axis_min=250,
                     custom_y_range=False,
                     y_axis_max=4000,
                     y_axis_min=0,
                     x_title=x_title,
                     y_title=y_title,
                     extra_pad=0,
                     norm_hists=norm_hists,
                     plot_name=plot_name.replace('NLO_Validation','NLO_vsLO'+x)+'_BM_%(bm_name)s' % vars(),
                     label=bm['label'],
                     norm_bins=True,
                     IncErrors=True,
                     skipCols=1)


    plotting.CompareHists(hists=[h_BM_lo.Clone() if x != '_inc_kfactors' else h_BM_lo_kfacts.Clone(), h_BM_lo_noint.Clone() if x != '_inc_kfactors' else h_BM_lo_noint_kfacts.Clone() ],
                 legend_titles=['Inc. intef.', 'No intef.'] + (['Inc. K-factor scaling'] if x == '_inc_kfactors' else []),
                 title="LO+PS",
                 ratio=True,
                 log_y=False,
                 log_x=False,
                 ratio_range="0.0,2.0",
                 custom_x_range=False,
                 x_axis_max=1000,
                 x_axis_min=250,
                 custom_y_range=False,
                 y_axis_max=4000,
                 y_axis_min=0,
                 x_title=x_title,
                 y_title=y_title,
                 extra_pad=0,
                 norm_hists=norm_hists,
                 plot_name=plot_name.replace('plots_NLO','plots_CompWOInt').replace('NLO_Validation','LO_CompWOInt'+x)+'_BM_%(bm_name)s' % vars(),
                 label=bm['label'],
                 norm_bins=True,
                 IncErrors=True,
                 wideLeg=True,     
                 skipCols=1) 
 
    for log_y in [False, True]:
      plotting.CompareHists(hists=[h_BM_lo.Clone() if x != '_inc_kfactors' else h_BM_lo_kfacts.Clone(), h_BM_lo_noint.Clone() if x != '_inc_kfactors' else h_BM_lo_noint_kfacts.Clone(), sH_lo.Clone() if x != '_inc_kfactors' else sH_lo_kfacts.Clone(), h_BM_approx_lo.Clone() if x != '_inc_kfactors' else h_BM_approx_kfacts_lo.Clone() ],
                   legend_titles=['Inc. intef.', 'No intef.', 's-chan. only', 'hh (SM) + s-chan.'] + (['Inc. K-factor scaling'] if x == '_inc_kfactors' else []),
                   title="LO+PS",
                   ratio=True,
                   log_y=log_y,
                   log_x=False,
                   ratio_range="0.0,2.0",
                   #ratio_range= "0.0,3.0" if 'h_mass(' in plot else "0.0,2.0",
                   custom_x_range=False,
                   x_axis_max=1000,
                   x_axis_min=250,
                   custom_y_range=False,
                   y_axis_max=4000,
                   y_axis_min=0,
                   x_title=x_title,
                   y_title=y_title,
                   extra_pad=0,
                   norm_hists=norm_hists,
                   plot_name=plot_name.replace('plots_NLO','plots_CompWOInt').replace('NLO_Validation','LO_CompAll'+x)+'_BM_%(bm_name)s' % vars() + ('_logy' if log_y else ''),
                   label=bm['label'],
                   norm_bins=True,
                   IncErrors=True,
                   wideLeg=True,     
                   skipCols=1)

    if  bm_name == 'singlet_M260' and 'hh_mass(' in plot and not fineBins:     
      for log_y in [False, True]:
          plotting.CompareHists(hists=[h_BM_nonres_lo.Clone() if x != '_inc_kfactors' else h_BM_nonres_kfacts_lo.Clone(), h_SM_newbins_lo.Clone() if x != '_inc_kfactors' else h_SM_newbins_kfacts_lo.Clone(), h_BM_kapt1_nonres_lo.Clone() if x != '_inc_kfactors' else h_BM_kapt1_nonres_kfacts_lo.Clone() ],
                   legend_titles=['#kappa_{#lambda_{Hhh}}=0.87, #kappa_{t}^{h}=0.97','SM', '#kappa_{#lambda_{Hhh}}=0.87, #kappa_{t}^{h}=1.0'] + (['Inc. K-factor scaling'] if x == '_inc_kfactors' else []),
                   title="LO+PS",
                   ratio=True,
                   log_y=log_y,
                   log_x=False,
                   ratio_range="0.2,1.8",
                   #ratio_range= "0.0,6.0" if 'M260' in bm_name else "0.0,2.0",
                   custom_x_range=False,
                   x_axis_max=1000,
                   x_axis_min=250,
                   custom_y_range=False,
                   y_axis_max=4000,
                   y_axis_min=0,
                   x_title=x_title,
                   y_title=y_title,
                   extra_pad=0,
                   norm_hists=norm_hists,
                   plot_name=plot_name.replace('plots_NLO','plots_CompWOInt').replace('NLO_Validation','LO_CompNonRes'+x)+'_BM_%(bm_name)s' % vars() + ('_logy' if log_y else ''),
                   label=bm['label'],
                   norm_bins=True,
                   IncErrors=True,
                   wideLeg=True,     
                   skipCols=1)

    if bm_name == 'singlet_M600':
    
        plotting.CompareHists(hists=[h_BM_weighted.Clone() if x != '_inc_kfactors' else h_BM_weighted_kfacts.Clone(), h_BM_weighted_noint.Clone() if x != '_inc_kfactors' else h_BM_weighted_noint_kfacts.Clone()],
                     legend_titles=['Inc. intef.', 'No intef.'] + (['Inc. K-factor scaling'] if x == '_inc_kfactors' else []),
                     title="NLO-approx+PS",
                     ratio=True,
                     log_y=False,
                     log_x=False,
                     ratio_range="0.,1.5",
                     custom_x_range=False,
                     x_axis_max=1000,
                     x_axis_min=250,
                     custom_y_range=False,
                     y_axis_max=4000,
                     y_axis_min=0,
                     x_title=x_title,
                     y_title=y_title,
                     extra_pad=0,
                     norm_hists=norm_hists,
                     plot_name=plot_name.replace('plots_NLO','plots_CompWOInt').replace('NLO_Validation','NLO_CompWOInt'+x)+'_BM_%(bm_name)s' % vars(),
                     label=bm['label'],
                     norm_bins=True,
                     IncErrors=True,
                     wideLeg=True,     
                     skipCols=1)                 

print('SH xs (LO) = ', sH_lo.Integral(-1,-1))
print('SH xs (NNLO) = ', sH_lo_kfacts.Integral(-1,-1))
print('full xs (LO) = ', h_BM_lo.Integral(-1,-1))
print('full xs (NNLO) = ', h_BM_lo_kfacts.Integral(-1,-1))
