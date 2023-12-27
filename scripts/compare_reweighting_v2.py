import ROOT
import argparse
import plotting

parser = argparse.ArgumentParser()
parser.add_argument('--use_smeared_mass', help= 'Use di-Higgs masses with experimental smearing', type=int, default=0)
args = parser.parse_args()

use_smeared_mass=args.use_smeared_mass

var = 'hh_mass'

name_extra = ''
if use_smeared_mass==1:   
    var = 'hh_mass_smear_improved'
    name_extra = '_smeared'
elif use_smeared_mass==2: 
    var = 'hh_mass_smear_2b2ta'
    name_extra = '_smeared_2b2ta'

xs_box_nnlo = 70.3874
xs_Sh_nnlo = 11.0595
xs_box_Sh_int_nnlo = -50.4111
xs_SH_M600_nnlo = 2.006
xs_SH_M650_nnlo = 1.352

xs_box_lo = 27.4968
xs_Sh_lo = 3.6962
xs_box_Sh_int_lo = -18.0793
xs_SH_M600_lo = 0.7309
xs_SH_M650_lo = 0.4949

k_box = xs_box_nnlo/xs_box_lo
k_sh = xs_Sh_nnlo/xs_Sh_lo
k_box_sh_int = xs_box_Sh_int_nnlo/xs_box_Sh_int_lo
k_sH_M600 = xs_SH_M600_nnlo/xs_SH_M600_lo
k_sH_M650 = xs_SH_M650_nnlo/xs_SH_M650_lo
k_sH_box_int_M600 = (k_box*k_sH_M600)**.5
k_sH_sh_int_M600 = (k_sh*k_sH_M600)**.5
k_sH_box_int_M650 = (k_box*k_sH_M650)**.5
k_sH_sh_int_M650 = (k_sh*k_sH_M650)**.5

print '\n***************'
print 'K-factors:'

print 'k_box =', xs_box_nnlo/xs_box_lo
print 'k_sh =', xs_Sh_nnlo/xs_Sh_lo
print 'k_box_sh_int =', xs_box_Sh_int_nnlo/xs_box_Sh_int_lo
print 'k_sH_M600 =', xs_SH_M600_nnlo/xs_SH_M600_lo
print 'k_sH_M650 =', xs_SH_M650_nnlo/xs_SH_M650_lo
print 'k_sH_box_int_M600 =', (k_box*k_sH_M600)**.5
print 'k_sH_sh_int_M600 =', (k_sh*k_sH_M600)**.5
print 'k_sH_box_int_M650 =', (k_box*k_sH_M650)**.5
print 'k_sH_sh_int_M650 =', (k_sh*k_sH_M650)**.5 

print '***************\n'

benchmarks = {}

benchmarks['KappasEq1'] = {
  'kappa_h_t' : 1.0,
  'kappa_H_t' : 1.0,
  'kappa_h_lam' : 1.0,
  'kappa_H_lam' : 1.0,
  'width' : 6,
  'width_name' : '0p01',
  'mass': 600.,
}

benchmarks['singlet_M600'] = {
  'kappa_h_t' : 0.9854491056576354, 
  'kappa_H_t' : 0.16997076265807162,
  'kappa_h_lam' : 0.9491226120544515, 
  'kappa_H_lam' : 5.266738184342865, 
  'width' : 4.98,
  'width_name' : '0p0083',
  'mass': 600.,
} 

benchmarks['2HDM_M650_cosbma_0p1'] = {
  'kappa_h_t' : 1.00832077044,
  'kappa_H_t' : 0.232664991614,
  'kappa_h_lam' : 0.670764959202,
  'kappa_H_lam' : 3.96079660446,
  'width' : 2.47253022,
  'width_name': '0p0038',
  'mass': 650.,
} 

benchmarks['2HDM_M650_cosbma_0p25'] = {
  'kappa_h_t' : 1.00157916989,
  'kappa_H_t' : 0.379099444874,
  'kappa_h_lam' : 0.814929605311,
  'kappa_H_lam' : -4.99100163989,
  'width' : 11.8110613,
  'width_name': '0p018',
  'mass': 650.,
}

benchmarks['2HDM_M650_cosbma_0p2_mA400'] = {
  'kappa_h_t' : 0.999795897113,
  'kappa_H_t' : 0.297979589711,
  'kappa_h_lam' : 1.00104407463,
  'kappa_H_lam' : -5.21172993052,
  'width' : 68.2467897,
  'width_name': '0p105',
  'mass': 650.,
}

benchmarks['2HDM_M650_cosbma_0p2_mA450'] = {
  'kappa_h_t' : 0.999795897113,
  'kappa_H_t' : 0.297979589711,
  'kappa_h_lam' : 1.00104407463,
  'kappa_H_lam' : -5.21172993052,
  'width' : 40.5824759,
  'width_name': '0p062',
  'mass': 650.,
}

benchmarks['2HDM_M650_cosbma_0p2_mA500'] = {
  'kappa_h_t' : 0.999795897113,
  'kappa_H_t' : 0.297979589711,
  'kappa_h_lam' : 1.00104407463,
  'kappa_H_lam' : -5.21172993052,
  'width' : 21.0952487,
  'width_name': '0p032',
  'mass': 650.,
}

benchmarks['2HDM_M650_cosbma_0p2_mA550'] = {
  'kappa_h_t' : 0.999795897113,
  'kappa_H_t' : 0.297979589711,
  'kappa_h_lam' : 1.00104407463,
  'kappa_H_lam' : -5.21172993052,
  'width' : 10.8927402,
  'width_name': '0p017',
  'mass': 650.,
}

benchmarks['2HDM_M650_cosbma_0p2_mA330'] = {
  'kappa_h_t' : 0.999795897113,
  'kappa_H_t' : 0.297979589711,
  'kappa_h_lam' : 1.00104407463,
  'kappa_H_lam' : -5.21172993052,
  'width' : 114.660086,
  'width_name': '0p176',
  'mass': 650.,
}

h_box = ROOT.TH1D('h_box','',200,200,1000)
h_schannel_h = ROOT.TH1D('h_schannel_h','',200,200,1000)
h_box_and_schannel_h_i = ROOT.TH1D('h_box_and_schannel_h_i','',200,200,1000)

# first get non-resonant histograms that don't depend on mass and width of heavy scalar
f1 = ROOT.TFile('outputs_new/output_HH_loop_sm_twoscalar_BOX_Ntot_200000_Njob_10000.root')
f2 = ROOT.TFile('outputs_new/output_HH_loop_sm_twoscalar_BOX_SChan_h_inteference_Ntot_200000_Njob_10000.root')
f3 = ROOT.TFile('outputs_new/output_HH_loop_sm_twoscalar_SChan_h_Ntot_200000_Njob_10000.root')

lum = 300000. # 300/pb

#if use_smeared_mass == 1:
#  lum*=0.001716 # multiply by bbgamgam BR for realistic yields

def DrawHist(f, h, var,wt_extra='1'):
  t = f.Get('ntuple')
  #N = t.GetEntries()
  N = abs(t.GetEntries('wt_nom>0')-t.GetEntries('wt_nom<0'))
  h_name = str(h.GetName())
  t.Draw('%(var)s>>%(h_name)s(200,200,1000)'  % vars(),'wt_nom*'+wt_extra, 'goff')
  h = t.GetHistogram()
  h.Scale(1000./N) # units from pb to fb
  return h

def DrawHistExcludeRange(f, h, var, exclude_range):
  t = f.Get('ntuple')
  #N = t.GetEntries()
  N = abs(t.GetEntries('wt_nom>0')-t.GetEntries('wt_nom<0'))
  h_name = str(h.GetName())
  r_min = exclude_range[0]
  r_max = exclude_range[1]
  t.Draw('%(var)s>>%(h_name)s(200,200,1000)'  % vars(),'wt_nom*(hh_mass<%g||hh_mass>%g)' % (r_min, r_max), 'goff')
  h = t.GetHistogram()
  h.Scale(1000./N) # units from pb to fb
  return h

h_box = DrawHist(f1,h_box,var)
h_box_and_schannel_h_i = DrawHist(f2,h_box_and_schannel_h_i,var)
h_schannel_h = DrawHist(f3,h_schannel_h,var)

h_SM = h_box.Clone()
h_SM.SetName('h_SM')
h_SM.Add(h_box_and_schannel_h_i)
h_SM.Add(h_schannel_h)


print 'box,box-S_{h},S_{h} =', h_box.Integral(), h_box_and_schannel_h_i.Integral(), h_schannel_h.Integral()

for bm_name in benchmarks:

  bm = benchmarks[bm_name]

  kappa_h_t = bm['kappa_h_t'] 
  kappa_H_t = bm['kappa_H_t'] 
  kappa_h_lam = bm['kappa_h_lam'] 
  kappa_H_lam = bm['kappa_H_lam'] 
  width = bm['width']
  width_name = bm['width_name']
  mass = bm['mass']

  width_sf = 0.01*mass/width 

  box_SF                         = kappa_h_t**4
  schannel_H_SF                  = kappa_H_t**2*kappa_H_lam**2
  schannel_h_SF                  = kappa_h_t**2*kappa_h_lam**2
  box_and_schannel_h_i_SF        = kappa_h_t**3*kappa_h_lam
  box_and_schannel_H_i_SF        = kappa_h_t**2*kappa_H_t*kappa_H_lam
  schannel_H_and_schannel_h_i_SF = kappa_H_t*kappa_H_lam*kappa_h_t*kappa_h_lam 

  h_schannel_H = ROOT.TH1D('h_schannel_H_%(bm_name)s' % vars(),'',200,200,1000)
  h_box_and_schannel_H_i = ROOT.TH1D('h_box_and_schannel_H_i_%(bm_name)s' % vars(),'',200,200,1000)
  h_schannel_H_and_schannel_h_i = ROOT.TH1D('h_schannel_H_and_schannel_h_i_%(bm_name)s' % vars(),'',200,200,1000)
  h_schannel_H.SetDirectory(0)
  h_box_and_schannel_H_i.SetDirectory(0)
  h_schannel_H_and_schannel_h_i.SetDirectory(0)

  h_schannel_H_NWA = ROOT.TH1D('h_schannel_H_NWA_%(bm_name)s' % vars(),'',200,200,1000)
  h_box_and_schannel_H_i_NWA = ROOT.TH1D('h_box_and_schannel_H_i_NWA_%(bm_name)s' % vars(),'',200,200,1000)
  h_schannel_H_and_schannel_h_i_NWA = ROOT.TH1D('h_schannel_H_and_schannel_h_i_NWA_%(bm_name)s' % vars(),'',200,200,1000)
  h_schannel_H_NWA.SetDirectory(0)
  h_box_and_schannel_H_i_NWA.SetDirectory(0)
  h_schannel_H_and_schannel_h_i_NWA.SetDirectory(0)
  h_box_and_schannel_H_i_NWA_v2 = ROOT.TH1D('h_box_and_schannel_H_i_NWA_v2_%(bm_name)s' % vars(),'',200,200,1000)
  h_schannel_H_and_schannel_h_i_NWA_v2 = ROOT.TH1D('h_schannel_H_and_schannel_h_i_NWA_v2_%(bm_name)s' % vars(),'',200,200,1000)
  h_box_and_schannel_H_i_NWA_v2.SetDirectory(0)
  h_schannel_H_and_schannel_h_i_NWA_v2.SetDirectory(0)

  h_box_mod = h_box.Clone()
  h_box_mod.SetDirectory(0)
  h_box_and_schannel_h_i_mod = h_box_and_schannel_h_i.Clone()
  h_box_and_schannel_h_i_mod.SetDirectory(0)
  h_schannel_h_mod = h_schannel_h.Clone()
  h_schannel_h_mod.SetDirectory(0)
  h_box_mod.SetName(h_box_mod.GetName()+'_%(bm_name)s' % vars())
  h_box_and_schannel_h_i_mod.SetName(h_box_and_schannel_h_i_mod.GetName()+'_%(bm_name)s' % vars())
  h_schannel_h_mod.SetName(h_schannel_h_mod.GetName()+'_%(bm_name)s' % vars())

  print '\n\nBenchmark: %(bm_name)s' % vars()

  print '\nScale factors applied:'
  print 'width SF = %.3f' % width_sf
  print 'box_SF = %.3f' % box_SF
  print 'schannel_H_SF = %.3f' % schannel_H_SF
  print 'schannel_h_SF = %.3f' % schannel_h_SF
  print 'box_and_schannel_H_i_SF = %.3f' % box_and_schannel_H_i_SF
  print 'box_and_schannel_h_i_SF = %.3f' % box_and_schannel_h_i_SF
  print 'schannel_H_and_schannel_h_i_SF = %.3f' % schannel_H_and_schannel_h_i_SF 

  f4 = ROOT.TFile('outputs_new/output_HH_loop_sm_twoscalar_BOX_SChan_eta0_inteference_M_%i_RelWidth_%s_Ntot_200000_Njob_10000.root' % (mass,width_name))
  f5 = ROOT.TFile('outputs_new/output_HH_loop_sm_twoscalar_SChan_eta0_M_%i_RelWidth_%s_Ntot_200000_Njob_10000.root' % (mass,width_name))
  f6 = ROOT.TFile('outputs_new/output_HH_loop_sm_twoscalar_SChan_h_SChan_eta0_inteference_M_%i_RelWidth_%s_Ntot_200000_Njob_10000.root' % (mass,width_name))
  
  f7 = ROOT.TFile('outputs_new/output_HH_loop_sm_twoscalar_BOX_SChan_eta0_inteference_M_%i_RelWidth_0p01_Ntot_200000_Njob_10000.root' % mass)
  f8 = ROOT.TFile('outputs_new/output_HH_loop_sm_twoscalar_SChan_eta0_M_%i_RelWidth_0p01_Ntot_200000_Njob_10000.root' % mass)
  f9 = ROOT.TFile('outputs_new/output_HH_loop_sm_twoscalar_SChan_h_SChan_eta0_inteference_M_%i_RelWidth_0p01_Ntot_200000_Njob_10000.root' % mass)

  f10 = ROOT.TFile('outputs_new/output_HH_loop_sm_twoscalar_BOX_SChan_eta0_inteference_M_%i_RelWidth_0p2_Ntot_200000_Njob_10000.root' % mass)
  f11 = ROOT.TFile('outputs_new/output_HH_loop_sm_twoscalar_SChan_h_SChan_eta0_inteference_M_%i_RelWidth_0p2_Ntot_200000_Njob_10000.root' % mass)


  h_box_and_schannel_H_i = DrawHist(f4,h_box_and_schannel_H_i,var)
  h_schannel_H = DrawHist(f5,h_schannel_H,var)
  h_schannel_H_and_schannel_h_i = DrawHist(f6,h_schannel_H_and_schannel_h_i,var)

  h_box_and_schannel_H_i_NWA = DrawHist(f7,h_box_and_schannel_H_i_NWA,var)
  h_schannel_H_NWA = DrawHist(f8,h_schannel_H_NWA,var)
  h_schannel_H_and_schannel_h_i_NWA = DrawHist(f9,h_schannel_H_and_schannel_h_i_NWA,var)

  h_box_and_schannel_H_i_NWA_v2 = DrawHist(f10,h_box_and_schannel_H_i_NWA_v2,var)
  h_schannel_H_and_schannel_h_i_NWA_v2 = DrawHist(f11,h_schannel_H_and_schannel_h_i_NWA_v2,var)
  

  # account for a12 parameter when samples are produced
  h_schannel_H_and_schannel_h_i.Scale(2.)
  h_box_and_schannel_H_i.Scale(2.**1.5)

  h_schannel_H_and_schannel_h_i_NWA.Scale(2.)
  h_box_and_schannel_H_i_NWA.Scale(2.**1.5)

  h_schannel_H_and_schannel_h_i_NWA_v2.Scale(2.)
  h_box_and_schannel_H_i_NWA_v2.Scale(2.**1.5)

  # scale non-res
  h_box_mod.Scale(box_SF)
  h_schannel_h_mod.Scale(schannel_h_SF)
  h_box_and_schannel_h_i_mod.Scale(box_and_schannel_h_i_SF)

  # scale resonant parts
  h_schannel_H.Scale(schannel_H_SF)
  h_schannel_H_and_schannel_h_i.Scale(schannel_H_and_schannel_h_i_SF)
  h_box_and_schannel_H_i.Scale(box_and_schannel_H_i_SF)

  h_schannel_H_NWA.Scale(schannel_H_SF*width_sf)
  h_schannel_H_and_schannel_h_i_NWA.Scale(schannel_H_and_schannel_h_i_SF)
  h_box_and_schannel_H_i_NWA.Scale(box_and_schannel_H_i_SF)

  h_schannel_H_and_schannel_h_i_NWA_v2.Scale(schannel_H_and_schannel_h_i_SF)
  h_box_and_schannel_H_i_NWA_v2.Scale(box_and_schannel_H_i_SF)

  h_BSM_noint = h_schannel_H.Clone()
  h_BSM_noint.Add(h_box_mod)
  h_BSM_noint.Add(h_schannel_h_mod)
  h_BSM_noint.Add(h_box_and_schannel_h_i_mod)

  if bm_name == 'KappasEq1':
      print '\n\n****************************************'
      print 'Cross sections for kappas Equal 1:'
      print 'S_{H} = %.4f fb' % h_schannel_H.Integral(-1,-1)
      print 'S_{H}-S_{h} = %.4f fb' % h_schannel_H_and_schannel_h_i.Integral(-1,-1)
      print 'S_{H}-box = %.4f fb' % h_box_and_schannel_H_i.Integral(-1,-1)
      print 'box = %.4f fb' % h_box_mod.Integral(-1,-1)
      print 'S_{h} = %.4f fb' % h_schannel_h_mod.Integral(-1,-1)
      print 'S_{h}-box = %.4f fb' % h_box_and_schannel_h_i_mod.Integral(-1,-1)
      print '****************************************\n\n'

  h_BSM_total = h_schannel_H.Clone()
  h_BSM_total.Add(h_schannel_H_and_schannel_h_i)
  h_BSM_total.Add(h_box_and_schannel_H_i)
  h_BSM_total.Add(h_box_mod)
  h_BSM_total.Add(h_schannel_h_mod)
  h_BSM_total.Add(h_box_and_schannel_h_i_mod)


  if mass == 600:
      k_sH_sh_int = k_sH_sh_int_M600
      k_sH_box_int = k_sH_box_int_M600
      k_sH = k_sH_M600
  else:
      k_sH_sh_int = k_sH_sh_int_M650
      k_sH_box_int = k_sH_box_int_M650
      k_sH = k_sH_M650


  h_BSM_kfacts = h_schannel_H.Clone()
  h_BSM_kfacts.Scale(k_sH)
  h_BSM_kfacts.Add(h_schannel_H_and_schannel_h_i, k_sH_sh_int)
  h_BSM_kfacts.Add(h_box_and_schannel_H_i, k_sH_box_int)
  h_BSM_kfacts.Add(h_box_mod, k_box)
  h_BSM_kfacts.Add(h_schannel_h_mod, k_sh)
  h_BSM_kfacts.Add(h_box_and_schannel_h_i_mod, k_box_sh_int)

  h_BSM_NWA = h_schannel_H_NWA.Clone()
  h_BSM_NWA.Add(h_schannel_H_and_schannel_h_i_NWA)
  h_BSM_NWA.Add(h_box_and_schannel_H_i_NWA)
  h_BSM_NWA.Add(h_box_mod)
  h_BSM_NWA.Add(h_schannel_h_mod)
  h_BSM_NWA.Add(h_box_and_schannel_h_i_mod)


  h_BSM_NWA_v2 = h_schannel_H_NWA.Clone()
  h_BSM_NWA_v2.Add(h_schannel_H_and_schannel_h_i_NWA_v2)
  h_BSM_NWA_v2.Add(h_box_and_schannel_H_i_NWA_v2)
  h_BSM_NWA_v2.Add(h_box_mod)
  h_BSM_NWA_v2.Add(h_schannel_h_mod)
  h_BSM_NWA_v2.Add(h_box_and_schannel_h_i_mod)

  h_BSM_approx = h_SM.Clone()
  h_BSM_approx.Add(h_schannel_H)
 
  if bm_name == 'singlet_M600':
      # produce a plot for validation
      #f12 = ROOT.TFile('outputs_new/output_HH_loop_sm_twoscalar_Full_M_600_BM_Ntot_200000_Njob_10000.root')
      f12 = ROOT.TFile('outputs_new/output_HH_loop_sm_twoscalar_Full_M_600_BM_Ntot_1000000_Njob_10000.root')
      #f12 = ROOT.TFile('outputs_new/output_HH_loop_sm_twoscalar_Full_M_600_BM_v2_Ntot_200000_Njob_10000.root')
      h_BM = ROOT.TH1D('h_BM','',200,200,1000)
      h_BM = DrawHist(f12,h_BM,var)

      # we also compare to reweighted distributions
      #wt_schannel_H_Mass_600_RelWidth_0p0083
      f13 = ROOT.TFile('outputs_new/output_HH_loop_sm_twoscalar_SChan_eta0_M_600_RelWidth_0p01_Ntot_200000_Njob_10000_with_weights.root')
      f14 = ROOT.TFile('outputs_new/output_HH_loop_sm_twoscalar_SM_Ntot_200000_Njob_10000.root')

      h_box_mod_reweighted = ROOT.TH1D('h_box_mod_reweighted','',200,200,1000)
      h_schannel_h_mod_reweighted = ROOT.TH1D('h_schannel_h_mod_reweighted','',200,200,1000)
      h_box_and_schannel_h_i_mod_reweighted = ROOT.TH1D('h_box_and_schannel_h_i_mod_reweighted','',200,200,1000)
      h_schannel_H_reweighted = ROOT.TH1D('h_schannel_H_reweighted','',200,200,1000)
      h_schannel_H_and_schannel_h_i_reweighted = ROOT.TH1D('h_schannel_H_and_schannel_h_i_reweighted','',200,200,1000)
      h_box_and_schannel_H_i_reweighted = ROOT.TH1D('h_box_and_schannel_H_i_reweighted','',200,200,1000)

      h_schannel_H_reweighted = DrawHist(f13,h_schannel_H_reweighted,var,'wt_schannel_H_Mass_600_RelWidth_0p0083')

      h_box_mod_reweighted = DrawHist(f14,h_box_mod_reweighted,var,'wt_box')
      h_schannel_h_mod_reweighted = DrawHist(f14,h_schannel_h_mod_reweighted,var,'wt_schannel_h')
      h_box_and_schannel_h_i_mod_reweighted = DrawHist(f14,h_box_and_schannel_h_i_mod_reweighted,var,'wt_box_and_schannel_h_i')
      h_box_and_schannel_H_i_reweighted = DrawHist(f14,h_box_and_schannel_H_i_reweighted,var,'wt_box_and_schannel_H_i_Mass_600_RelWidth_0p0083')
      h_schannel_H_and_schannel_h_i_reweighted = DrawHist(f14,h_schannel_H_and_schannel_h_i_reweighted,var,'wt_schannel_H_and_schannel_h_i_Mass_600_RelWidth_0p0083')

      h_nonres_reweighted = ROOT.TH1D('h_nonres_reweighted','',200,200,1000)
      wt_str = '(%(box_SF)g*wt_box + %(schannel_h_SF)g*wt_schannel_h + %(box_and_schannel_h_i_SF)g*wt_box_and_schannel_h_i + %(box_and_schannel_H_i_SF)g*wt_box_and_schannel_H_i_Mass_600_RelWidth_0p0083 + %(schannel_H_and_schannel_h_i_SF)g*wt_schannel_H_and_schannel_h_i_Mass_600_RelWidth_0p0083)' % vars()
      h_nonres_reweighted = DrawHist(f14,h_nonres_reweighted,var,wt_str)

      # scale non-res
      h_box_mod_reweighted.Scale(box_SF)
      h_schannel_h_mod_reweighted.Scale(schannel_h_SF)
      h_box_and_schannel_h_i_mod_reweighted.Scale(box_and_schannel_h_i_SF)
    
      # scale resonant parts
      h_schannel_H_reweighted.Scale(schannel_H_SF)
      h_schannel_H_and_schannel_h_i_reweighted.Scale(schannel_H_and_schannel_h_i_SF)
      h_box_and_schannel_H_i_reweighted.Scale(box_and_schannel_H_i_SF)

      h_BSM_reweighted = h_schannel_H_reweighted.Clone()
      #h_BSM_reweighted.Add(h_schannel_H_and_schannel_h_i_reweighted)
      #h_BSM_reweighted.Add(h_box_and_schannel_H_i_reweighted)
      #h_BSM_reweighted.Add(h_box_mod_reweighted)
      #h_BSM_reweighted.Add(h_schannel_h_mod_reweighted)
      #h_BSM_reweighted.Add(h_box_and_schannel_h_i_mod_reweighted)

      h_BSM_reweighted.Add(h_nonres_reweighted)

      plotting.CompareHists(hists=[h_BM.Clone(), h_BSM_total.Clone(), h_BSM_reweighted.Clone()],
                   legend_titles=['Directly generated', 'Seperatly generated','Reweighted', '#Chi^{2}/ndf = %.1f/%i' % (h_BM.Chi2Test(h_BSM_total,'Chi2 WW'), int((h_BM.Chi2Test(h_BSM_total,'Chi2 WW')/h_BM.Chi2Test(h_BSM_total,'CHI2/NDF WW')))), '#Chi^{2}/ndf = %.1f/%i' % (h_BM.Chi2Test(h_BSM_reweighted,'Chi2 WW'), int((h_BM.Chi2Test(h_BSM_reweighted,'Chi2 WW')/h_BM.Chi2Test(h_BSM_reweighted,'CHI2/NDF WW'))))],
                   title="m_{H} = %(mass)g GeV,  #Gamma_{H} = %(width).1f GeV" % vars(),
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
                   x_title="m_{hh} (GeV)",
                   y_title="d#sigma/dm_{hh} (fb/GeV)",
                   extra_pad=0,
                   norm_hists=False,
                   plot_name="plots_v2/dihiggs_Validation%(name_extra)s" % vars(),
                   label='',
                   norm_bins=True,
                   IncErrors=True)

  print '\n*************************'
  print 'cross-section Full, S_channel_only, SM = ', h_BSM_total.Integral(-1,-1), h_schannel_H.Integral(-1,-1), h_SM.Integral(-1,-1)    
  print 'cross-section times k-factors Full, S_channel_only, SM = ', h_BSM_kfacts.Integral(-1,-1), h_schannel_H.Integral(-1,-1)*k_sH, xs_box_nnlo+xs_Sh_nnlo+xs_box_Sh_int_nnlo 
  print '*************************\n'


  # make plot comparing HH SM to HH BSM
  plotting.CompareHists(hists=[h_BSM_total.Clone(), h_schannel_H.Clone(),h_BSM_approx.Clone()],
               legend_titles=['BSM hh', 'S_{H}', 'S_{H} + SM hh'],
               title="m_{H} = %(mass)g GeV,  #Gamma_{H} = %(width).1f GeV" % vars(),
               ratio=True,
               log_y=False,
               log_x=False,
               ratio_range="0.,4.0",
               custom_x_range=True,
               x_axis_max=1000,
               x_axis_min=250,
               custom_y_range=False,
               y_axis_max=4000,
               y_axis_min=0,
               x_title="m_{hh} (GeV)",
               y_title="d#sigma/dm_{hh} (fb/GeV)",
               extra_pad=0,
               norm_hists=False,
               plot_name="plots_v2/dihiggs_total_%(bm_name)s_Mass%(mass)g_Width%(width_name)s%(name_extra)s" % vars(),
               label="",
               norm_bins=True)


  plotting.CompareHists(hists=[h_BSM_total.Clone(), h_BSM_kfacts.Clone()],
               legend_titles=['LO', 'LO #times k-factors', ],
               title="m_{H} = %(mass)g GeV,  #Gamma_{H} = %(width).1f GeV" % vars(),
               ratio=True,
               log_y=False,
               log_x=False,
               ratio_range="0.,4.0",
               custom_x_range=True,
               x_axis_max=1000,
               x_axis_min=250,
               custom_y_range=False,
               y_axis_max=4000,
               y_axis_min=0,
               x_title="m_{hh} (GeV)",
               y_title="d#sigma/dm_{hh} (fb/GeV)",
               extra_pad=0,
               norm_hists=False,
               plot_name="plots_v2/dihiggs_kfacts_%(bm_name)s_Mass%(mass)g_Width%(width_name)s%(name_extra)s" % vars(),
               label="",
               norm_bins=True)

  # make plot comparing NWA to full BSM
  plotting.CompareHists(hists=[h_BSM_total.Clone(), h_BSM_NWA.Clone()],
               legend_titles=['BSM hh', 'BSM hh (NWA)'],
               title="m_{H} = %(mass)g GeV,  #Gamma_{H} = %(width).1f GeV" % vars(),
               ratio=True,
               log_y=False,
               log_x=False,
               ratio_range="0.0,2.0",
               custom_x_range=True,
               x_axis_max=1000,
               x_axis_min=250,
               custom_y_range=False,
               y_axis_max=4000,
               y_axis_min=0,
               x_title="m_{hh} (GeV)",
               y_title="d#sigma/dm_{hh} (fb/GeV)",
               extra_pad=0,
               norm_hists=False,
               plot_name="plots_v2/dihiggs_NWA_comp_%(bm_name)s_Mass%(mass)g_Width%(width_name)s%(name_extra)s" % vars(),
               label="",
               norm_bins=True)

  # make plots of seperate contributions

  plotting.CompareHists(
               hists=[h_schannel_h_mod.Clone(), h_box_mod.Clone(), h_box_and_schannel_h_i_mod.Clone(), h_schannel_H.Clone(), h_box_and_schannel_H_i.Clone(), h_schannel_H_and_schannel_h_i.Clone()],
               legend_titles=['S_{h}', '#Box', 'S_{h}-#Box', 'S_{H}', 'S_{H}-#Box', 'S_{H}-S_{h}'],
               title="m_{H} = %(mass)g GeV,  #Gamma_{H} = %(width).1f GeV" % vars(),
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
               x_title="m_{hh} (GeV)",
               y_title="d#sigma/dm_{hh} (fb/GeV)",
               extra_pad=0,
               norm_hists=False,
               plot_name="plots_v2/dihiggs_contributions_%(bm_name)s_Mass%(mass)g_Width%(width_name)s%(name_extra)s" % vars(),
               label="",
               norm_bins=True)

  plotting.CompareHists(hists=[h_schannel_H.Clone(), h_box_mod.Clone(), h_schannel_h_mod.Clone(), h_box_and_schannel_h_i_mod.Clone(), h_schannel_H_and_schannel_h_i_NWA.Clone(), h_box_and_schannel_H_i_NWA.Clone()],
               legend_titles=['S_{H}', 'S_{H}-S_{h}', 'S_{H}-#Box', '#Box', 'S_{h}', 'S_{h}-#Box'],
               title="m_{H} = %(mass)g GeV,  #Gamma_{H} = %(width).1f GeV" % vars(),
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
               x_title="m_{hh} (GeV)",
               y_title="d#sigma/dm_{hh} (fb/GeV)",
               extra_pad=0,
               norm_hists=False,
               plot_name="plots_v2/dihiggs_contributions_NWA_%(bm_name)s_Mass%(mass)g_Width%(width_name)s%(name_extra)s" % vars(),
               label="",
               norm_bins=True)

  if width_name != '0p105': continue

  plotting.CompareHists(hists=[h_box_and_schannel_H_i_NWA.Clone(), h_box_and_schannel_H_i.Clone(), h_box_and_schannel_H_i_NWA_v2.Clone()],
               legend_titles=['#Gamma_{H}/m_{H} = 0.01', '#Gamma_{H}/m_{H} = %.2f' % (width/mass), '#Gamma_{H}/m_{H} = 0.2'],
               title="m_{H} = %(mass)g GeV,  S_{H}-#Box" % vars(),
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
               x_title="m_{hh} (GeV)",
               y_title="d#sigma/dm_{hh} (fb/GeV)",
               extra_pad=0,
               norm_hists=False,
               plot_name="plots_v2/dihiggs_Box-SH_widthComp_%(bm_name)s_Mass%(mass)g_Width%(width_name)s%(name_extra)s" % vars(),
               label="",
               norm_bins=True)
