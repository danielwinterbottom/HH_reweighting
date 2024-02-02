import ROOT
import plotting
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mass_cut', help= 'apply reco-mass cut for plots (500-700 GeV)', action='store_true')
args = parser.parse_args()


f1 = ROOT.TFile('outputs_4b_old/output_powheg_pythia_sm_v2.root') # remove v2!
f2 = ROOT.TFile('outputs_4b_old/output_powheg_pythia_box.root')
f3 = ROOT.TFile('outputs_4b_old/output_powheg_pythia_chhh10_v2.root') # remove v2!

f4 = ROOT.TFile('outputs_4b_old/output_powheg_pythia_from_single_H_width_5GeV.root')
f5 = ROOT.TFile('outputs_4b_old/output_powheg_pythia_from_single_H_width_12GeV.root')
f6 = ROOT.TFile('outputs_4b_old/output_mg_pythia_sm.root')
f7 = ROOT.TFile('outputs_4b_old/output_mg_pythia_width_5GeV.root')


benchmarks = {}
benchmarks['singlet_M600'] = {
  'kappa_h_t' : 0.9854491056576354,
  'kappa_H_t' : 0.16997076265807162,
  'kappa_h_lam' : 0.9491226120544515,
  'kappa_H_lam' : 5.266738184342865,
  'width' : 4.98,
  'width_name' : '0p0083',
  'mass': 600.,
}

bm = benchmarks['singlet_M600']

kappa_h_t = bm['kappa_h_t']
kappa_H_t = bm['kappa_H_t']
kappa_h_lam = bm['kappa_h_lam']
kappa_H_lam = bm['kappa_H_lam']

norm_hists=False

partial_width = 0.06098 # partial width for kap112=1 and M=600 GeV

xs_box_nnlo = 70.3874
xs_Sh_nnlo = 11.0595
xs_box_Sh_int_nnlo = -50.4111
xs_SH_nnlo = 2.006

xs_box_lo = 27.4968
xs_Sh_lo = 3.6962
xs_box_Sh_int_lo = -18.0793
xs_SH_lo = 0.7309

#box: 60.3702386403 58.4478649041 1.03289040138
#Sh: 9.16625205944 8.30809509287 1.10329166397
#Sh-box int: -42.3503809411 -39.5673159149 1.07033747329

xs_box_nlo = 60.3702386403
xs_Sh_nlo = 9.16625205944
xs_box_Sh_int_nlo = -42.3503809411
xs_SH_nlo = 1.5129402

xs_box_nlo_rw = 58.4478649041
xs_Sh_nlo_rw = 8.30809509287
xs_box_Sh_int_nlo_rw = -39.5673159149
xs_SH_nlo_rw = 1.5395984 # assuming you reweight a 12 GeV width to a 5 GeV one

k_box_lo = xs_box_nnlo/xs_box_lo
k_sh_lo = xs_Sh_nnlo/xs_Sh_lo
k_box_sh_int_lo = xs_box_Sh_int_nnlo/xs_box_Sh_int_lo
k_sH_lo = xs_SH_nnlo/xs_SH_lo
k_sH_box_int_lo = (k_box_lo*k_sH_lo)**.5
k_sH_sh_int_lo = (k_sh_lo*k_sH_lo)**.5

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

print '\n***************'
print 'K-factors (LO->NNLO):'

print 'k_box =', k_box_lo
print 'k_sh =', k_sh_lo
print 'k_box_sh_int =', k_box_sh_int_lo
print 'k_sH =', k_sH_lo
print 'k_sH_box_int =', k_sH_box_int_lo
print 'k_sH_sh_int =', k_sH_sh_int_lo

print '\nK-factors (NLO->NNLO):'

print 'k_box =', k_box_nlo
print 'k_sh =', k_sh_nlo
print 'k_box_sh_int =', k_box_sh_int_nlo
print 'k_sH =', k_sH_nlo
print 'k_sH_box_int =', k_sH_box_int_nlo
print 'k_sH_sh_int =', k_sH_sh_int_nlo

print '\nK-factors (NLOApprox->NNLO):'

print 'k_box =', k_box_nlo_rw
print 'k_sh =', k_sh_nlo_rw
print 'k_box_sh_int =', k_box_sh_int_nlo_rw
print 'k_sH =', k_sH_nlo_rw
print 'k_sH_box_int =', k_sH_box_int_nlo_rw
print 'k_sH_sh_int =', k_sH_sh_int_nlo_rw

print '***************\n'

def DrawHist(f, h, plot,wt_extra='1'):
  t = f.Get('ntuple')
  N = t.GetEntries()
  h_name = str(h.GetName())
  if 'abs(' not in plot: 
      var = plot.split('(')[0]
      bins = '('+plot.split('(')[1]
  else:   
      var = '('.join(plot.split('(')[0:2])
      bins = '('+plot.split('(')[2] 

  if args.mass_cut: t.Draw('%(var)s>>%(h_name)s%(bins)s'  % vars(),'(hh_mass_smear_improved<700&&hh_mass_smear_improved>500)*(wt_nom)*'+wt_extra, 'goff')
  else: t.Draw('%(var)s>>%(h_name)s%(bins)s'  % vars(),'(wt_nom)*'+wt_extra, 'goff')

  h = t.GetHistogram()
  h.Scale(1000./N) # units from pb to fb
  return h

plots = ['hh_mass(200,200,1000)', 'hh_mass_smear(75,250,1000)', 'hh_mass_smear_improved(75,250,1000)',
        'hh_pT(75,0,300)', 'hh_pT_smear(75,0,300)', 'hh_pT_smear_improved(75,0,300)',
        'h1_pT(50,0,500)','h2_pT(50,0,500)','h1_pT_smear(50,0,500)','h2_pT_smear(50,0,500)', 'h1_pT_smear_improved(50,0,500)','h2_pT_smear_improved(50,0,500)',
        'h1_eta(100,-7,7)','h2_eta(100,-7,7)','h1_eta_smear(100,-7,7)','h2_eta_smear(100,-7,7)',
        'hh_dR(100,0,10)', 'hh_dR_smear(100,0,10)',
        'hh_dphi(100,0,5)', 'hh_dphi_smear(100,0,5)',
        'fabs(h1_eta-h2_eta)(100,0,7)', 'fabs(h1_eta_smear-h2_eta_smear)(100,0,7)',
        'hh_eta(100,-7,7)', 'hh_eta_smear(100,-7,7)'
        ]

for plot in plots:

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

  h_sm = DrawHist(f1,h_sm,plot)
  h_box = DrawHist(f2,h_box,plot)
  h_chhh10 = DrawHist(f3,h_chhh10,plot)

  plot_mod = plot
  if 'hh_mass' in plot and 'hh_mass_smear_improved' not in plot: plot_mod = 'hh_mass(50,500,700)'

  h_sH = DrawHist(f4,h_sH,plot_mod)
  h_sH.Scale(partial_width/5.)

  h_sH_weighted = DrawHist(f5,h_sH_weighted,plot_mod,'(wt_schannel_H_Mass_600_RelWidth_0p008333)')
  h_sH_weighted.Scale(partial_width/12.)
  h_sH_lo = DrawHist(f7,h_sH_lo,plot_mod)

  h_sH_box_weighted = DrawHist(f1,h_sH_box_weighted,plot,'(wt_box_and_schannel_H_i_Mass_600_RelWidth_0p008333)')
  h_sH_box_lo = DrawHist(f6,h_sH_box_lo,plot,'(wt_box_and_schannel_H_i_Mass_600_RelWidth_0p008333)')

  h_sH_sh_weighted = DrawHist(f1,h_sH_sh_weighted,plot,'(wt_schannel_H_and_schannel_h_i_Mass_600_RelWidth_0p008333)')
  h_sH_sh_lo = DrawHist(f6,h_sH_sh_lo,plot,'(wt_schannel_H_and_schannel_h_i_Mass_600_RelWidth_0p008333)')

  #= h_chhh10 - h_box - 10*h_sm + 10*h_box
  #= h_chhh10 + 9*h_box - 10 *h_sm
  h_sh = h_chhh10.Clone()
  h_sh.Add(h_box,9)
  h_sh.Add(h_sm,-10)
  h_sh.Scale(1./90)

  #=h_sm -h_box -(h_chhh10 + 9*h_box - 10 *h_sm)/90
  #=h_sm - h_box - h_chhh10/90 - h_box/10 + h_sm/9
  #=h_sm*(10./9) - h_box*(11./10) - h_chhh10/90.  
  h_int = h_sm.Clone()
  h_int.Scale(10./9.)
  h_int.Add(h_box, -11./10.)
  h_int.Add(h_chhh10,-1./90.)


  h_box_weighted = DrawHist(f1,h_box_weighted,plot, box_wt)
  h_chhh10_weighted = DrawHist(f1,h_chhh10_weighted,plot, chhh10_wt)

  h_sh_weighted = DrawHist(f1,h_sh_weighted,plot, sh_wt)
  h_int_weighted = DrawHist(f1,h_int_weighted,plot, int_wt)

  print 'box:', h_box.Integral(-1,-1), h_box_weighted.Integral(-1,-1), h_box.Integral(-1,-1)/h_box_weighted.Integral(-1,-1)
  print 'Sh:', h_sh.Integral(-1,-1), h_sh_weighted.Integral(-1,-1), h_sh.Integral(-1,-1)/h_sh_weighted.Integral(-1,-1)
  print 'Sh-box int:', h_int.Integral(-1,-1), h_int_weighted.Integral(-1,-1), h_int.Integral(-1,-1)/h_int_weighted.Integral(-1,-1)

  # get LO distributions
  h_sm_lo = DrawHist(f6, h_sm_lo, plot)
  h_box_lo = DrawHist(f6, h_box_lo, plot, box_wt)
  h_sh_lo = DrawHist(f6, h_sh_lo, plot, sh_wt)
  h_int_lo = DrawHist(f6, h_int_lo, plot, int_wt)


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

  else: 
      raise Exception('invalid plot: %s' % plot) 


  if args.mass_cut: plot_name = plot_name.replace('plots_NLO', 'plots_NLO_masscut')

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

  for x in ['','_inc_kfactors']:

    # first make plots without k-factor scaling
    plotting.CompareHists(hists=[h_box.Clone(), h_box_weighted.Clone(), h_box_lo.Clone()],
               legend_titles=['NLO+PS','NLO-approx+PS','LO+PS'],
               scale_factors = [k_box_nlo, k_box_nlo_rw, k_box_lo] if x == '_inc_kfactors' else None,
               title="#Box",
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
               label='',
               norm_bins=True,
               IncErrors=True)

    plotting.CompareHists(hists=[h_sh.Clone(), h_sh_weighted.Clone(), h_sh_lo.Clone()],
                 legend_titles=['NLO+PS','NLO-approx+PS','LO+PS'],
                 scale_factors = [k_sh_nlo, k_sh_nlo_rw, k_sh_lo] if x == '_inc_kfactors' else None,
                 title="S_{h}",
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
                 label='',
                 norm_bins=True,
                 IncErrors=True)

    plotting.CompareHists(hists=[h_int.Clone(), h_int_weighted.Clone(), h_int_lo.Clone()],
                 legend_titles=['NLO+PS','NLO-approx+PS','LO+PS'],
                 scale_factors = [k_box_sh_int_nlo, k_box_sh_int_nlo_rw, k_box_sh_int_lo] if x == '_inc_kfactors' else None,
                 title="S_{h}-#Box",
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
                 label='',
                 norm_bins=True,
                 IncErrors=True)

    # do same for H in s-channel
    plotting.CompareHists(hists=[h_sH.Clone(), h_sH_weighted.Clone(), h_sH_lo.Clone()],
                 legend_titles=['NLO+PS','NLO-approx+PS','LO+PS'],
                 scale_factors = [k_sH_nlo, k_sH_nlo_rw, k_sH_lo] if x == '_inc_kfactors' else None,
                 title="S_{H}",
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
                 label='',
                 norm_bins=True,
                 IncErrors=True)

   # now makes plots of inteferences without the NLO exact (as it does not exist)

    plotting.CompareHists(hists=[h_sH_box_weighted.Clone(), h_sH_box_lo.Clone()],
                 legend_titles=['NLO-approx+PS','LO+PS'],
                 scale_factors = [k_sH_box_int_nlo_rw, k_sH_box_int_lo] if x == '_inc_kfactors' else None,
                 title="S_{H}-#Box",
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
                 label='',
                 norm_bins=True,
                 IncErrors=True,
                 skipCols=1)

    plotting.CompareHists(hists=[h_sH_sh_weighted.Clone(), h_sH_sh_lo.Clone()],
                 legend_titles=['NLO-approx+PS','LO+PS'],
                 scale_factors = [k_sH_sh_int_nlo_rw, k_sH_sh_int_lo] if x == '_inc_kfactors' else None,
                 title="S_{H}-S_{h}",
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
                 label='',
                 norm_bins=True,
                 IncErrors=True,
                 skipCols=1)

    # make plots comparing benchmarks


    box_SF                         = kappa_h_t**4
    schannel_H_SF                  = kappa_H_t**2*kappa_H_lam**2
    schannel_h_SF                  = kappa_h_t**2*kappa_h_lam**2
    box_and_schannel_h_i_SF        = kappa_h_t**3*kappa_h_lam
    box_and_schannel_H_i_SF        = kappa_h_t**2*kappa_H_t*kappa_H_lam
    schannel_H_and_schannel_h_i_SF = kappa_H_t*kappa_H_lam*kappa_h_t*kappa_h_lam

    # make sH histograms again so that they do not include the zoomed binning for mass, and so that 5 GeV width samples are used for both NLO and LO
    h_sH_weighted_v2 = DrawHist(f4,h_sH_weighted_v2,plot)
    h_sH_weighted_v2.Scale(partial_width/5.)
    h_sH_lo_v2 = DrawHist(f7,h_sH_lo_v2,plot)

    wt_BM = '(wt_box*%(box_SF)g + wt_schannel_h*%(schannel_h_SF)g + wt_box_and_schannel_h_i*%(box_and_schannel_h_i_SF)g + wt_box_and_schannel_H_i_Mass_600_RelWidth_0p008333*%(box_and_schannel_H_i_SF)g + wt_schannel_H_and_schannel_h_i_Mass_600_RelWidth_0p008333*%(schannel_H_and_schannel_h_i_SF)g)' % vars()

    wt_BM_kfacts_nlo_rw = '(wt_box*%(box_SF)g*%(k_box_nlo_rw)g + wt_schannel_h*%(schannel_h_SF)g*%(k_sh_nlo_rw)g + wt_box_and_schannel_h_i*%(box_and_schannel_h_i_SF)g*%(k_box_sh_int_nlo_rw)g + wt_box_and_schannel_H_i_Mass_600_RelWidth_0p008333*%(box_and_schannel_H_i_SF)g*%(k_sH_box_int_nlo_rw)g + wt_schannel_H_and_schannel_h_i_Mass_600_RelWidth_0p008333*%(schannel_H_and_schannel_h_i_SF)g*%(k_sH_sh_int_nlo_rw)g)' % vars()

    wt_BM_kfacts_lo = '(wt_box*%(box_SF)g*%(k_box_lo)g + wt_schannel_h*%(schannel_h_SF)g*%(k_sh_lo)g + wt_box_and_schannel_h_i*%(box_and_schannel_h_i_SF)g*%(k_box_sh_int_lo)g + wt_box_and_schannel_H_i_Mass_600_RelWidth_0p008333*%(box_and_schannel_H_i_SF)g*%(k_sH_box_int_lo)g + wt_schannel_H_and_schannel_h_i_Mass_600_RelWidth_0p008333*%(schannel_H_and_schannel_h_i_SF)g*%(k_sH_sh_int_lo)g)' % vars()
    
    wt_BM_noint_kfacts_lo = '(wt_box*%(box_SF)g*%(k_box_lo)g + wt_schannel_h*%(schannel_h_SF)g*%(k_sh_lo)g + wt_box_and_schannel_h_i*%(box_and_schannel_h_i_SF)g*%(k_box_sh_int_lo)g)' % vars()

    wt_BM_noint = '(wt_box*%(box_SF)g + wt_schannel_h*%(schannel_h_SF)g + wt_box_and_schannel_h_i*%(box_and_schannel_h_i_SF)g)' % vars()

    wt_BM_noint_kfacts_nlo_rw = '(wt_box*%(box_SF)g*%(k_box_nlo_rw)g + wt_schannel_h*%(schannel_h_SF)g*%(k_sh_nlo_rw)g + wt_box_and_schannel_h_i*%(box_and_schannel_h_i_SF)g*%(k_box_sh_int_nlo_rw)g)' % vars()

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

    h_BM_lo_noint = ROOT.TH1D()
    h_BM_lo_noint.SetName('BM_lo')
    h_BM_lo_noint = DrawHist(f6,h_BM_lo,plot, wt_BM_noint)
    h_BM_lo_noint.Add(h_sH_lo_v2,schannel_H_SF)

    h_BM_lo_kfacts = ROOT.TH1D()
    h_BM_lo_kfacts.SetName('BM_lo_kfacts')
    h_BM_lo_kfacts = DrawHist(f6,h_BM_lo_kfacts,plot, wt_BM_kfacts_lo)
    h_BM_lo_kfacts.Add(h_sH_lo_v2,schannel_H_SF*k_sH_lo)

    h_BM_lo_noint_kfacts = ROOT.TH1D()
    h_BM_lo_noint_kfacts.SetName('BM_lo_kfacts')
    h_BM_lo_noint_kfacts = DrawHist(f6,h_BM_lo_kfacts,plot, wt_BM_noint_kfacts_lo)
    h_BM_lo_noint_kfacts.Add(h_sH_lo_v2,schannel_H_SF*k_sH_lo)    


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
                 plot_name=plot_name.replace('NLO_Validation','NLO_vsLO'+x)+'_BM',
                 label='',
                 norm_bins=True,
                 IncErrors=True,
                 skipCols=1)

   # print 'KolmogorovTest for plot %s = %.4f' % (plot,h_BM_lo.KolmogorovTest(h_BM_lo_noint))

    plotting.CompareHists(hists=[h_BM_lo.Clone() if x != '_inc_kfactors' else h_BM_lo_kfacts.Clone(), h_BM_lo_noint.Clone() if x != '_inc_kfactors' else h_BM_lo_noint_kfacts.Clone()],
                 legend_titles=['LO+PS (inc. intef.)', 'LO+PS (no intef.)'],
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
                 plot_name=plot_name.replace('plots_NLO','plots_CompWOInt').replace('NLO_Validation','LO_CompWOInt'+x)+'_BM',
                 label='',
                 norm_bins=True,
                 IncErrors=True,
                 skipCols=1)     

    plotting.CompareHists(hists=[h_BM_weighted.Clone() if x != '_inc_kfactors' else h_BM_weighted_kfacts.Clone(), h_BM_weighted_noint.Clone() if x != '_inc_kfactors' else h_BM_weighted_noint_kfacts.Clone()],
                 legend_titles=['NLO-approx+PS (inc. intef.)', 'NLO-approx+PS (no intef.)'],
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
                 plot_name=plot_name.replace('plots_NLO','plots_CompWOInt').replace('NLO_Validation','NLO_CompWOInt'+x)+'_BM',
                 label='',
                 norm_bins=True,
                 IncErrors=True,
                 skipCols=1)                 

