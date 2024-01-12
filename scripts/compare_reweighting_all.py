import ROOT
import plotting


f1 = ROOT.TFile('outputs_new/output_powheg_sm_v3.root')
f2 = ROOT.TFile('outputs_new/output_powheg_box.root')
f3 = ROOT.TFile('outputs_new/output_powheg_chhh10_v2.root')

f4 = ROOT.TFile('outputs_new/output_powheg_pythia_from_single_H_width_5GeV.root')
f5 = ROOT.TFile('outputs_new/output_powheg_pythia_from_single_H_width_12GeV.root')
f6 = ROOT.TFile('outputs_new/output_mg_pythia_sm.root')
f7 = ROOT.TFile('outputs_new/output_mg_pythia_width_6GeV_temp.root')



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
xs_SH_nlo = 1.3046625

xs_box_nlo_rw = 58.4478649041
xs_Sh_nlo_rw = 8.30809509287
xs_box_Sh_int_nlo_rw = -39.5673159149
xs_SH_nlo_rw = 1.3319042 # assuming you reweight a 12 GeV width to a 5 GeV one

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
  var = plot.split('(')[0]
  bins = '('+plot.split('(')[1]
  #t.Draw('%(var)s>>%(h_name)s%(bins)s'  % vars(),'(wt_nom)*(jet_pdgid==21)*'+wt_extra, 'goff')
  #t.Draw('%(var)s>>%(h_name)s%(bins)s'  % vars(),'(wt_nom)*(jet_pdgid!=21)*'+wt_extra, 'goff')
  t.Draw('%(var)s>>%(h_name)s%(bins)s'  % vars(),'(wt_nom)*'+wt_extra, 'goff')
  h = t.GetHistogram()
  h.Scale(1000./N) # units from pb to fb
  return h

plots = ['hh_mass(75,250,1000)', 'hh_pT(75,0,300)']

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

  h_sH_lo = ROOT.TH1D()
  h_sH_lo.SetName('sH_lo')
  h_sH = ROOT.TH1D()
  h_sH.SetName('sH')
  h_sH_weighted = ROOT.TH1D()
  h_sH_weighted.SetName('sH_weighted')


  h_sm = DrawHist(f1,h_sm,plot)
  h_box = DrawHist(f2,h_box,plot)
  h_chhh10 = DrawHist(f3,h_chhh10,plot)

  h_sH = DrawHist(f4,h_sH,plot)
  h_sH.Scale(partial_width/5.)

  h_sH_weighted = DrawHist(f5,h_sH_weighted,plot,'(wt_schannel_H_Mass_600_RelWidth_0p008333)')
  h_sH_weighted.Scale(partial_width/12.)

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
  h_sH_lo = DrawHist(f7,h_sH_lo,plot)


  if 'hh_mass' in plot: 
    x_title="m_{hh} (GeV)"
    y_title="d#sigma/dm_{hh} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_mass'

  if 'hh_pT' in plot:
    x_title="p_{T}^{hh} (GeV)"
    y_title="d#sigma/dp_{T}^{hh} (fb/GeV)"
    plot_name = 'plots_NLO/dihiggs_NLO_Validation_hh_pT'


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

  # make plots comparing H s-channels with different widths

  # make plots comparing LO to NLO to NLO-approx

  for x in ['','_inc_kfactors']:

#k_box_nlo_rw = xs_box_nnlo/xs_box_nlo_rw
#k_sh_nlo_rw = xs_Sh_nnlo/xs_Sh_nlo_rw
#k_box_sh_int_nlo_rw = xs_box_Sh_int_nnlo/xs_box_Sh_int_nlo_rw
#k_sH_nlo_rw = xs_SH_nnlo/xs_SH_nlo_rw
#k_sH_box_int_nlo_rw = (k_box_nlo_rw*k_sH_nlo_rw)**.5
#k_sH_sh_int_nlo_rw = (k_sh_nlo_rw*k_sH_nlo_rw)**.5

    # first make plots without k-factor scaling
    plotting.CompareHists(hists=[h_box.Clone(), h_box_weighted.Clone(), h_box_lo.Clone()],
               legend_titles=['NLO','NLO-approx','LO'],
               scale_factors = [k_box_nlo, k_box_nlo_rw, k_box_lo] if x == '_inc_kfactors' else None,
               title="",
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
                 legend_titles=['NLO','NLO-approx','LO'],
                 scale_factors = [k_sh_nlo, k_sh_nlo_rw, k_sh_lo] if x == '_inc_kfactors' else None,
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
                 plot_name=plot_name.replace('NLO_Validation','NLO_vsLO'+x)+'_Sh',
                 label='',
                 norm_bins=True,
                 IncErrors=True)

    plotting.CompareHists(hists=[h_int.Clone(), h_int_weighted.Clone(), h_int_lo.Clone()],
                 legend_titles=['NLO','NLO-approx','LO'],
                 scale_factors = [k_box_sh_int_nlo, k_box_sh_int_nlo_rw, k_box_sh_int_lo] if x == '_inc_kfactors' else None,
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
                 plot_name=plot_name.replace('NLO_Validation','NLO_vsLO'+x)+'_box_Sh_int',
                 label='',
                 norm_bins=True,
                 IncErrors=True)

    # do same for H in s-channel
    plotting.CompareHists(hists=[h_sH.Clone(), h_sH_weighted.Clone(), h_sH_lo.Clone()],
                 legend_titles=['NLO','NLO-approx','LO'],
                 scale_factors = [k_sH_nlo, k_sH_nlo_rw, k_sH_lo] if x == '_inc_kfactors' else None,
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
                 plot_name=plot_name.replace('NLO_Validation','NLO_vsLO'+x)+'_SH',
                 label='',
                 norm_bins=True,
                 IncErrors=True)
