import ROOT
import plotting

f1 = ROOT.TFile('outputs_new/output_powheg_sm_v3_v2.root')
f2 = ROOT.TFile('outputs_new/output_powheg_box_v2.root')
f3 = ROOT.TFile('outputs_new/output_powheg_chhh10_v2_v2.root')

norm_hists=False

partial_width = 0.06098 # partial width for kap112=1 and M=600 GeV

def DrawHist(f, h, plot,wt_extra='1'):
  t = f.Get('ntuple')
  N = t.GetEntries()
  h_name = str(h.GetName())
  var = plot.split('(')[0]
  bins = '('+plot.split('(')[1]
  #t.Draw('%(var)s>>%(h_name)s%(bins)s'  % vars(),'(wt_nom)*(jet_pdgid==21)*'+wt_extra, 'goff')
  #t.Draw('%(var)s>>%(h_name)s%(bins)s'  % vars(),'(wt_nom)*(jet_pdgid!=21)*'+wt_extra, 'goff')
  t.Draw('%(var)s>>%(h_name)s%(bins)s'  % vars(),'(wt_nom)*(part1_pdgid==21&&part2_pdgid==21)*'+wt_extra, 'goff')
  #t.Draw('%(var)s>>%(h_name)s%(bins)s'  % vars(),'(wt_nom)*'+wt_extra, 'goff')
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

  h_sm = DrawHist(f1,h_sm,plot)
  h_box = DrawHist(f2,h_box,plot)
  h_chhh10 = DrawHist(f3,h_chhh10,plot)


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


  #wt_schannel_h = float( ((wt_box_and_schannel_h_2-wt_box) - 10.*(wt_box_and_schannel_h_1-wt_box))/90. )

  #wt_box_and_schannel_h_i = wt_box_and_schannel_h_1 - wt_box - wt_schannel_h
  

  h_box_weighted = DrawHist(f1,h_box_weighted,plot, box_wt)
  h_chhh10_weighted = DrawHist(f1,h_chhh10_weighted,plot, chhh10_wt)

  h_sh_weighted = DrawHist(f1,h_sh_weighted,plot, sh_wt)
  h_int_weighted = DrawHist(f1,h_int_weighted,plot, int_wt)

  print 'box:', h_box.Integral(-1,-1), h_box_weighted.Integral(-1,-1), h_box.Integral(-1,-1)/h_box_weighted.Integral(-1,-1)
  print 'Sh:', h_sh.Integral(-1,-1), h_sh_weighted.Integral(-1,-1), h_sh.Integral(-1,-1)/h_sh_weighted.Integral(-1,-1)
  print 'Sh-box int:', h_int.Integral(-1,-1), h_int_weighted.Integral(-1,-1), h_int.Integral(-1,-1)/h_int_weighted.Integral(-1,-1)

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
