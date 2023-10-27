import ROOT
import plotting

mass='600'
width='12'

f1 = ROOT.TFile('hhmass_output_BM%(mass)s_Width%(width)s.root' % vars())

# first make plots of contributions with kappas all set to 1
box = f1.Get('BM_box_KappasEq1')
schannel_H = f1.Get('BM_schannel_H_KappasEq1')
schannel_h = f1.Get('BM_schannel_h_KappasEq1')
box_and_schannel_H_i = f1.Get('BM_box_and_schannel_H_i_KappasEq1')
box_and_schannel_h_i = f1.Get('BM_box_and_schannel_h_i_KappasEq1')
schannel_H_and_schannel_h_i = f1.Get('BM_schannel_H_and_schannel_h_i_KappasEq1')
sm = box.Clone()
sm.Add(schannel_h)
sm.Add(box_and_schannel_h_i)

hists = [sm,schannel_H,box_and_schannel_H_i,schannel_H_and_schannel_h_i]
titles = ['SM hh', 'S_{H}', 'S_{H}-#Box interference', 'S_{H}-S_{h} interference']

plotting.CompareHists(hists=hists,
             legend_titles=titles,
             title="m_{H} = %(mass)s GeV,  #Gamma_{H} = %(width)s GeV" % vars(),
             ratio=False,
             log_y=False,
             log_x=False,
             ratio_range="0.7,1.3",
             custom_x_range=True,
             x_axis_max=900,
             x_axis_min=250,
             custom_y_range=False,
             y_axis_max=4000,
             y_axis_min=0,
             x_title="m_{hh} (GeV)",
             y_title="a.u",
             extra_pad=0,
             norm_hists=False,
             plot_name="plots/dihiggs_contributions_Mass%(mass)s_Width%(width)s".replace('.','p') % vars(),
             label="",
             norm_bins=False)


#now make a plots for specific benchmark scenarios

# BM scenario for 600 GeV
mass = '600'
width = '4p97918'

for x in ['', '_smeared']:

    f2 = ROOT.TFile('hhmass_output_BM%(mass)s_Width%(width)s%(x)s.root' % vars())
    
    box = f2.Get('BM_box')
    schannel_H = f2.Get('BM_schannel_H')
    schannel_h = f2.Get('BM_schannel_h')
    box_and_schannel_H_i = f2.Get('BM_box_and_schannel_H_i')
    box_and_schannel_h_i = f2.Get('BM_box_and_schannel_h_i')
    schannel_H_and_schannel_h_i = f2.Get('BM_schannel_H_and_schannel_h_i')
    sm_like = box.Clone()
    sm_like.Add(schannel_h)
    sm_like.Add(box_and_schannel_h_i)
    
    total = sm_like.Clone()
    total.Add(schannel_H)
    total.Add(box_and_schannel_H_i)
    total.Add(schannel_H_and_schannel_h_i)
    
    hists = [sm_like,schannel_H,box_and_schannel_H_i,schannel_H_and_schannel_h_i]
    titles = ['SM-like hh', 'S_{H}', 'S_{H}-#Box interference', 'S_{H}-S_{h} interference','total']
    
    width_val = float(width.replace('p','.'))
    
    plotting.CompareHists(hists=hists,
                 legend_titles=titles,
                 title="m_{H} = %(mass)s GeV,  #Gamma_{H} = %(width_val).2f GeV" % vars(),
                 ratio=False,
                 log_y=False,
                 log_x=False,
                 ratio_range="0.7,1.3",
                 custom_x_range=True,
                 x_axis_max=900,
                 x_axis_min=250,
                 custom_y_range=False,
                 y_axis_max=4000,
                 y_axis_min=0,
                 x_title="m_{hh} (GeV)",
                 y_title="a.u",
                 extra_pad=0,
                 norm_hists=False,
                 plot_name="plots/dihiggs_contributions_Mass%(mass)s_Width%(width)s_BM_scaled%(x)s".replace('.','p') % vars(),
                 label="",
                 norm_bins=False)
    
    total_approx_1 = sm.Clone()
    total_approx_1.Add(schannel_H)

    total_approx_2 = sm_like.Clone()
    total_approx_2.Add(schannel_H)
    
    hists = [total, schannel_H,total_approx_1]#, total_approx_2]
    titles = ['BSM total','S_{H}','S_{H}+SM']#, 'S_{H}+SM-like']
    
    plotting.CompareHists(hists=hists,
                 legend_titles=titles,
                 title="m_{H} = %(mass)s GeV,  #Gamma_{H} = %(width_val).2f GeV" % vars(),
                 ratio=True,
                 log_y=False,
                 log_x=False,
                 ratio_range="0.,3.",
                 custom_x_range=True,
                 x_axis_max=900,
                 x_axis_min=250,
                 custom_y_range=False,
                 y_axis_max=4000,
                 y_axis_min=0,
                 x_title="m_{hh} (GeV)",
                 y_title="a.u",
                 extra_pad=0,
                 norm_hists=False,
                 plot_name="plots/dihiggs_totals_Mass%(mass)s_Width%(width)s_BM_scaled%(x)s".replace('.','p') % vars(),
                 label="",
                 norm_bins=False) 


    total_sm_sub=total.Clone()
    total_sm_sub.Add(sm,-1)
    hists = [total_sm_sub, schannel_H]#, total_approx_2]
    titles = ['BSM-SM','S_{H}']#, 'S_{H}+SM-like']

    print total_sm_sub.GetMinimum()

    plotting.CompareHists(hists=hists,
                 legend_titles=titles,
                 title="m_{H} = %(mass)s GeV,  #Gamma_{H} = %(width_val).2f GeV" % vars(),
                 ratio=True,
                 log_y=False,
                 log_x=False,
                 ratio_range="0.,3.",
                 custom_x_range=True,
                 x_axis_max=900,
                 x_axis_min=250,
                 custom_y_range=False,
                 y_axis_max=4000,
                 y_axis_min=0,
                 x_title="m_{hh} (GeV)",
                 y_title="a.u",
                 extra_pad=0,
                 norm_hists=False,
                 plot_name="plots/dihiggs_totals_SM_subtracted_Mass%(mass)s_Width%(width)s_BM_scaled%(x)s".replace('.','p') % vars(),
                 label="",
                 norm_bins=False)   

    if x!='':
        lo_i, hi_i = plotting.GetRangeAroundMax(schannel_H) 

        val_used=schannel_H.Integral(lo_i, hi_i)
        val_actual=total_sm_sub.Integral(lo_i, hi_i)
        print val_used, val_actual, val_actual/val_used
