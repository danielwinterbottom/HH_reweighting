import ROOT

use_smeared_mass=True

#f1 = ROOT.TFile('outputs/output_W4p97918_1M.root')
f1 = ROOT.TFile('outputs/output_W4p97918_1M.root')
f2 = ROOT.TFile('outputs/output_W12_10000.root')
f3 = ROOT.TFile('outputs/output_W3_10000.root')
if use_smeared_mass: fout = ROOT.TFile('hhmass_compareWidths_smeared_output.root' % vars(),'RECREATE')
else: fout = ROOT.TFile('hhmass_compareWidths_output.root' % vars(),'RECREATE')

t1 = f1.Get('ntuple')
t2 = f2.Get('ntuple')
t3 = f3.Get('ntuple')

N1=t1.GetEntries()
N2=t2.GetEntries()
N3=t3.GetEntries()

weights = ['box', 'schannel_h', 'box_and_schannel_h_i', 'schannel_H', 'box_and_schannel_H_i', 'schannel_H_and_schannel_h_i']

var = 'hh_mass'
if use_smeared_mass: var = 'hh_mass_smear_improved'

for wt in weights:

  h1 = ROOT.TH1D('h1','',50,200,1000)
  h2 = ROOT.TH1D('h2','',50,200,1000)
  h3 = ROOT.TH1D('h3','',50,200,1000)

  t1.Draw('%(var)s>>h1' % vars(),'wt_%(wt)s' % vars(),'goff')
  h1 = t1.GetHistogram()
  t2.Draw('%(var)s>>h2' % vars(),'wt_%(wt)s' % vars(),'goff')
  h2 = t2.GetHistogram()
  t3.Draw('%(var)s>>h3' % vars(),'wt_%(wt)s' % vars(),'goff')
  h3 = t3.GetHistogram()
 
  h1.Scale(1./N1) 
  h2.Scale(1./N2) 
  h3.Scale(1./N3) 

  fout.cd()
  h1.Write('%(wt)s_Width4p97918' % vars()) 
  h2.Write('%(wt)s_Width12' % vars()) 
  h3.Write('%(wt)s_Width3' % vars()) 

  print wt, h1.Integral(-1,-1), h2.Integral(-1,-1), h3.Integral(-1,-1)
  del h1, h2, h3
