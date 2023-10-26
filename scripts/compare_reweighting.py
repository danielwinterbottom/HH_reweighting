import ROOT
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use_smeared_mass', help= 'Use di-Higgs masses with experimental smearing', type=int, default=0)
parser.add_argument('--width','-w', help= 'The width used for generating the reweighted templates (changes the input file names)', default='4p97918')
args = parser.parse_args()


BM='600'
width=args.width
use_smeared_mass=args.use_smeared_mass

f1 = ROOT.TFile('outputs/output_W%(width)s_1M.root' % vars())
f2 = ROOT.TFile('outputs/output_BM_10000.root')
f3 = ROOT.TFile('outputs/output_SM_10000.root')
if use_smeared_mass==1: fout = ROOT.TFile('hhmass_output_BM%(BM)s_Width%(width)s_smeared.root' % vars(),'RECREATE')
elif use_smeared_mass==2: fout = ROOT.TFile('hhmass_output_BM%(BM)s_Width%(width)s_smeared_4b.root' % vars(),'RECREATE')
else: fout = ROOT.TFile('hhmass_output_BM%(BM)s_Width%(width)s.root' % vars(),'RECREATE')

t1 = f1.Get('ntuple')
t2 = f2.Get('ntuple')
t3 = f3.Get('ntuple')

N1=t1.GetEntries()
N2=t2.GetEntries()
N3=t3.GetEntries()

bins = '(50,200,1000)'

BM_params = {}

sm_lambda = 31.803878252 

BM_params['600'] = {
  'a12': 0.1708,
  'kap111': 30.18578,
  'kap112': 167.5027,
  'width': 4.979180, 
}

params = BM_params[BM]

width_sf=float(width.replace('p','.'))/params['width']

kappa_h_t = math.cos(params['a12'])
kappa_H_t = math.sin(params['a12'])

kappa_h_lam = params['kap111'] / sm_lambda
kappa_H_lam = params['kap112'] / sm_lambda

print 'Re-weighting to kappa_h_t = %.3f, kappa_H_t = %.3f, kappa_h_lam = %.3f, kappa_H_lam = %.3f' % (kappa_h_t, kappa_H_t, kappa_h_lam, kappa_H_lam )

h1 = ROOT.TH1D('h1','',50,200,1000)
h2_box = ROOT.TH1D('h2_box','',50,200,1000)
h2_schannel_h = ROOT.TH1D('h2_schannel_h','',50,200,1000)
h2_schannel_H = ROOT.TH1D('h2_schannel_H','',50,200,1000)
h2_box_and_schannel_H_i = ROOT.TH1D('h2_box_and_schannel_H_i','',50,200,1000)
h2_box_and_schannel_h_i = ROOT.TH1D('h2_box_and_schannel_h_i','',50,200,1000)
h2_schannel_H_and_schannel_h_i = ROOT.TH1D('h2_schannel_H_and_schannel_h_i','',50,200,1000)
h3 = ROOT.TH1D('h3','',50,200,1000)

var='hh_mass'
if use_smeared_mass==1:   var = 'hh_mass_smear_improved'
elif use_smeared_mass==2: var = 'hh_mass_smear_4b_improved'

# get SM distribution generated directly
t3.Draw('%(var)s>>h3'  % vars(),'wt_nom', 'goff')
h3 = t3.GetHistogram()
h3.Scale(1./N3)

# get sample generated directly for the benchmark
t2.Draw('%(var)s>>h1' % vars(),'wt_nom', 'goff')
h1 = t2.GetHistogram()
h1.Scale(1./N2)

# get individual templates
t1.Draw('%(var)s>>h2_box' % vars(),'wt_box', 'goff')
h2_box = t1.GetHistogram()
t1.Draw('%(var)s>>h2_schannel_h' % vars(),'wt_schannel_h' % vars(), 'goff')
h2_schannel_h = t1.GetHistogram()
t1.Draw('%(var)s>>h2_box_and_schannel_h_i' % vars(),'wt_box_and_schannel_h_i', 'goff')
h2_box_and_schannel_h_i = t1.GetHistogram()
t1.Draw('%(var)s>>h2_schannel_H' % vars(),'wt_schannel_H', 'goff')
h2_schannel_H = t1.GetHistogram()
t1.Draw('%(var)s>>h2_box_and_schannel_H_i' % vars(),'wt_box_and_schannel_H_i', 'goff')
h2_box_and_schannel_H_i = t1.GetHistogram()
t1.Draw('%(var)s>>h2_schannel_H_and_schannel_h_i' % vars(),'wt_schannel_H_and_schannel_h_i', 'goff')
h2_schannel_H_and_schannel_h_i = t1.GetHistogram()

h2_box.Scale(1./N1)
h2_schannel_h.Scale(1./N1)
h2_box_and_schannel_h_i.Scale(1./N1)
h2_schannel_H.Scale(1./N1)
h2_box_and_schannel_H_i.Scale(1./N1)
h2_schannel_H_and_schannel_h_i.Scale(1./N1)

h_sm = h2_box.Clone()
h_sm.Add(h2_schannel_h)
h_sm.Add(h2_box_and_schannel_h_i)

box_SF                         = kappa_h_t**4
schannel_H_SF                  = kappa_H_t**2*kappa_H_lam**2*width_sf
schannel_h_SF                  = kappa_h_t**2*kappa_h_lam**2
box_and_schannel_h_i_SF        = kappa_h_t**3*kappa_h_lam
box_and_schannel_H_i_SF        = kappa_h_t**2*kappa_H_t*kappa_H_lam
schannel_H_and_schannel_h_i_SF = kappa_H_t*kappa_H_lam*kappa_h_t*kappa_h_lam

print '\nScale factors applied:'
print 'width SF = %.3f' % width_sf
print 'box_SF = %.3f' % box_SF 
print 'schannel_H_SF = %.3f' % schannel_H_SF 
print 'schannel_h_SF = %.3f' % schannel_h_SF
print 'box_and_schannel_H_i_SF = %.3f' % box_and_schannel_H_i_SF 
print 'box_and_schannel_h_i_SF = %.3f' % box_and_schannel_h_i_SF 
print 'schannel_H_and_schannel_h_i_SF = %.3f' % schannel_H_and_schannel_h_i_SF

h2_box.Scale(box_SF)
h2_schannel_H.Scale(schannel_H_SF)
h2_schannel_h.Scale(schannel_h_SF)
h2_box_and_schannel_H_i.Scale(box_and_schannel_H_i_SF)
h2_box_and_schannel_h_i.Scale(box_and_schannel_h_i_SF)
h2_schannel_H_and_schannel_h_i.Scale(schannel_H_and_schannel_h_i_SF)

h_sm_like = h2_box.Clone()
h_sm_like.Add(h2_schannel_h)
h_sm_like.Add(h2_box_and_schannel_h_i)

print '\nIntegral computed from BM model = %.5f' % h1.Integral()

h2 = h2_box.Clone()
h2.Add(h2_schannel_H)
h2.Add(h2_schannel_h)
h2.Add(h2_box_and_schannel_H_i)
h2.Add(h2_box_and_schannel_h_i)
h2.Add(h2_schannel_H_and_schannel_h_i)

print '\nIntegrals after applying SFs:'

print 'box = %.5f' % h2_box.Integral(-1,-1)
print 'schannel_H = %.5f' % h2_schannel_H.Integral(-1,-1)
print 'schannel_h = %.5f' % h2_schannel_h.Integral(-1,-1)
print 'box_and_schannel_H_i = %.5f' % h2_box_and_schannel_H_i.Integral(-1,-1)
print 'box_and_schannel_h_i = %.5f' % h2_box_and_schannel_h_i.Integral(-1,-1)
print 'schannel_H_and_schannel_h_i = %.5f' % h2_schannel_H_and_schannel_h_i.Integral(-1,-1)
print 'total = %.5f' % h2.Integral(-1,-1)

fout.cd()
h1.Write('BM_generated')
h_sm.Write('SM')
h_sm_like.Write('SM_like')
h2_box.Write('BM_box')
h2_schannel_H.Write('BM_schannel_H')
h2_schannel_h.Write('BM_schannel_h')
h2_box_and_schannel_H_i.Write('BM_box_and_schannel_H_i')
h2_box_and_schannel_h_i.Write('BM_box_and_schannel_h_i')
h2_schannel_H_and_schannel_h_i.Write('BM_schannel_H_and_schannel_h_i')
h2.Write('BM_total')
h3.Write('SM_generated')

h_approx = h_sm.Clone()
h_approx.Add(h2_schannel_H)

h_approx.Write('BM_total_approx')

h_approx_2 = h_sm_like.Clone()
h_approx_2.Add(h2_schannel_H)

h_approx_2.Write('BM_total_approx_2')

fout.Close()
f1.Close()
f2.Close()
