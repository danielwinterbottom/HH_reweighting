import ROOT
import math
from array import array
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help= 'LHE file to be converted')
parser.add_argument('--output', '-o', help= 'Name of output file',default='output.root')
parser.add_argument('--n_events', '-n', help= 'Maximum number of events to process', default=-1, type=int)
parser.add_argument('--no_weights', help= 'Do not store weights', action='store_true')
args = parser.parse_args()

# Open the LHE file for reading
lhe_filename = args.input
lhe_file = open(lhe_filename, "r")

# Create a ROOT TTree to store the event information
root_filename = args.output
root_file = ROOT.TFile(root_filename, "RECREATE")
tree = ROOT.TTree("ntuple", "Event Tree")

# Define variables to store event information
hh_mass  = array('f',[0])
hh_mass_smear  = array('f',[0])
hh_mass_smear_improved  = array('f',[0])
hh_mass_smear_4b  = array('f',[0])
hh_mass_smear_4b_improved  = array('f',[0])
wt_nom_out  = array('f',[0])
wt_box_out  = array('f',[0])
wt_schannel_h_out  = array('f',[0])
wt_box_and_schannel_h_i_out  = array('f',[0])
wt_schannel_H_out  = array('f',[0])
wt_schannel_H_alt_out  = array('f',[0])
wt_box_and_schannel_H_i_out  = array('f',[0])
wt_schannel_H_and_schannel_h_i_out  = array('f',[0])

hh_dphi  = array('f',[0])
hh_deta  = array('f',[0])
hh_dR  = array('f',[0])
hh_pt1  = array('f',[0])
hh_pt2  = array('f',[0])
hh_eta1  = array('f',[0])
hh_eta2  = array('f',[0])
hh_phi1  = array('f',[0])
hh_phi2  = array('f',[0])

# mbb resolution taken from Figure 5 here: https://arxiv.org/pdf/1912.06046.pdf
func1 = ROOT.TF1("func1","TMath::Gaus(x,0,0.12)",-5*0.12,5*0.12)
# mgamgam resolutiosn taken from page 3 here: https://arxiv.org/pdf/2310.01643.pdf
func2 = ROOT.TF1("func2","TMath::Gaus(x,0,0.01)",-5*0.01,5*0.01)
 
# create the branches and assign the fill-variables to them
tree.Branch("hh_mass",  hh_mass,  'hh_mass/F')
tree.Branch("hh_mass_smear",  hh_mass_smear,  'hh_mass_smear/F')
tree.Branch("hh_mass_smear_improved",  hh_mass_smear_improved,  'hh_mass_smear_improved/F')

tree.Branch("hh_mass_smear_4b",  hh_mass_smear_4b,  'hh_mass_smear_4b/F')
tree.Branch("hh_mass_smear_4b_improved",  hh_mass_smear_4b_improved,  'hh_mass_smear_4b_improved/F')

tree.Branch("hh_dphi",  hh_dphi,  'hh_dphi/F')
tree.Branch("hh_deta",  hh_deta,  'hh_deta/F')
tree.Branch("hh_dR",  hh_dR,  'hh_dR/F')
tree.Branch("hh_pt1",  hh_pt1,  'hh_pt1/F')
tree.Branch("hh_pt2",  hh_pt2,  'hh_pt2/F')
tree.Branch("hh_eta1",  hh_eta1,  'hh_eta1/F')
tree.Branch("hh_eta2",  hh_eta2,  'hh_eta2/F')
tree.Branch("hh_phi1",  hh_phi1,  'hh_phi1/F')
tree.Branch("hh_phi2",  hh_phi2,  'hh_phi2/F')

tree.Branch("wt_nom",  wt_nom_out,  'wt_nom/F')
if not args.no_weights:
    tree.Branch("wt_box",  wt_box_out,  'wt_box/F')
    tree.Branch("wt_schannel_h",  wt_schannel_h_out,  'wt_schannel_h/F')
    tree.Branch("wt_box_and_schannel_h_i",  wt_box_and_schannel_h_i_out,  'wt_box_and_schannel_h_i/F')
    tree.Branch("wt_schannel_H",  wt_schannel_H_out,  'wt_schannel_H/F')
    tree.Branch("wt_schannel_H_alt",  wt_schannel_H_alt_out,  'wt_schannel_H_alt/F')
    tree.Branch("wt_box_and_schannel_H_i",  wt_box_and_schannel_H_i_out,  'wt_box_and_schannel_H_i/F')
    tree.Branch("wt_schannel_H_and_schannel_h_i",  wt_schannel_H_and_schannel_h_i_out,  'wt_schannel_H_and_schannel_h_i/F')

# Parse the LHE file and fill the TTree
event_started = False
count=0
for line in lhe_file:
    line = line.strip()
    if line.startswith("<event>"):
        event_started = True
        if count % 10000 == 0: print "Processing %ith event" % count
        count+=1

        higgs_bosons=[]
        wt_box = None
        wt_box_and_schannel_H_1 = None
        wt_box_and_schannel_H_2 = None
        wt_box_and_schannel_h_1 = None
        wt_box_and_schannel_h_2 = None
        wt_all = None
        wt_schannel_h = None
        wt_schannel_H = None

    elif line.startswith("</event>"):
        # end of event
        # compute weights and store variables
        event_started = False

        # store di-Higgs mass
        if len(higgs_bosons)==2: 
            hh_mass[0]=(higgs_bosons[0]+higgs_bosons[1]).M()
            rand1 = 1.+func1.GetRandom() 
            rand2 = 1.+func2.GetRandom()
            rand3 = 1.+func1.GetRandom()

            higgs_smeared_1 = higgs_bosons[0]*rand1
            higgs_smeared_2 = higgs_bosons[1]*rand2

            higgs_smeared_2_bb = higgs_bosons[1]*rand3

            hh_mass_smear[0] = (higgs_smeared_1+higgs_smeared_2).M()
            hh_mass_smear_improved[0] = hh_mass_smear[0] - (higgs_smeared_1.M()-125.) - (higgs_smeared_2.M()-125.)
            hh_mass_smear_4b[0] = (higgs_smeared_1+higgs_smeared_2_bb).M()
            hh_mass_smear_4b_improved[0] = hh_mass_smear_4b[0] - (higgs_smeared_1.M()-125.) - (higgs_smeared_2_bb.M()-125.)

            #print rand1, rand2, higgs_smeared_1.M(), higgs_smeared_2.M(), hh_mass[0], hh_mass_smear[0], hh_mass_smear_improved[0]

            hh_dphi[0] = abs(higgs_bosons[0].DeltaPhi(higgs_bosons[1]))
            hh_deta[0] = abs(higgs_bosons[0].Eta() - higgs_bosons[1].Eta())
            hh_dR[0] = higgs_bosons[0].DeltaR(higgs_bosons[1])
            hh_phi1[0] = higgs_bosons[0].Phi()
            hh_phi2[0] = higgs_bosons[1].Phi()
            hh_eta1[0] = higgs_bosons[0].Eta()
            hh_eta2[0] = higgs_bosons[1].Eta()
            hh_pt1[0] = higgs_bosons[0].Pt()
            hh_pt2[0] = higgs_bosons[1].Pt()
        else: 
            h_mass[0]=-1
            hh_mass_smear[0]=-1
            hh_mass_smear_improved[0]=-1
            hh_mass_smear_4b[0]=-1
            hh_mass_smear_4b_improved[0]=-1
            hh_dphi[0] = -1 
            hh_deta[0] = -1
            hh_dR[0] = -1
            hh_phi1[0] = -1
            hh_phi2[0] = -1
            hh_eta1[0] = -1
            hh_eta2[0] = -1
            hh_pt1[0] = -1
            hh_pt2[0] = -1

        # compute weights which will all be multiplied by factors of sin(a) and cos(a) for now

        if not args.no_weights:

            wt_schannel_H_alt_out[0] = wt_schannel_H
    
            # get the s-channel weights by solving simultaneous equations for cases where kappa_lambda equals 1 and 100
            wt_schannel_h = float( ((wt_box_and_schannel_h_2-wt_box) - 10.*(wt_box_and_schannel_h_1-wt_box))/90. )
            wt_schannel_H = float( ((wt_box_and_schannel_H_2-wt_box) - 10.*(wt_box_and_schannel_H_1-wt_box))/90. )
    
    
            # get the inteference contributions
            wt_box_and_schannel_H_i = wt_box_and_schannel_H_1 - wt_box - wt_schannel_H
            wt_box_and_schannel_h_i = wt_box_and_schannel_h_1 - wt_box - wt_schannel_h
    
            wt_schannel_H_and_schannel_h_i = wt_all - wt_box - wt_schannel_H - wt_schannel_h - wt_box_and_schannel_H_i - wt_box_and_schannel_h_i
    
            # multiple weights to account for effect of mixing angles on top Yukawa couplings
            # by convention scale all weights to the scenario where both Yukawa couplings equal 1
    
            a = 0.785398
            ca = math.cos(a)
            sa = math.sin(a)
    
            wt_box_out[0] = float(wt_box/ca**4)
            wt_schannel_h_out[0] = float(wt_schannel_h/ca**2)
            wt_schannel_H_out[0] = float(wt_schannel_H/sa**2)
            wt_box_and_schannel_H_i_out[0] = float(wt_box_and_schannel_H_i/(ca**2*sa))
            wt_box_and_schannel_h_i_out[0] = float(wt_box_and_schannel_h_i/ca**3)
            wt_schannel_H_and_schannel_h_i_out[0] = float(wt_schannel_H_and_schannel_h_i/(ca*sa))

        tree.Fill()

    elif event_started:
        parts = line.split()
        if len(parts) == 6 and not line.startswith("<"):
            # read in nominal event weight
            wt_nom_out[0] = float(parts[2])
            #print 'wt_nom = ', wt_nom
        elif len(parts) == 13:
            # read in particle information
            if int(parts[0])==25:
                lvec=ROOT.TLorentzVector(float(parts[6]),float(parts[7]),float(parts[8]),float(parts[9]))
                higgs_bosons.append(lvec)
        elif line.startswith("<wgt id="):
            # read in weights for reweighting
            if line.startswith("<wgt id=\'box\'>"):
                wt_box=float(parts[2])
                #print 'wt_box = ', wt_box
            elif line.startswith("<wgt id=\'box_and_schannel_h_1\'>"):
                wt_box_and_schannel_h_1=float(parts[2])
                #print 'wt_box_and_schannel_h_1 = ', wt_box_and_schannel_h_1
            elif line.startswith("<wgt id=\'box_and_schannel_h_2\'>"):
                wt_box_and_schannel_h_2=float(parts[2])
                #print 'wt_box_and_schannel_h_2 = ', wt_box_and_schannel_h_2
            elif line.startswith("<wgt id=\'box_and_schannel_H_1\'>"):
                wt_box_and_schannel_H_1=float(parts[2])
                #print 'wt_box_and_schannel_H_1 = ', wt_box_and_schannel_H_1
            elif line.startswith("<wgt id=\'box_and_schannel_H_2\'>"):
                wt_box_and_schannel_H_2=float(parts[2])
                #print 'wt_box_and_schannel_H_2 = ', wt_box_and_schannel_H_2
            elif line.startswith("<wgt id=\'all\'>"):
                wt_all=float(parts[2])
                #print 'wt_all = ', wt_all
            elif line.startswith("<wgt id=\'schannel_H\'>"):
                wt_schannel_H=float(parts[2])
                #print 'wt_schannel_H = ', wt_schannel_H

    if count>args.n_events and args.n_events>=0: break

# Write the TTree to the ROOT file
root_file.Write()
root_file.Close()
lhe_file.close()
