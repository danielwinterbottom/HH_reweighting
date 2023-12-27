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

weights=[]

# first determine names of all weights
for line in lhe_file:

    line = line.strip()
    if line.startswith('<weight id='):
        if '\'' in line: weights.append(line.split('\'')[1])
    if line.startswith("<event>"):
      break # break loop once we have read all weights


# go back to start of the file
lhe_file.seek(0)

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
hh_mass_smear_2b2ta  = array('f',[0])
wt_nom_out  = array('f',[0])

weights_map = {}

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
func3 = ROOT.TF1("func3","TMath::Gaus(x,0,0.1)",-5*0.1,5*0.1)
 
# create the branches and assign the fill-variables to them
tree.Branch("hh_mass",  hh_mass,  'hh_mass/F')
tree.Branch("hh_mass_smear",  hh_mass_smear,  'hh_mass_smear/F')
tree.Branch("hh_mass_smear_improved",  hh_mass_smear_improved,  'hh_mass_smear_improved/F')

tree.Branch("hh_mass_smear_2b2ta",  hh_mass_smear_2b2ta,  'hh_mass_smear_2b2ta/F')
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

for wt in weights:
    if wt not in weights_map: 
        weights_map[wt] = array('f',[0])
        tree.Branch("wt_%(wt)s" % vars(), weights_map[wt], 'wt_%(wt)s/F' % vars())
  
# Parse the LHE file and fill the TTree
event_started = False
count=0

# go back to start of the file
lhe_file.seek(0)

for line in lhe_file:
    line = line.strip()
    if line.startswith("<event>"):
        event_started = True
        if count % 10000 == 0: print "Processing %ith event" % count
        count+=1

        higgs_bosons=[]

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
            rand4 = 1.+func3.GetRandom()

            higgs_smeared_1 = higgs_bosons[0]*rand1
            higgs_smeared_2 = higgs_bosons[1]*rand2

            higgs_smeared_2_bb = higgs_bosons[1]*rand3

            hh_mass_smear[0] = (higgs_smeared_1+higgs_smeared_2).M()
            hh_mass_smear_improved[0] = hh_mass_smear[0] - (higgs_smeared_1.M()-125.) - (higgs_smeared_2.M()-125.)
            hh_mass_smear_4b[0] = (higgs_smeared_1+higgs_smeared_2_bb).M()
            hh_mass_smear_4b_improved[0] = hh_mass_smear_4b[0] - (higgs_smeared_1.M()-125.) - (higgs_smeared_2_bb.M()-125.)

            hh_mass_smear_2b2ta[0] = hh_mass[0]*rand4

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
            hh_mass_smear_2b2ta[0]=-1
            hh_dphi[0] = -1 
            hh_deta[0] = -1
            hh_dR[0] = -1
            hh_phi1[0] = -1
            hh_phi2[0] = -1
            hh_eta1[0] = -1
            hh_eta2[0] = -1
            hh_pt1[0] = -1
            hh_pt2[0] = -1

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
            if line.startswith("<wgt id=\'"):
                name = parts[1].split('\'')[1]
                if name not in weights: continue
                val = float(parts[2])
                weights_map[name][0] = val

    if count>args.n_events and args.n_events>=0: break

# Write the TTree to the ROOT file
root_file.Write()
root_file.Close()
lhe_file.close()
