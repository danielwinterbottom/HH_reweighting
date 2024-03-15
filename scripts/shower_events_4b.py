import pythia8
import argparse
import ROOT
from array import array
import random
import math
import numpy as np


class SmearBJet():
    def __init__(self):
        ## take resolutions vs pT from Fig 3 in https://arxiv.org/pdf/1912.06046.pdf (left plot for DNN)
        #x_vals = array('d',[30.,50.,70.,90.,125.,175.,224.,275.,350.])
        #y_vals = array('d',[0.111,0.106,0.094,0.087,0.082,0.076,0.073,0.068,0.064])
        #self.res_graph = ROOT.TGraph(len(x_vals), x_vals, y_vals)
        #self.min_pt = x_vals[0]
        #self.max_pt = x_vals[-1]
        #self.func = ROOT.TF1("func","TMath::Gaus(x,0,[0])",-0.5,0.5)
        # for pT resolution
        # take resolution from H mass resolution from Figure 5 here: https://arxiv.org/pdf/1912.06046.pdf (using DNN)
        # since we smear jets seperatly we need to scale this resolution up by sqrt(2) [assuming energy is split 50-50 by 2 jets from Higgs decay]
        # = 15.4/124.6*sqrt(2) = 0.18
        # but smearing of jet eta and phi below also affects mass resolution
        # so we smear first by phi and eta and then adjust jet E smearing to give correct "width" for h125 mass peaks (~15.4 GeV)
        # approximatly 15% smearing required
        self.func = ROOT.TF1("func","TMath::Gaus(x,0,0.15)",-1,1)
        # for phi resolution
        self.func_phi = ROOT.TF1("func_phi","TMath::Gaus(x,0,0.05)",-1,1)
        # for eta resolution
        self.func_eta = ROOT.TF1("func_eta","TMath::Gaus(x,0,0.05)",-1,1)

    def Smear(self,j):
        #pt = max(min(j.Pt(),self.max_pt), self.min_pt)
        #sigma = self.res_graph.Eval(pt)
        #self.func.SetParameter(0,sigma)
        rand = 1.+self.func.GetRandom()
        rand_dphi = self.func_phi.GetRandom()
        rand_deta = self.func_eta.GetRandom()


        j_smeared = ROOT.TLorentzVector(j)

        phi = j_smeared.Phi()
        new_phi = ROOT.TVector2.Phi_mpi_pi(phi + rand_dphi)
        new_eta = j_smeared.Eta() + rand_deta
        new_theta = 2 * math.atan(math.exp(-new_eta))
        j_smeared.SetPhi(new_phi)
        j_smeared.SetTheta(new_theta)
        j_smeared *= rand

        return j_smeared   

class SmearGamma():
    def __init__(self):
        # energy resolution from https://arxiv.org/pdf/2012.06888.pdf Fig 11
        self.func = ROOT.TF1("func","TMath::Gaus(x,0,0.02)",-1,1)
        # eta and phi resolutions from https://cds.cern.ch/record/349375/files/ECAL_TDR.pdf page 7 assuming E ~60 GeV
        # also page 3 here: https://www.physics.smu.edu/web/research/preprints/SMU-HEP-09-16.pdf
        # for phi resolution
        self.func_phi = ROOT.TF1("func_phi","TMath::Gaus(x,0,0.005)",-1,1)
        # for eta resolution
        self.func_eta = ROOT.TF1("func_eta","TMath::Gaus(x,0,0.005)",-1,1)

    def Smear(self,g):
        rand = 1.+self.func.GetRandom()
        rand_dphi = self.func_phi.GetRandom()
        rand_deta = self.func_eta.GetRandom()

        g_smeared = ROOT.TLorentzVector(g)

        phi = g_smeared.Phi()
        new_phi = ROOT.TVector2.Phi_mpi_pi(phi + rand_dphi)
        new_eta = g_smeared.Eta() + rand_deta
        new_theta = 2 * math.atan(math.exp(-new_eta))
        g_smeared.SetPhi(new_phi)
        g_smeared.SetTheta(new_theta)
        g_smeared *= rand

        return g_smeared         

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help= 'LHE file to be converted')
parser.add_argument('--output', '-o', help= 'Name of output file',default='pythia_output.root')
parser.add_argument('--cmnd_file', '-c', help= 'Pythia8 command file')
parser.add_argument('--n_events', '-n', help= 'Maximum number of events to process', default=-1, type=int)
parser.add_argument('--n_skip', '-s', help= 'skip n_events*n_skip', default=0, type=int)

args = parser.parse_args()

# Create a ROOT TTree to store the event information
root_filename = args.output
root_file = ROOT.TFile(root_filename, "RECREATE")
tree = ROOT.TTree("ntuple", "Event Tree")

pythia = pythia8.Pythia("", False)
pythia.readFile(args.cmnd_file)

pythia.readString("Beams:frameType = 4")
pythia.readString("Beams:LHEF = %s" % args.input)
pythia.init()

wt_nom  = array('f',[0])
hh_mass  = array('f',[0])
hh_pT  = array('f',[0])
h1_pT = array('f',[0])
h2_pT = array('f',[0])
h1_eta = array('f',[0])
h2_eta = array('f',[0])
hh_dR = array('f',[0])
hh_dphi = array('f',[0])
hh_eta = array('f',[0])

hh_mass_smear  = array('f',[0])
hh_pT_smear  = array('f',[0])
h1_pT_smear = array('f',[0])
h2_pT_smear = array('f',[0])
h1_eta_smear = array('f',[0])
h2_eta_smear = array('f',[0])
hh_dR_smear = array('f',[0])
hh_dphi_smear = array('f',[0])
hh_eta_smear = array('f',[0])
h1_mass_smear = array('f',[0])
h2_mass_smear = array('f',[0])

b1_pT = array('f',[0])
b2_pT = array('f',[0])
b3_pT = array('f',[0])
b4_pT = array('f',[0])
b1_pT_smear = array('f',[0])
b2_pT_smear = array('f',[0])
b3_pT_smear = array('f',[0])
b4_pT_smear = array('f',[0])
b1_eta = array('f',[0])
b2_eta = array('f',[0])
b3_eta = array('f',[0])
b4_eta = array('f',[0])
b1_eta_smear = array('f',[0])
b2_eta_smear = array('f',[0])
b3_eta_smear = array('f',[0])
b4_eta_smear = array('f',[0])


hh_mass_smear_improved  = array('f',[0])
hh_mass_smear_improved_2  = array('f',[0])
hh_pT_smear_improved  = array('f',[0])
h1_pT_smear_improved  = array('f',[0])
h2_pT_smear_improved  = array('f',[0])

hh_mass_smear_bbgg = array('f',[0])
hh_mass_smear_bbgg_improved = array('f',[0])

hh_mass_smear_improved_MX = array('f',[0])
hh_mass_smear_bbgg_improved_MX = array('f',[0])

alphas_out = array('f',[0])

higgs_1 = ROOT.std.vector('double')([0.0]*5)
higgs_2 = ROOT.std.vector('double')([0.0]*5)


tree.Branch("wt_nom",  wt_nom,  'wt_nom/F')
tree.Branch("hh_mass",  hh_mass,  'hh_mass/F')
tree.Branch("hh_pT",  hh_pT,  'hh_pT/F')
tree.Branch("h1_pT",  h1_pT,  'h1_pT/F')
tree.Branch("h2_pT",  h2_pT,  'h2_pT/F')
tree.Branch("h1_eta",  h1_eta,  'h1_eta/F')
tree.Branch("h2_eta",  h2_eta,  'h2_eta/F')
tree.Branch("hh_dR",  hh_dR,  'hh_dR/F')
tree.Branch("hh_dphi",  hh_dphi,  'hh_dphi/F')
tree.Branch("hh_eta",  hh_eta,  'hh_eta/F')

tree.Branch("hh_mass_smear",  hh_mass_smear,  'hh_mass_smear/F')
tree.Branch("hh_pT_smear",  hh_pT_smear,  'hh_pT_smear/F')
tree.Branch("h1_pT_smear",  h1_pT_smear,  'h1_pT_smear/F')
tree.Branch("h2_pT_smear",  h2_pT_smear,  'h2_pT_smear/F')
tree.Branch("h1_eta_smear",  h1_eta_smear,  'h1_eta_smear/F')
tree.Branch("h2_eta_smear",  h2_eta_smear,  'h2_eta_smear/F')
tree.Branch("hh_dR_smear",  hh_dR_smear,  'hh_dR_smear/F')
tree.Branch("hh_dphi_smear",  hh_dphi_smear,  'hh_dphi_smear/F')
tree.Branch("hh_eta_smear",  hh_eta_smear,  'hh_eta_smear/F')
tree.Branch("h1_mass_smear",  h1_mass_smear,  'h1_mass_smear/F')
tree.Branch("h2_mass_smear",  h2_mass_smear,  'h2_mass_smear/F')

tree.Branch("hh_mass_smear_improved",  hh_mass_smear_improved,  'hh_mass_smear_improved/F')
tree.Branch("hh_mass_smear_improved_2",  hh_mass_smear_improved_2,  'hh_mass_smear_improved_2/F')
tree.Branch("hh_pT_smear_improved",  hh_pT_smear_improved,  'hh_pT_smear_improved/F')
tree.Branch("h1_pT_smear_improved",  h1_pT_smear_improved,  'h1_pT_smear_improved/F')
tree.Branch("h2_pT_smear_improved",  h2_pT_smear_improved,  'h2_pT_smear_improved/F')


tree.Branch("b1_pT", b1_pT, 'b1_pT/F')
tree.Branch("b2_pT", b2_pT, 'b2_pT/F')
tree.Branch("b3_pT", b3_pT, 'b3_pT/F')
tree.Branch("b4_pT", b4_pT, 'b4_pT/F')
tree.Branch("b1_pT_smear", b1_pT_smear, 'b1_pT_smear/F')
tree.Branch("b2_pT_smear", b2_pT_smear, 'b2_pT_smear/F')
tree.Branch("b3_pT_smear", b3_pT_smear, 'b3_pT_smear/F')
tree.Branch("b4_pT_smear", b4_pT_smear, 'b4_pT_smear/F')
tree.Branch("b1_eta", b1_eta, 'b1_eta/F')
tree.Branch("b2_eta", b2_eta, 'b2_eta/F')
tree.Branch("b3_eta", b3_eta, 'b3_eta/F')
tree.Branch("b4_eta", b4_eta, 'b4_eta/F')
tree.Branch("b1_eta_smear", b1_eta_smear, 'b1_eta_smear/F')
tree.Branch("b2_eta_smear", b2_eta_smear, 'b2_eta_smear/F')
tree.Branch("b3_eta_smear", b3_eta_smear, 'b3_eta_smear/F')
tree.Branch("b4_eta_smear", b4_eta_smear, 'b4_eta_smear/F')

tree.Branch("hh_mass_smear_bbgg_improved",  hh_mass_smear_bbgg_improved,  'hh_mass_smear_bbgg_improved/F')
tree.Branch("hh_mass_smear_bbgg",  hh_mass_smear_bbgg,  'hh_mass_smear_bbgg/F') 

tree.Branch("hh_mass_smear_improved_MX",  hh_mass_smear_improved_MX,  'hh_mass_smear_improved_MX/F')
tree.Branch("hh_mass_smear_bbgg_improved_MX",  hh_mass_smear_bbgg_improved_MX,  'hh_mass_smear_bbgg_improved_MX/F')

tree.Branch("alphas",  alphas_out,  'alphas/F')  
tree.Branch("higgs_1",  higgs_1)
tree.Branch("higgs_2",  higgs_2)

weights_map = {}
weight_names = []
# first get names of all weights from the first event to define tree branches
pythia.LHAeventSkip(args.n_skip*args.n_events)
pythia.next()
for i in range (pythia.infoPython().numberOfWeights()):
    name = pythia.infoPython().weightNameByIndex(i).replace('AUX_','')
    if name[0].isdigit(): continue
    weight_names.append(name)

for wt in weight_names:
    if wt not in weights_map:
        weights_map[wt] = array('f',[0])
        tree.Branch("wt_LHE_%(wt)s" % vars(), weights_map[wt], 'wt_LHE_%(wt)s/F' % vars())    


# initialise smearing

smear = SmearBJet()
smear_gam = SmearGamma()

def sort_by_pt(vector):
    return vector.Pt()

stopGenerating = False
count = 0
while not stopGenerating:

    stopGenerating = pythia.infoPython().atEndOfFile()
    if args.n_events>0 and count+1 >= args.n_events: stopGenerating = True

    wt_nom[0] = pythia.infoPython().weight()

    for i in range (pythia.infoPython().numberOfWeights()):
        name =  pythia.infoPython().weightNameByIndex(i).replace('AUX_','')
        if name not in weights_map: continue
        val =  pythia.infoPython().weightValueByIndex(i)
        weights_map[name][0] = val

    higgs_bosons_first = []
    higgs_bosons_last = []
    higgs_decay_prods = []

    for i, part in enumerate(pythia.event):
        pdgid = part.id()
   
        if pdgid != 25: continue
        lastCopy = len(part.daughterList()) == 2 and abs(pythia.event[part.daughterList()[0]].id()) == 5 and abs(pythia.event[part.daughterList()[1]].id()) == 5 
        firstCopy = True not in [pythia.event[p].id() == 25 for p in part.motherList()]
        if not (firstCopy or lastCopy): continue
        lvec = ROOT.TLorentzVector(part.px(),part.py(),part.pz(),part.e())
        if firstCopy: higgs_bosons_first.append(lvec)
        if lastCopy:
            lvec = ROOT.TLorentzVector(part.px(),part.py(),part.pz(),part.e())
            higgs_bosons_last.append(lvec)
            decay_prods = part.daughterList()
            decay_prods_lvec = []    
            for b_i in decay_prods:
                b_jet = pythia.event[b_i] 
                lvec = ROOT.TLorentzVector(b_jet.px(),b_jet.py(),b_jet.pz(),b_jet.e())   
                decay_prods_lvec.append(lvec) 
            higgs_decay_prods.append(decay_prods_lvec)    

    if len(higgs_decay_prods) == 2 and len(higgs_decay_prods[0]) == 2 and len(higgs_decay_prods[1]) == 2:
        j1 = higgs_decay_prods[0][0] 
        j2 = higgs_decay_prods[0][1] 
        j3 = higgs_decay_prods[1][0] 
        j4 = higgs_decay_prods[1][1]  

        hh_mass[0] = (j1+j2+j3+j4).M()

        unsorted_jets = [j1,j2,j3,j4]
        sorted_jets = sorted(unsorted_jets, key=sort_by_pt, reverse=True)

        b1_pT[0] = sorted_jets[0].Pt()
        b2_pT[0] = sorted_jets[1].Pt()
        b3_pT[0] = sorted_jets[2].Pt()
        b4_pT[0] = sorted_jets[3].Pt()

        b1_eta[0] = sorted_jets[0].Rapidity()
        b2_eta[0] = sorted_jets[1].Rapidity()
        b3_eta[0] = sorted_jets[2].Rapidity()
        b4_eta[0] = sorted_jets[3].Rapidity()

        #print '!!!!!!'
        #print hh_mass[0], (higgs_bosons_last[0]+higgs_bosons_last[1]).M() 

        hh_pT[0] = (j1+j2+j3+j4).Pt()
        if (j1+j2).Pt() >= (j3+j4).Pt():
            h1_pT[0] = (j1+j2).Pt()
            h2_pT[0] = (j3+j4).Pt()
            h1_eta[0] = (j1+j2).Rapidity()
            h2_eta[0] = (j3+j4).Rapidity()
        else:    
            h1_pT[0] = (j3+j4).Pt()
            h2_pT[0] = (j1+j2).Pt()
            h1_eta[0] = (j3+j4).Rapidity()
            h2_eta[0] = (j1+j2).Rapidity()

        hh_dR[0] = (j3+j4).DeltaR(j1+j2)
        hh_dphi[0] = abs((j3+j4).DeltaPhi(j1+j2))
        hh_eta[0] = (j1+j2+j3+j4).Rapidity()        

        j1_smear = smear.Smear(j1)
        j2_smear = smear.Smear(j2)
        j3_smear = smear.Smear(j3)
        j4_smear = smear.Smear(j4)

        unsorted_smeared_jets = [j1_smear,j2_smear,j3_smear,j4_smear]
        sorted_smeared_jets = sorted(unsorted_smeared_jets, key=sort_by_pt, reverse=True)        

        b1_pT_smear[0] = sorted_smeared_jets[0].Pt()
        b2_pT_smear[0] = sorted_smeared_jets[1].Pt()
        b3_pT_smear[0] = sorted_smeared_jets[2].Pt()
        b4_pT_smear[0] = sorted_smeared_jets[3].Pt()

        b1_eta_smear[0] = sorted_smeared_jets[0].Rapidity()
        b2_eta_smear[0] = sorted_smeared_jets[1].Rapidity()
        b3_eta_smear[0] = sorted_smeared_jets[2].Rapidity()
        b4_eta_smear[0] = sorted_smeared_jets[3].Rapidity()

        g1_smear = smear_gam.Smear(j1)
        g2_smear = smear_gam.Smear(j2)
        g3_smear = smear_gam.Smear(j3)
        g4_smear = smear_gam.Smear(j4)

        h1_smear = j1_smear+j2_smear
        h2_smear = j3_smear+j4_smear
        h1_smear_imp = h1_smear*(125./h1_smear.M())
        h2_smear_imp = h2_smear*(125./h2_smear.M())

        hh_mass_smear[0] = (h1_smear+h2_smear).M()
        hh_pT_smear[0] = (h1_smear+h2_smear).Pt()

        hh_mass_smear_improved[0] = (h1_smear_imp+h2_smear_imp).M()


        def MassDiff(h1,h2):
            return (h1.M()-125.)**2 + (h2.M()-125.)**2
            #return abs(h1.M() - h2.M())

        pairs = [
            (j1_smear+j2_smear,j3_smear+j4_smear),    
            (j1_smear+j3_smear,j2_smear+j4_smear),
            (j1_smear+j4_smear,j2_smear+j3_smear)
        ]

        # realistic pairings:

        min_mass_diff = None
        best_pairs = None
        for pair in pairs:
            massdiff = MassDiff(pair[0], pair[1])
            if (not min_mass_diff) or massdiff<min_mass_diff:
                min_mass_diff = massdiff
                best_pairs = pair 

        h1_smear_imp_2 = best_pairs[0]*(125./best_pairs[0].M())
        h2_smear_imp_2 = best_pairs[1]*(125./best_pairs[1].M())          

        hh_mass_smear_improved_2[0] = (h1_smear_imp_2+h2_smear_imp_2).M()
        

        hh_pT_smear_improved[0] = (h1_smear_imp+h2_smear_imp).Pt()

        hh_mass_smear_improved_MX[0] = hh_mass_smear[0] - (h1_smear.M()-125.) - (h2_smear.M()-125.)


        h1_smear_gg = g1_smear+g2_smear
        h2_smear_gg = g3_smear+g4_smear


        h1_smear_gg_imp = h1_smear_gg*(125./h1_smear_gg.M())
        h2_smear_gg_imp = h2_smear_gg*(125./h2_smear_gg.M())

        # for bbgg smearing, randomly choose which higgs decays to each final state
        if random.random() < 0.5:
            hh_mass_smear_bbgg[0] = (h1_smear+h2_smear_gg).M()
            hh_mass_smear_bbgg_improved[0] = (h1_smear_imp+h2_smear_gg_imp).M()
            hh_mass_smear_bbgg_improved_MX[0] = hh_mass_smear_bbgg[0] - (h1_smear.M()-125.) - (h2_smear_gg.M()-125.)
        else:
            hh_mass_smear_bbgg[0] = (h1_smear_gg+h2_smear).M()
            hh_mass_smear_bbgg_improved[0] = (h1_smear_gg_imp+h2_smear_imp).M()
            hh_mass_smear_bbgg_improved_MX[0] = hh_mass_smear_bbgg[0] - (h1_smear_gg.M()-125.) - (h2_smear.M()-125.)            

        if h1_smear.Pt() >= h2_smear.Pt():
            h1_mass_smear[0] = h1_smear.M()
            h2_mass_smear[0] = h2_smear.M()
            h1_pT_smear[0] = h1_smear.Pt()
            h2_pT_smear[0] = h2_smear.Pt()
            h1_eta_smear[0] = h1_smear.Rapidity()
            h2_eta_smear[0] = h2_smear.Rapidity()

            h1_pT_smear_improved[0] = h1_smear_imp.Pt()
            h2_pT_smear_improved[0] = h2_smear_imp.Pt()
   
        else:  
            h1_mass_smear[0] = h2_smear.M()  
            h2_mass_smear[0] = h1_smear.M()  
            h1_pT_smear[0] = h2_smear.Pt()
            h2_pT_smear[0] = h1_smear.Pt()
            h1_eta_smear[0] = h2_smear.Rapidity()
            h2_eta_smear[0] = h1_smear.Rapidity()
            h1_pT_smear_improved[0] = h2_smear_imp.Pt()
            h2_pT_smear_improved[0] = h1_smear_imp.Pt()

        hh_dR_smear[0] = h2_smear.DeltaR(h1_smear)
        hh_dphi_smear[0] = abs(h2_smear.DeltaPhi(h1_smear))
        hh_eta_smear[0] = (h1_smear+h2_smear).Rapidity()
        

        #print hh_mass[0], hh_mass_smear[0], hh_mass_smear_improved[0]  

    if len(higgs_bosons_first) == 2:
        # need to shift masses to 125 GeV, otherwise we get lots of errors
        if higgs_bosons_first[0].M() != 125.: higgs_bosons_first[0].SetE((higgs_bosons_first[0].P()**2+125.**2)**.5)
        if higgs_bosons_first[1].M() != 125.: higgs_bosons_first[1].SetE((higgs_bosons_first[1].P()**2+125.**2)**.5)
        parts = [
          [25., higgs_bosons_first[0].E(), higgs_bosons_first[0].Px(), higgs_bosons_first[0].Py(), higgs_bosons_first[0].Pz()],
          [25., higgs_bosons_first[1].E(), higgs_bosons_first[1].Px(), higgs_bosons_first[1].Py(), higgs_bosons_first[1].Pz()],
        ]

        alphas = pythia.infoPython().alphaS()
        higgs_1.assign(parts[0])
        higgs_2.assign(parts[1])
        alphas_out[0] = alphas 

    else:
        higgs_1.assign([0.0]*5)
        higgs_2.assign([0.0]*5)
        alphas_out[0] = 0.118
        

    if hh_mass[0] >0:
        tree.Fill()

    count += 1
    if count % 10000 == 0: 
        print 'Processed %i events' % count
        root_file.Write() 

    pythia.next()

root_file.Write()
root_file.Close()
