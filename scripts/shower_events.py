import pythia8
import argparse
import ROOT
from array import array
import random

class SmearBJet():
    def __init__(self):
        # take resolutions vs pT from Fig 3 in https://arxiv.org/pdf/1912.06046.pdf (left plot for DNN)
        x_vals = array('d',[30.,50.,70.,90.,125.,175.,224.,275.,350.])
        y_vals = array('d',[0.111,0.106,0.094,0.087,0.082,0.076,0.073,0.068,0.064])
        self.res_graph = ROOT.TGraph(len(x_vals), x_vals, y_vals)
        self.min_pt = x_vals[0]
        self.max_pt = x_vals[-1]
        self.func = ROOT.TF1("func","TMath::Gaus(x,0,[0])",-0.5,0.5)

    def Smear(self,j):
        pt = max(min(j.Pt(),self.max_pt), self.min_pt)
        sigma = self.res_graph.Eval()
        print pt, sigma
        self.res_graph.SetParameter(0,sigma)
        rand = 1.+self.func.GetRandom()
        print rand
        j_smeared = j*rand

        return j_smeared    

class SmearHHbbgamgam():
    def __init__(self):
        # mgamgam resolutiosn taken from page 3 here: https://arxiv.org/pdf/2310.01643.pdf
        res_gamgam = 0.01
        # mbb resolution taken from Figure 5 here: https://arxiv.org/pdf/1912.06046.pdf
        res_bb = 0.12

        self.func1 = ROOT.TF1("func1","TMath::Gaus(x,0,%g)" % res_bb,-5*res_bb,5*res_bb)
        self.func2 = ROOT.TF1("func2","TMath::Gaus(x,0,%g)" % res_gamgam,-5*res_gamgam,5*res_gamgam)

    def Smear(self,h1,h2):
      rand1 = 1.+self.func1.GetRandom()
      rand2 = 1.+self.func2.GetRandom()

      if random.random() > 0.5:
          h1_smeared = h1*rand1
          h2_smeared = h2*rand2
      else:
          h1_smeared = h1*rand2
          h2_smeared = h2*rand1

      h1_smeared = h1*rand1
      h2_smeared = h2*rand2

      return h1_smeared, h2_smeared


parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help= 'LHE file to be converted')
parser.add_argument('--output', '-o', help= 'Name of output file',default='pythia_output.root')
parser.add_argument('--cmnd_file', '-c', help= 'Pythia8 command file')
parser.add_argument('--n_events', '-n', help= 'Maximum number of events to process', default=-1, type=int)
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
hh_mass_first  = array('f',[0])
hh_pT_first  = array('f',[0])
h1_pT_first = array('f',[0])
h2_pT_first = array('f',[0])
hh_mass_smear  = array('f',[0])
hh_pT_smear  = array('f',[0])
h1_pT_smear = array('f',[0])
h2_pT_smear = array('f',[0])
hh_mass_smear_improved  = array('f',[0])

tree.Branch("wt_nom",  wt_nom,  'wt_nom/F')
tree.Branch("hh_mass",  hh_mass,  'hh_mass/F')
tree.Branch("hh_mass_first",  hh_mass_first,  'hh_mass_first/F')
tree.Branch("hh_pT",  hh_pT,  'hh_pT/F')
tree.Branch("h1_pT",  h1_pT,  'h1_pT/F')
tree.Branch("h2_pT",  h2_pT,  'h2_pT/F')
tree.Branch("h1_pT_first",  h1_pT_first,  'h1_pT_first/F')
tree.Branch("h2_pT_first",  h2_pT_first,  'h2_pT_first/F')
tree.Branch("hh_pT_first",  hh_pT_first,  'hh_pT_first/F')
tree.Branch("hh_mass_smear",  hh_mass_smear,  'hh_mass_smear/F')
tree.Branch("hh_pT_smear",  hh_pT_smear,  'hh_pT_smear/F')
tree.Branch("hh_mass_smear_improved",  hh_mass_smear_improved,  'hh_mass_smear_improved/F')
tree.Branch("h1_pT_smear",  h1_pT_smear,  'h1_pT_smear/F')
tree.Branch("h2_pT_smear",  h2_pT_smear,  'h2_pT_smear/F')

weights_map = {}
weight_names = []
# first get names of all weights from the first event to define tree branches
pythia.next()
for i in range (pythia.infoPython().numberOfWeights()):
    name = pythia.infoPython().weightNameByIndex(i).replace('AUX_','')
    if name[0].isdigit(): continue
    weight_names.append(name)

for wt in weight_names:
    if wt not in weights_map:
        weights_map[wt] = array('f',[0])
        tree.Branch("wt_%(wt)s" % vars(), weights_map[wt], 'wt_%(wt)s/F' % vars())

# initialise smearing

smear = SmearHHbbgamgam()

stopGenerating = False
count = 0
while not stopGenerating:

    stopGenerating = pythia.infoPython().atEndOfFile()
    if args.n_events>0 and count >= args.n_events: stopGenerating = True

    wt_nom[0] = pythia.infoPython().weight()

    for i in range (pythia.infoPython().numberOfWeights()):
        name =  pythia.infoPython().weightNameByIndex(i).replace('AUX_','')
        if name not in weights_map: continue
        val =  pythia.infoPython().weightValueByIndex(i)
        weights_map[name][0] = val

    higgs_bosons_first = []
    higgs_bosons = []


    # loop over LHE particles
    for part in pythia.process:
        pdgid = part.id()
        if pdgid != 25: continue
        lvec = ROOT.TLorentzVector(part.px(),part.py(),part.pz(),part.e()) 
        higgs_bosons_first.append(lvec)

    for part in pythia.event:
        pdgid = part.id()

        if pdgid != 25: continue
 
        lastCopy = len(part.daughterList()) == 2 and abs(pythia.event[part.daughterList()[0]].id()) == 5 and abs(pythia.event[part.daughterList()[1]].id()) == 5 
        
        if not lastCopy: continue

        lvec = ROOT.TLorentzVector(part.px(),part.py(),part.pz(),part.e()) 
        higgs_bosons.append(lvec)

    if len(higgs_bosons_first) == 2:
        hh_mass_first[0] = (higgs_bosons_first[0]+higgs_bosons_first[1]).M()
        hh_pT_first[0] = (higgs_bosons_first[0]+higgs_bosons_first[1]).Pt()
        h1_pT_first[0] = max(higgs_bosons_first[0].Pt(), higgs_bosons_first[1].Pt())
        h2_pT_first[0] = min(higgs_bosons_first[0].Pt(), higgs_bosons_first[1].Pt())
    else: 
        hh_mass_first[0] = -9999
        hh_pT_first[0] = -9999
        h1_pT_first[0] = -9999
        h2_pT_first[0] = -9999

    if len(higgs_bosons) == 2:
        hh_mass[0] = (higgs_bosons[0]+higgs_bosons[1]).M()
        hh_pT[0] = (higgs_bosons[0]+higgs_bosons[1]).Pt()
        h1_pT[0] = max(higgs_bosons[0].Pt(), higgs_bosons[1].Pt())
        h2_pT[0] = min(higgs_bosons[0].Pt(), higgs_bosons[1].Pt())

        h1_smeared, h2_smeared = smear.Smear(higgs_bosons[0],higgs_bosons[1])

        hh_mass_smear[0] = (h1_smeared+h2_smeared).M()
        hh_mass_smear_improved[0] = hh_mass_smear[0] - (h1_smeared.M()-125.) - (h2_smeared.M()-125.)
        hh_pT_smear[0] = (h1_smeared+h2_smeared).Pt()

        h1_pT_smear[0] = max(h1_smeared.Pt(), h2_smeared.Pt())
        h2_pT_smear[0] = min(h1_smeared.Pt(), h2_smeared.Pt())
    else:
        hh_mass[0] = -9999
        hh_pT[0] = -9999
        hh_mass_smear[0] = -9999
        hh_pT_smear[0] = -9999
        h1_pT[0] = -9999
        h2_pT[0] = -9999
        h1_pT_smear[0] = -9999
        h2_pT_smear[0] = -9999

    if hh_mass[0] >0:
        tree.Fill()

    count += 1
    if count % 10000 == 0: print 'Processed %i events' % count
    pythia.next()

root_file.Write()
root_file.Close()
