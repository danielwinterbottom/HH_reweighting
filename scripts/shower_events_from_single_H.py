import pythia8
import argparse
import ROOT
from array import array
from python.reweight import *
import random

class SmearHHbbgamgam():
    def __init__(self):
        # mgamgam resolutiosn taken from page 3 here: https://arxiv.org/pdf/2310.01643.pdf
        res_gamgam = 0.01
        # mbb resolution taken from Figure 5 here: https://arxiv.org/pdf/1912.06046.pdf
        res_bb = 0.12

        self.func1 = ROOT.TF1("func1","TMath::Gaus(x,0,%g)" % res_bb,-5*res_bb,5*res_bb)
        self.func2 = ROOT.TF1("func2","TMath::Gaus(x,0,%g)" % res_gamgam,-5*res_gamgam,5*res_gamgam)

    def Smear(self,h1,h2):
      rand1 = 1.+smear.func1.GetRandom()
      rand2 = 1.+smear.func2.GetRandom()

      if random.random() > 0.5:
          h1_smeared = h1*rand1
          h2_smeared = h2*rand2
      else: 
          h1_smeared = h1*rand2
          h2_smeared = h2*rand1

      return h1_smeared, h2_smeared


parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help= 'LHE file to be converted')
parser.add_argument('--output', '-o', help= 'Name of output file',default='pythia_output.root')
parser.add_argument('--cmnd_file', '-c', help= 'Pythia8 command file')
parser.add_argument('--n_events', '-n', help= 'Maximum number of events to process', default=-1, type=int)
parser.add_argument('--ref_width', help= 'Width of S-channel process reference sample')
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
H_pT_first  = array('f',[0])
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
tree.Branch("hh_pT_first",  hh_pT_first,  'hh_pT_first/F')
tree.Branch("H_pT_first",  H_pT_first,  'H_pT_first/F')
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

# initialise reweighting

rw = HHReweight([600],[0.008333,0.01,0.02],ReweightSChan=True,RefMassRelWidth=(600,float(args.ref_width)))
rw_names = rw.GetWeightNames()

for wt in rw_names:
    if wt not in weights_map:
        weights_map[wt] = array('f',[0])
        tree.Branch("wt_%(wt)s" % vars(), weights_map[wt], 'wt_%(wt)s/F' % vars())

# initialise smearing

smear = SmearHHbbgamgam()

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
    higgs_bosons = []
    H_pT_first[0] = -9999
    for part in pythia.event:
        pdgid = part.id()

        if pdgid == 35:
            #print part.id(), part.status(), part.px(), part.py()
            #print 'mothers:'
            #print [pythia.event[p].id() for p in part.motherList()]
            #print 'daughters:'
            #print [pythia.event[d].id() for d in part.daughterList()] 
            #print part.pT()
            firstCopyH = True not in [pythia.event[p].id() == 35 for p in part.motherList()] 
            if firstCopyH: H_pT_first[0] = part.pT()

        if pdgid != 25: continue

        firstCopy = True not in [pythia.event[p].id() == 25 for p in part.motherList()]
 
        lastCopy = len(part.daughterList()) == 2 and abs(pythia.event[part.daughterList()[0]].id()) == 5 and abs(pythia.event[part.daughterList()[1]].id()) == 5 

        if not (firstCopy or lastCopy): continue

        lvec = ROOT.TLorentzVector(part.px(),part.py(),part.pz(),part.e()) 

        if firstCopy: higgs_bosons_first.append(lvec)
        elif lastCopy: higgs_bosons.append(lvec)

        if firstCopy and lastCopy: higgs_bosons.append(lvec)

    if len(higgs_bosons_first) == 2:
        hh_mass_first[0] = (higgs_bosons_first[0]+higgs_bosons_first[1]).M()
        hh_pT_first[0] = (higgs_bosons_first[0]+higgs_bosons_first[1]).Pt()
    else: 
        hh_mass_first[0] = -9999
        hh_pT_first[0] = -9999

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

    if len(higgs_bosons_first) == 2:
        # need to shift masses to 125 GeV, otherwise we get lots of errors
        higgs_bosons_first[0].SetE((higgs_bosons_first[0].P()**2+125.**2)**.5)
        higgs_bosons_first[1].SetE((higgs_bosons_first[1].P()**2+125.**2)**.5)
        parts = [
          [25, higgs_bosons_first[0].E(), higgs_bosons_first[0].Px(), higgs_bosons_first[0].Py(), higgs_bosons_first[0].Pz()],
          [25, higgs_bosons_first[1].E(), higgs_bosons_first[1].Px(), higgs_bosons_first[1].Py(), higgs_bosons_first[1].Pz()],
        ]

        alphas = pythia.infoPython().alphaS()
        rweights = rw.ReweightEvent(parts,alphas)

        for key in rweights: 
            weights_map[key][0] = rweights[key]
        

    if hh_mass[0] >0:
        tree.Fill()

    count += 1
    if count % 10000 == 0: print 'Processed %i events' % count
    pythia.next()

root_file.Write()
root_file.Close()
