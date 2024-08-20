import pythia8
import argparse
import ROOT
from array import array
from python.reweight import *
import math



parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help= 'ROOT file containing showered events to be reweighted')
parser.add_argument('--output', '-o', help= 'Name of output file', default=None)
parser.add_argument('--n_events', '-n', help= 'Maximum number of events to process', default=-1, type=int)
parser.add_argument('--ref_width', help= 'Width of S-channel process reference sample', default=None)
parser.add_argument('--ref_mass', help= 'Mass of the H in the reference sample', default=600, type=float)
args = parser.parse_args()

infile = ROOT.TFile(args.input)
intree = infile.Get("ntuple")

if args.output: root_filename = args.output
else: root_filename = args.input.replace('.root','_reweighted.root')
outfile = ROOT.TFile(root_filename, "RECREATE")
#outtree = ROOT.TTree("ntuple", "Event Tree")
outtree = intree.CloneTree(0)

weights_map = {}
weight_names = []

# initialise reweighting

mass_widths_dict = {
  260: [0.0008519, 0.002199, 0.0004948],
  380: [0.002737, 0.0006079, 0.002219],
  440: [0.003323,0.0008803],
  450: [0.00152,0.00143],
  500: [0.001164, 0.0049212, 0.001115, 0.005545, 0.004702],
  560: [0.005333],
  600: [0.008333,0.01,0.02,0.05,0.10],
  680: [0.008925],
  620: [0.007422],
  800: [0.00297, 0.01226],
  870: [0.01097],
}

if args.ref_width:
    mass_widths_dict_new = {}
    mass_widths_dict_new[args.ref_mass] = mass_widths_dict[args.ref_mass] 
    mass_widths_dict = mass_widths_dict_new   

if args.ref_width: rw = HHReweight(ReweightSChan=True,RefMassRelWidth=(args.ref_mass,float(args.ref_width)),mass_widths_dict=mass_widths_dict)
else: rw = HHReweight(mass_widths_dict=mass_widths_dict)
rw_names = rw.GetWeightNames()

for wt in rw_names:
    if wt not in weights_map:
        weights_map[wt] = array('f',[0])
        outtree.Branch("wt_%(wt)s" % vars(), weights_map[wt], 'wt_%(wt)s/F' % vars())

# loop over events and get HH pairs

for i in range(intree.GetEntries()):

    if args.n_events>0 and i >= args.n_events: break

    intree.GetEntry(i)

    parts = []

    alphas = intree.alphas
    higgs_1 = intree.higgs_1
    higgs_2 = intree.higgs_2

    p1 = [int(higgs_1[0]), higgs_1[1], higgs_1[2], higgs_1[3], higgs_1[4]]
    p2 = [int(higgs_2[0]), higgs_2[1], higgs_2[2], higgs_2[3], higgs_2[4]]

    parts = [p1,p2]

    if len(parts) == 2 and parts[0][0] == 25 and parts[1][0] == 25:

        rweights = rw.ReweightEvent(parts,alphas)

        for key in rweights: 
            weights_map[key][0] = rweights[key]
        outtree.Fill()
        

    if i+1 % 1000 == 0: print('Processed %i events' % i+1)

outfile.cd()
outtree.Write()
outfile.Close()
infile.Close()
