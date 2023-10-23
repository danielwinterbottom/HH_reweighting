import ROOT
import math
from array import array
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help= 'LHE file to be converted')
parser.add_argument('--output', '-o', help= 'Name of output file',default='output.root')
parser.add_argument('--n_events', '-n', help= 'Maximum number of events to process', default=-1, type=int)
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
wt_nom_out  = array('f',[0])
wt_box_out  = array('f',[0])
wt_schannel_h_out  = array('f',[0])
wt_box_and_schannel_h_i_out  = array('f',[0])
wt_schannel_H_out  = array('f',[0])
wt_box_and_schannel_H_i_out  = array('f',[0])
wt_schannel_H_and_schannel_h_i_out  = array('f',[0])
 
# create the branches and assign the fill-variables to them
tree.Branch("hh_mass",  hh_mass,  'hh_mass/F')
tree.Branch("wt_nom",  wt_nom_out,  'wt_nom/F')
tree.Branch("wt_box",  wt_box_out,  'wt_box/F')
tree.Branch("wt_schannel_h",  wt_schannel_h_out,  'wt_schannel_h/F')
tree.Branch("wt_box_and_schannel_h_i",  wt_box_and_schannel_h_i_out,  'wt_box_and_schannel_h_i/F')
tree.Branch("wt_schannel_H",  wt_schannel_H_out,  'wt_schannel_H/F')
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
        wt_box_and_schannel_H = None
        wt_box_and_schannel_h = None
        wt_all = None
        wt_schannel_h = None
        wt_schannel_H = None

    elif line.startswith("</event>"):
        # end of event
        # compute weights and store variables
        event_started = False

        # store di-Higgs mass
        if len(higgs_bosons)==2: hh_mass[0]=(higgs_bosons[0]+higgs_bosons[1]).M()
        else: h_mass[0]=-1

        # compute weights
        # get the inteference contributions
        wt_box_and_schannel_H_i = wt_box_and_schannel_H - wt_box
        wt_box_and_schannel_h_i = wt_box_and_schannel_h - wt_box
        wt_schannel_H_and_schannel_h_i = wt_all - wt_box - wt_schannel_H - wt_schannel_h - wt_box_and_schannel_H_i - wt_box_and_schannel_h_i

        # the weights that we need to produce our final templates are:
        # wt_box, wt_schannel_H, wt_schannel_h, wt_box_and_schannel_H_i, wt_box_and_schannel_h_i, and wt_schannel_H_and_schannel_h_i 

        # we scale all these weights so that they correspond to kappa_t=1 for both h and H
        # the mixing angle used during generation of events was 0.785398 and kappa_t for h and H = cos(alpha) and sin(alpha) respectively
        # note that the lambdas are already set equal to the SM lambda
        wt_box*=(1./math.cos(0.785398))**4
        wt_schannel_h*=(1./math.cos(0.785398))**2 
        wt_box_and_schannel_h_i*=(1./math.cos(0.785398))**3
        wt_schannel_H*=(1./math.sin(0.785398))**2 
        wt_box_and_schannel_h_i*=(1./math.cos(0.785398))**2*(1./math.sin(0.785398))

        wt_box_out[0] = wt_box
        wt_schannel_h_out[0] = wt_schannel_h
        wt_box_and_schannel_h_i_out[0] = wt_box_and_schannel_h_i
        wt_schannel_H_out[0] = wt_schannel_H
        wt_box_and_schannel_H_i_out[0] = wt_box_and_schannel_H_i
        wt_schannel_H_and_schannel_h_i_out[0] = wt_schannel_H_and_schannel_h_i
    
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
            elif line.startswith("<wgt id=\'box_and_schannel_H\'>"):
                wt_box_and_schannel_H=float(parts[2])
                #print 'wt_box_and_schannel_H = ', wt_box_and_schannel_H
            elif line.startswith("<wgt id=\'box_and_schannel_h\'>"):
                wt_box_and_schannel_h=float(parts[2])
                #print 'wt_box_and_schannel_h = ', wt_box_and_schannel_h
            elif line.startswith("<wgt id=\'all\'>"):
                wt_all=float(parts[2])
                #print 'wt_all = ', wt_all
            elif line.startswith("<wgt id=\'schannel_h\'>"):
                wt_schannel_h=float(parts[2])
                #print 'wt_schannel_h = ', wt_schannel_h
            elif line.startswith("<wgt id=\'schannel_H\'>"):
                wt_schannel_H=float(parts[2])
                #print 'wt_schannel_H = ', wt_schannel_H

    if count>args.n_events and args.n_events>=0: break

# Write the TTree to the ROOT file
root_file.Write()
root_file.Close()
lhe_file.close()
