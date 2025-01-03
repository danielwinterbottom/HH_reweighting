import ROOT
import argparse
import math
from array import array

# each LEP detector should have about 140000 Z->tautau events (not accounting for acceptance effects)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Input ROOT file.")
parser.add_argument("-o", "--output", required=True, help="Output ROOT file.")
parser.add_argument('--n_events', '-n', help='Maximum number of events to process', default=-1, type=int)
parser.add_argument('--n_skip', '-s', help='skip n_events*n_skip', default=0, type=int)

args = parser.parse_args()

# Open the input ROOT file
input_root = ROOT.TFile(args.input, "READ")
if input_root.IsZombie():
    raise Exception(f"Unable to open file {args.input}")

# Access the tree in the input ROOT file
tree = input_root.Get("tree")  # Replace 'tree' with your tree name
if not tree:
    raise Exception("Tree not found in the input file.")
    input_root.Close()

# Create an output ROOT file
output_root = ROOT.TFile(args.output, "RECREATE")

# Create a new tree to store the output
new_tree = ROOT.TTree("new_tree","Event Tree")

branches = [
        'cosn_plus',
        'cosr_plus',
        'cosk_plus',
        'cosn_minus',
        'cosr_minus',
        'cosk_minus',
        'cosTheta',
]
branch_vals = {}
for b in branches:
    branch_vals[b] = array('f',[0])
    new_tree.Branch(b,  branch_vals[b],  '%s/F' % b)

# Determine the range of entries to process
n_entries = tree.GetEntries()
start_entry = args.n_skip * args.n_events + 1 if args.n_events > 0 else 1
end_entry = start_entry + args.n_events if args.n_events > 0 else n_entries

if start_entry >= n_entries:
    raise Exception("Error: Start entry exceeds total number of entries in the tree.")
    input_root.Close()
    output_root.Close()

end_entry = min(end_entry, n_entries)

# Loop over the entries in the input tree
count = 0
for i in range(start_entry, end_entry):

    tree.GetEntry(i)

    # get tau 4-vectors
    taup = ROOT.TLorentzVector(tree.taup_px, tree.taup_py, tree.taup_pz, tree.taup_e)
    taun = ROOT.TLorentzVector(tree.taun_px, tree.taun_py, tree.taun_pz, tree.taun_e)

    # compute coordinate vectors here (n,r,k)
    # p is direction of e+ beam
    p = ROOT.TVector3(tree.z_x, tree.z_y, tree.z_z).Unit()
    # k is direction of tau+
    k = taup.Vect().Unit()
    n = (p.Cross(k)).Unit()
    cosTheta = p.Dot(k)
    r = (p - (k*cosTheta)).Unit() 
    #print('!!!!!')
    #print('n:', n.X(), n.Y(), n.Z())
    #print('p:', p.X(), p.Y(), p.Z())
    #print('k:', k.X(), k.Y(), k.Z())
    #print('r:', r.X(), r.Y(), r.Z())

    # get pion 4-vectors
    taup_pi = ROOT.TLorentzVector(tree.taup_pi1_px, tree.taup_pi1_py, tree.taup_pi1_pz, tree.taup_pi1_e)
    taun_pi = ROOT.TLorentzVector(tree.taun_pi1_px, tree.taun_pi1_py, tree.taun_pi1_pz, tree.taun_pi1_e)
    # boost into tau rest frames
    taup_pi.Boost(-taup.BoostVector())
    taun_pi.Boost(-taun.BoostVector())

    taup_s = taup_pi.Vect().Unit()
    taun_s = taun_pi.Vect().Unit()
    branch_vals['cosn_plus'][0] = taup_s.Dot(n)
    branch_vals['cosr_plus'][0] = taup_s.Dot(r)
    branch_vals['cosk_plus'][0] = taup_s.Dot(k)
    branch_vals['cosn_minus'][0] = taun_s.Dot(n)
    branch_vals['cosr_minus'][0] = taun_s.Dot(r)
    branch_vals['cosk_minus'][0] = taun_s.Dot(k)
    branch_vals['cosTheta'][0] = cosTheta


    ## Fill the new tree
    new_tree.Fill()
    count+=1
    if count % 1000 == 0:
        print('Processed %i events' % count)

# Write the new tree to the output file
new_tree.Write()

# Close the files
input_root.Close()
output_root.Close()
