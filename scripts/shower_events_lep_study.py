import pythia8
import argparse
from pyHepMC3 import HepMC3
from Pythia8ToHepMC3 import Pythia8ToHepMC3
import ROOT
from array import array

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help= 'LHE file to be converted')
parser.add_argument('--output', '-o', help= 'Name of output file',default='pythia_output.hepmc')
parser.add_argument('--cmnd_file', '-c', help= 'Pythia8 command file')
parser.add_argument('--n_events', '-n', help= 'Maximum number of events to process', default=-1, type=int)
parser.add_argument('--n_skip', '-s', help= 'skip n_events*n_skip', default=0, type=int)

args = parser.parse_args()


# setup output root file
# taup = positive tau, indices will label the pions
# for now only setup for pipi channel so only index 1 is used for the pion from tau->pinu decays

branches = [
'taup_px', 
'taup_py', 
'taup_pz',
'taup_e',
'taup_pi1_px',
'taup_pi1_py',
'taup_pi1_pz',
'taup_pi1_e',
'taup_pi1_vx',
'taup_pi1_vy',
'taup_pi1_vz',
'taun_px',
'taun_py',
'taun_pz',
'taun_e',
'taun_pi1_px',
'taun_pi1_py',
'taun_pi1_pz',
'taun_pi1_e',
'taun_pi1_vx',
'taun_pi1_vy',
'taun_pi1_vz',
]

branch_vals = {}

if '.hepmc' in args.output: root_output = args.output.replace('.hepmc','.root')
else: root_output=args.output+'.root'
fout = ROOT.TFile(root_output,'RECREATE')
tree = ROOT.TTree('tree','')

for b in branches:
    branch_vals[b] = array('f',[0])
    tree.Branch(b,  branch_vals[b],  '%s/F' % b)

pythia = pythia8.Pythia("", False)
pythia.readFile(args.cmnd_file)

pythia.readString("Beams:frameType = 4")
pythia.readString("Beams:LHEF = %s" % args.input)
pythia.init()

pythia.LHAeventSkip(args.n_skip*args.n_events)

hepmc_converter = Pythia8ToHepMC3()
hepmc_writer = HepMC3.WriterAscii(args.output)

def IsLastCopy(part, event):
    ''' 
    check if particle is the last copy by checking if it has no daughters of the same pdgid
    check may not work for some particle types - tested only for taus at the moment
    '''
    LastCopy = True
    pdgid = part.id()
    for p in part.daughterList():
        if event[p].id() == pdgid: LastCopy = False

    return LastCopy

def GetPiDaughters(part, event):
    pis = []
    for d in part.daughterList():
        daughter = event[d]
        if abs(daughter.id()) == 211:
            pis.append(daughter)
    return pis

stopGenerating = False
count = 0
while not stopGenerating:

    stopGenerating = pythia.infoPython().atEndOfFile()
    if args.n_events>0 and count+1 >= args.n_events: stopGenerating = True

    #print('-------------------------------')
    #print('event %i' % (count+1))

    #print('particles:')
    for i, part in enumerate(pythia.event):
        pdgid = part.id()
        mother_ids = [pythia.event[x].id() for x in part.motherList()]
        daughter_ids = [pythia.event[x].id() for x in part.daughterList()]
        #print(pdgid, part.e(), part.charge(), part.status(), mother_ids, daughter_ids)
        LastCopy = IsLastCopy(part, pythia.event)
        if abs(pdgid) == 15 and LastCopy:
            pis = GetPiDaughters(part,pythia.event)
            tau_name = 'taun' if pdgid == 15 else 'taup'
            branch_vals['%(tau_name)s_px' % vars()][0] = part.px()
            branch_vals['%(tau_name)s_py' % vars()][0] = part.py()
            branch_vals['%(tau_name)s_pz' % vars()][0] = part.pz()
            branch_vals['%(tau_name)s_e' % vars()][0]  = part.e()
            branch_vals['%(tau_name)s_pi1_px' % vars()][0] = pis[0].px()
            branch_vals['%(tau_name)s_pi1_py' % vars()][0] = pis[0].py()
            branch_vals['%(tau_name)s_pi1_pz' % vars()][0] = pis[0].pz()
            branch_vals['%(tau_name)s_pi1_e' % vars()][0]  = pis[0].e()
            branch_vals['%(tau_name)s_pi1_vx' % vars()][0] = pis[0].xProd()
            branch_vals['%(tau_name)s_pi1_vy' % vars()][0] = pis[0].yProd()
            branch_vals['%(tau_name)s_pi1_vz' % vars()][0] = pis[0].zProd()

    hepmc_event = HepMC3.GenEvent()
    hepmc_converter.fill_next_event1(pythia, hepmc_event, count+1)
    hepmc_writer.write_event(hepmc_event)

    tree.Fill()
    count+=1
    pythia.next()

# Finalize
#pythia.stat()
hepmc_writer.close()  

tree.Write()
fout.Close()
