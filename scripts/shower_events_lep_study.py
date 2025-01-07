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
'z_x',
'z_y',
'z_z',
'taup_px', 
'taup_py', 
'taup_pz',
'taup_e',
'taup_npi',
'taup_npizero',
'taup_pi1_px',
'taup_pi1_py',
'taup_pi1_pz',
'taup_pi1_e',
'taup_pi1_vx',
'taup_pi1_vy',
'taup_pi1_vz',
'taup_pi2_px',
'taup_pi2_py',
'taup_pi2_pz',
'taup_pi2_e',
'taup_pi2_vx',
'taup_pi2_vy',
'taup_pi2_vz',
'taup_pi3_px',
'taup_pi3_py',
'taup_pi3_pz',
'taup_pi3_e',
'taup_pi3_vx',
'taup_pi3_vy',
'taup_pi3_vz',
'taup_pizero1_px',
'taup_pizero1_py',
'taup_pizero1_pz',
'taup_pizero1_e',
'taun_px',
'taun_py',
'taun_pz',
'taun_e',
'taun_npi',
'taun_npizero',
'taun_pi1_px',
'taun_pi1_py',
'taun_pi1_pz',
'taun_pi1_e',
'taun_pi1_vx',
'taun_pi1_vy',
'taun_pi1_vz',
'taun_pi2_px',
'taun_pi2_py',
'taun_pi2_pz',
'taun_pi2_e',
'taun_pi2_vx',
'taun_pi2_vy',
'taun_pi2_vz',
'taun_pi3_px',
'taun_pi3_py',
'taun_pi3_pz',
'taun_pi3_e',
'taun_pi3_vx',
'taun_pi3_vy',
'taun_pi3_vz',
'taun_pizero1_px',
'taun_pizero1_py',
'taun_pizero1_pz',
'taun_pizero1_e',
]


if '.hepmc' in args.output: root_output = args.output.replace('.hepmc','.root')
else: root_output=args.output+'.root'
fout = ROOT.TFile(root_output,'RECREATE')
tree = ROOT.TTree('tree','')

branch_vals = {}
for b in branches:
    branch_vals[b] = array('f',[0])
    tree.Branch(b,  branch_vals[b],  '%s/F' % b)

pythia = pythia8.Pythia("", False)
pythia.readFile(args.cmnd_file)

if args.input:
    pythia.readString("Beams:frameType = 4")
    pythia.readString("Beams:LHEF = %s" % args.input)
else:
    print('Producing full event in pythia')
    # if no LHE file given then produce full event setup using pythia for the hard process as well
    pythia.readString("Beams:idA = -11") # Positron
    pythia.readString("Beams:idB = 11")  # Electron
    pythia.readString("Beams:eCM = 91.188")  # Center-of-mass energy = Z resonance
    pythia.readString("TauDecays:externalMode = 1")

    # Enable Z production and decay to taus
    pythia.readString("WeakSingleBoson:ffbar2gmZ = on")
    pythia.readString("23:onMode = off")  # Turn off all Z decays
    pythia.readString("23:onIfAny = 15")  # Enable Z -> tau+ tau-

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
    pi0s = []

    rho0_mass = 0.7755

    # Retrieve the charge of the parent tau
    tau_charge = part.charge()

    for d in part.daughterList():
        daughter = event[d]
        if abs(daughter.id()) == 211:
            pis.append(daughter)
        if abs(daughter.id()) == 111:
            pi0s.append(daughter)

    if len(pis) == 3:
        # Separate the pion with the opposite charge to the parent tau
        first_pi = next(pi for pi in pis if pi.charge() != tau_charge)

        # Remove the first pion from the list
        remaining_pis = [pi for pi in pis if pi != first_pi]

        # Sort the remaining pions based on the mass of the pair with first_pi
        remaining_pis.sort(
            key=lambda pi: abs((first_pi.p() + pi.p()).mCalc() - rho0_mass)
        )

        # Combine the sorted list
        sorted_pis = [first_pi] + remaining_pis

#        print('checking pi sorting:')
#        print('tau charge = %i' % tau_charge)
#        for i, pi in enumerate(sorted_pis):
#            print('pi%i charge = %i' % (i, pi.charge()))
#            if i>0:
#                print('mass = %.4f, mass diff = %.4f' % ((sorted_pis[0].p() + pi.p()).mCalc(), abs((sorted_pis[0].p() + pi.p()).mCalc() - rho0_mass)))

        return (sorted_pis, pi0s)

    return (pis, pi0s)

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
        if pdgid == 11 and len(mother_ids) == 0:
            # the e+ directions defines the z direction
            # not really needed to store this since its always in the -z direction due to how the sample is produced..
            z_x = part.px()
            z_y = part.py()
            z_z = part.pz()
            r = (z_x**2 + z_y**2 + z_z**2)**.5
            z_x/=r
            z_y/=r
            z_z/=r
            branch_vals['z_x' % vars()][0] = z_x
            branch_vals['z_y' % vars()][0] = z_y
            branch_vals['z_z' % vars()][0] = z_z
        if abs(pdgid) == 15 and LastCopy:
            pis, pi0s = GetPiDaughters(part,pythia.event)
            tau_name = 'taun' if pdgid == 15 else 'taup'
            branch_vals['%(tau_name)s_px' % vars()][0] = part.px()
            branch_vals['%(tau_name)s_py' % vars()][0] = part.py()
            branch_vals['%(tau_name)s_pz' % vars()][0] = part.pz()
            branch_vals['%(tau_name)s_e' % vars()][0]  = part.e()
            branch_vals['%(tau_name)s_npi' % vars()][0]  = len(pis)
            branch_vals['%(tau_name)s_npizero' % vars()][0]  = len(pi0s)
            branch_vals['%(tau_name)s_pi1_px' % vars()][0] = pis[0].px()
            branch_vals['%(tau_name)s_pi1_py' % vars()][0] = pis[0].py()
            branch_vals['%(tau_name)s_pi1_pz' % vars()][0] = pis[0].pz()
            branch_vals['%(tau_name)s_pi1_e' % vars()][0]  = pis[0].e()
            branch_vals['%(tau_name)s_pi1_vx' % vars()][0] = pis[0].xProd()
            branch_vals['%(tau_name)s_pi1_vy' % vars()][0] = pis[0].yProd()
            branch_vals['%(tau_name)s_pi1_vz' % vars()][0] = pis[0].zProd()

            if len(pis) > 1:
                branch_vals['%(tau_name)s_pi2_px' % vars()][0] = pis[1].px()
                branch_vals['%(tau_name)s_pi2_py' % vars()][0] = pis[1].py()
                branch_vals['%(tau_name)s_pi2_pz' % vars()][0] = pis[1].pz()
                branch_vals['%(tau_name)s_pi2_e' % vars()][0]  = pis[1].e()
                branch_vals['%(tau_name)s_pi2_vx' % vars()][0] = pis[1].xProd()
                branch_vals['%(tau_name)s_pi2_vy' % vars()][0] = pis[1].yProd()
                branch_vals['%(tau_name)s_pi2_vz' % vars()][0] = pis[1].zProd()

                branch_vals['%(tau_name)s_pi3_px' % vars()][0] = pis[2].px()
                branch_vals['%(tau_name)s_pi3_py' % vars()][0] = pis[2].py()
                branch_vals['%(tau_name)s_pi3_pz' % vars()][0] = pis[2].pz()
                branch_vals['%(tau_name)s_pi3_e' % vars()][0]  = pis[2].e()
                branch_vals['%(tau_name)s_pi3_vx' % vars()][0] = pis[2].xProd()
                branch_vals['%(tau_name)s_pi3_vy' % vars()][0] = pis[2].yProd()
                branch_vals['%(tau_name)s_pi3_vz' % vars()][0] = pis[2].zProd()

            if len(pi0s) > 0:
                branch_vals['%(tau_name)s_pizero1_px' % vars()][0] = pi0s[0].px()
                branch_vals['%(tau_name)s_pizero1_py' % vars()][0] = pi0s[0].py()
                branch_vals['%(tau_name)s_pizero1_pz' % vars()][0] = pi0s[0].pz()            
                branch_vals['%(tau_name)s_pizero1_e' % vars()][0]  = pi0s[0].e()

    hepmc_event = HepMC3.GenEvent()
    hepmc_converter.fill_next_event1(pythia, hepmc_event, count+1)
    hepmc_writer.write_event(hepmc_event)

    tree.Fill()
    count+=1
    if not stopGenerating: pythia.next()

# Finalize
#pythia.stat()
hepmc_writer.close()  

tree.Write()
fout.Close()
