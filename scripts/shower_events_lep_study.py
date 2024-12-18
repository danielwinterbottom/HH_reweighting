import pythia8
import argparse
from pyHepMC3 import HepMC3
from Pythia8ToHepMC3 import Pythia8ToHepMC3

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help= 'LHE file to be converted')
parser.add_argument('--output', '-o', help= 'Name of output file',default='pythia_output.hepmc')
parser.add_argument('--cmnd_file', '-c', help= 'Pythia8 command file')
parser.add_argument('--n_events', '-n', help= 'Maximum number of events to process', default=-1, type=int)
parser.add_argument('--n_skip', '-s', help= 'skip n_events*n_skip', default=0, type=int)

args = parser.parse_args()

pythia = pythia8.Pythia("", False)
pythia.readFile(args.cmnd_file)

pythia.readString("Beams:frameType = 4")
pythia.readString("Beams:LHEF = %s" % args.input)
pythia.init()

pythia.LHAeventSkip(args.n_skip*args.n_events)

hepmc_converter = Pythia8ToHepMC3()
hepmc_writer = HepMC3.WriterAscii(args.output)

stopGenerating = False
count = 0
while not stopGenerating:

    stopGenerating = pythia.infoPython().atEndOfFile()
    if args.n_events>0 and count+1 >= args.n_events: stopGenerating = True

    print('-------------------------------')
    print('event %i' % (count+1))

    print('particles:')
    for i, part in enumerate(pythia.event):
        pdgid = part.id()
        mother_ids = [pythia.event[x].id() for x in part.motherList()]
        print(pdgid, part.status(), mother_ids)

    hepmc_event = HepMC3.GenEvent()
    hepmc_converter.fill_next_event1(pythia, hepmc_event, count+1)
    hepmc_writer.write_event(hepmc_event)

    count+=1
    pythia.next()

# Finalize
pythia.stat()
hepmc_writer.close()    
