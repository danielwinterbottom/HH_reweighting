import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--events_per_job', help= 'Number of events per job', default=10000, type=int)
parser.add_argument('--total_events', help= 'Total number of events to produce', default=1000000, type=int)
parser.add_argument('--input', '-i', help= 'Name of input madgraph directory')
args = parser.parse_args()

current_dir=os.getcwd()
input = args.input
nevents = args.events_per_job
name = args.input.split('/')[-1]+'_Ntot_%i_Njob_%i' % (args.total_events, nevents)
gridpackname = args.input.split('/')[-1]

# first we need to make a gridpack
os.system('cd %(current_dir)s/%(input)s; { echo 0; echo set nevents 100; echo set gridpack True; } | python2 bin/generate_events %(gridpackname)s; cd %(current_dir)s' % vars())

out_string = '#!/bin/sh\n\
cd %(current_dir)s/batch_job_outputs/%(name)s/job_output_${SGE_TASK_ID}\n\
rm -rf %(current_dir)s/batch_job_outputs/%(name)s/job_output_${SGE_TASK_ID}/*\n\
tar -xvf %(current_dir)s/%(input)s/%(gridpackname)s_gridpack.tar.gz\n\
./run.sh %(nevents)i ${SGE_TASK_ID}\n\
mv events.lhe.gz events_${SGE_TASK_ID}.lhe.gz\n\
echo 0 | madevent/bin/madevent reweight events_${SGE_TASK_ID}.lhe.gz\n\
rm -rf madevent run.sh py.py' % vars()

# make an output directory
os.system('mkdir -p batch_job_outputs/%(name)s/logs' % vars())

for i in range(1,args.total_events/args.events_per_job+1):
    # make an directory for each job where the gridpack will be untarred and the output will be stored
    os.system('mkdir -p %(current_dir)s/batch_job_outputs/%(name)s/job_output_%(i)i' % vars())

print 'Writing job file: jobs/parajob_%(name)s.sh' % vars()
with open("jobs/parajob_%(name)s.sh" % vars(), "w") as output_file:
    output_file.write(out_string)
os.system('chmod +x jobs/parajob_%(name)s.sh' % vars())

total = int(args.total_events/args.events_per_job)
print 'Submitting jobs'
os.system('mkdir -p %(current_dir)s/batch_job_outputs/%(name)s/logs' % vars())
os.system('qsub -e %(current_dir)s/batch_job_outputs/%(name)s/logs -o %(current_dir)s/batch_job_outputs/%(name)s/logs -q hep.q -l h_rt=0:180:0 -t 1-%(total)i:1 jobs/parajob_%(name)s.sh' % vars())
