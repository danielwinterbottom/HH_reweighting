import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--events_per_job', help= 'Number of events per job', default=10000, type=int)
parser.add_argument('--total_events', help= 'Total number of events to produce', default=200000, type=int)
parser.add_argument('--input', '-i', help= 'Name of input gridpack')
args = parser.parse_args()

current_dir=os.getcwd()
input = args.input
nevents = args.events_per_job
name = args.input[:args.input.find('_el9')]+'_Ntot_%i_Njob_%i' % (args.total_events, nevents)
gridpackname = args.input

def Submit(job, jobname, N):
   print(jobname)

   job_file_name = 'jobs/job_%(jobname)s.job' % vars()
   job_file_out_name = 'jobs/job_%(jobname)s_$(CLUSTER)_$(PROCESS).job' % vars()

   out_str =  'executable = %s\n' % job
   out_str += 'arguments = \"$(CLUSTER) $(PROCESS)\"\n' % vars()
   out_str += 'output = %(job_file_out_name)s.out\n' % vars()
   out_str += 'error = %(job_file_out_name)s.err\n' % vars()
   out_str += 'log = %(job_file_out_name)s.log\n' % vars()
   out_str += '+MaxRuntime = 10000\n'
   out_str += 'queue %i' % N
   job_file = open(job_file_name,'w')
   job_file.write(out_str)
   job_file.close()

   os.system('condor_submit %(job_file_name)s' % vars())


# first we need to make a gridpack

out_string = '#!/bin/sh\n\
echo \"Cluster = $1 Process = $2\"\n\
cd %(current_dir)s/batch_job_outputs/%(name)s/job_output_$2\n\
rm -rf %(current_dir)s/batch_job_outputs/%(name)s/job_output_$2/*\n\
tar -xvf %(current_dir)s/gridpacks/%(gridpackname)s\n\
./runcmsgrid.sh %(nevents)i $2 1\n\
mv cmsgrid_final.lhe cmsgrid_final_$2.lhe\n\
ls | grep -v ".lhe" | xargs rm -r' % vars()

# make an output directory
os.system('mkdir -p batch_job_outputs/%(name)s' % vars())

for i in range(0,int(args.total_events/args.events_per_job)):
    # make an directory for each job where the gridpack will be untarred and the output will be stored
    os.system('mkdir -p %(current_dir)s/batch_job_outputs/%(name)s/job_output_%(i)i' % vars())

print('Writing job file: jobs/parajob_%(name)s.sh' % vars())
with open("jobs/parajob_%(name)s.sh" % vars(), "w") as output_file:
    output_file.write(out_string)
os.system('chmod +x jobs/parajob_%(name)s.sh' % vars())

total = int(args.total_events/args.events_per_job)
print('Submitting jobs')
#os.system('mkdir -p %(current_dir)s/batch_job_outputs/%(name)s/logs' % vars())
Submit('jobs/parajob_%(name)s.sh' % vars(), name, total)
#os.system('qsub -e %(current_dir)s/batch_job_outputs/%(name)s/logs -o %(current_dir)s/batch_job_outputs/%(name)s/logs -q hep.q -l h_rt=0:180:0 -t 1-%(total)i:1 jobs/parajob_%(name)s.sh' % vars())
