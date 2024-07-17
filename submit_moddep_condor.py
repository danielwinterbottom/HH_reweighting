import os

os.system('mkdir -p condor_jobs')
os.system('mkdir -p outputs_4b_BM_vals_17July')

def Submit(input, output, options, N):
    for i in range(N):
    
        output_i = '%(output)s_%(i)i.root' % vars()
        job_file_name = 'condor_jobs/job_sm_%(i)i.job' % vars()
        job_file_out_name = 'condor_jobs/job_sm_%(i)i_$(CLUSTER)_$(PROCESS).job' % vars()
    
    
        out_str =  'executable = run_batch_condor.sh\n'
        out_str += 'arguments = \"%(input)s %(output_i)s %(i)i %(options)s\"\n' % vars()
        out_str += 'output = %(job_file_out_name)s.out\n' % vars()
        out_str += 'error = %(job_file_out_name)s.err\n' % vars()
        out_str += 'log = %(job_file_out_name)s.log\n' % vars()
        out_str += '+MaxRuntime = 10000\n'
        out_str += 'queue'
        job_file = open(job_file_name,'w')
        job_file.write(out_str)
        job_file.close()
    
        os.system('condor_submit %(job_file_name)s' % vars())

#for bm in range(1,9):
for bm in range(1,2):

    input = 'batch_job_outputs/HH_loop_sm_twoscalar_BM%i_Ntot_200000_Njob_10000/cmsgrid_final_all.lhe' % bm
    output = 'outputs_4b_BM_vals_17July/output_mg_pythia_BM%i' % bm
    options = ''
    Submit(input, output, options, 100)
