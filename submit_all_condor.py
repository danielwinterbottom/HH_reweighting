import os

os.system('mkdir -p condor_jobs')
os.system('mkdir -p outputs_4b_Jun05')

def Submit(input, output, options, N, name=''):
    for i in range(N):
    
        output_i = '%(output)s_%(i)i.root' % vars()
        job_file_out_name = 'condor_jobs/job_%(name)s_%(i)i_$(CLUSTER)_$(PROCESS)' % vars()
        job_file_name = 'condor_jobs/job_%(name)s_%(i)i.job' % vars()
    
    
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
    
        os.system('condor_submit %s' % job_file_name)

#input = 'MG5_aMC_v3_5_4/QCD_pp_to_4b_LO_loose/Events/run_02/unweighted_events.lhe'
#output = 'outputs_new_vars_50pcimprove_v3/QCDTo4b_loose'
#options = ''
#Submit(input, output, options, 500)
#
#input = 'MG5_aMC_v3_5_4/QCD_pp_to_4b_LO/Events/run_01/unweighted_events.lhe'
#output = 'outputs_new_vars_50pcimprove_v3/QCDTo4b'
#options = ''
#Submit(input, output, options, 100)

#input = 'MG5_aMC_v3_5_4/HHto4b/Events/run_02_decayed_1/unweighted_events.lhe'
#output = 'outputs_new_vars_v3/HHto4b'
#options = ''
#Submit(input, output, options, 100)

#input = 'batch_job_outputs/HH_loop_sm_twoscalar_SM_Ntot_200000_Njob_10000/cmsgrid_final_all_reweighted_full.lhe'
input = 'batch_job_outputs/HH_loop_sm_twoscalar_SM_Ntot_200000_Njob_10000/cmsgrid_final_all.lhe'
output = 'outputs_4b_Mar12_v2/output_mg_pythia_sm'
options = ''
Submit(input, output, options, 100, 'sm')

#input = 'batch_job_outputs/HH_loop_sm_twoscalar_SChan_eta0_M_500_RelWidth_0p003_Ntot_200000_Njob_10000/cmsgrid_final_all.lhe'
output = 'outputs_4b_Mar12_v2/output_mg_pythia_mass_500GeV_relWidth0p003'
options = '\'--ref_mass=500 --ref_width=0.003\''
Submit(input, output, options, 100, 'bsm_m500')

#input = 'batch_job_outputs/HH_loop_sm_twoscalar_SChan_eta0_M_450_RelWidth_0p002_Ntot_200000_Njob_10000/cmsgrid_final_all.lhe'
#output = 'outputs_4b_Aug09/output_mg_pythia_mass_450GeV_relWidth0p002'
#options = '\'--ref_mass=450 --ref_width=0.002\''
#Submit(input, output, options, 100, 'bsm_m450')

#
#input = 'batch_job_outputs/HH_loop_sm_twoscalar_BOX_Ntot_200000_Njob_10000/cmsgrid_final_all.lhe'
#output = 'outputs_4b_Jun05/output_mg_pythia_box'
#options = ''
#Submit(input, output, options, 100)
#
#input = 'batch_job_outputs/HH_loop_sm_twoscalar_SChan_h_Ntot_200000_Njob_10000/cmsgrid_final_all.lhe'
#output = 'outputs_4b_Jun05/output_mg_pythia_Sh'
#options = ''
#Submit(input, output, options, 100)

#input = 'batch_job_outputs/HH_loop_sm_twoscalar_BOX_SChan_h_inteference_Ntot_200000_Njob_10000/cmsgrid_final_all.lhe'
#output = 'outputs_4b_Jun05/output_mg_pythia_int'
#options = ''
#Submit(input, output, options, 100)
