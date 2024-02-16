# HH_reweighting
A repository to develope a ME-based reweighting tool for di-Higgs processes

## Setup 

Download and install Madgraph5 and download the TRSM files

	./setup_2p6.sh

Create directories for the SM and BSM processes with various combinations of diagrams excluded:

	cd MG5_aMC_v2_6_7
	python2 bin/mg5_aMC -f ../scripts/mg_script

This produces directories for the hh processes for different scenarios:
- The SM scenario: "SM_all"
- The SM-like scenario (no additional Higgs) but with lambda set to 0 (bos diagram only): "SM_box" 
- The SM-like scenario with only the s-channel diagram included: "SM_schannel"
- The BSM di-Higgs scenario which includes both SM like h and an additional heavy Higgs H with all diagrams included: "BMS_hh"
- The BSM scenario including only the s-channel diagrams for the heavy Higgs: "BSM_schannel"


These directories can be copied and renamed and the params and reweight cards can be modified to produce samples for different mass/width scenarios, with different weights, e.g

	cp -r BSM_hh BSM_hh_M300_W3

The script "scripts/make_cards.py" can be used to produce a params and reweighting card 
E.g to produce cards for a 300 GeV Higgs with 1% width run:

	python ../scripts/make_cards.py -o BSM_hh_M300_W3/Cards/ --masses "300,450,600" -m 300 -w 0.01

To run some events cd into one of the directories

	cd BSM_hh_M300_W3
 
And run events using bin/generate_events, e.g

	{ echo 0; echo set run_card iseed 10; echo set nevents 100; } | python2 bin/generate_events run_output_0

This will run 100 events using random seet 10 and store the output in run_output_0

You can convert the LHE files into ROOT files using the convert_lhe_to_root.py script, e.g

python scripts/convert_lhe_to_root.py -i MG5_aMC_v2_6_7/BSM_hh_M300_W0p5406704/Events/run_01/unweighted_events.lhe -o outputs/output_M300_W0p5406704_10000.root 

This script also produce reco-like di-Higgs masses implementing some realistic smearing

# make a gridpack for running b atch jobs

Open run_card.dat and change nevents to small number e.g 100 and change gridpack option to True
Then run generate_events as normal. Note this will fix the parameters 

# running batch jobs

The scripts/submit_ic_batch_jobs.py script will produce a gridpack for running batch jobs and then submit them as batch jobs. It is only setup to use the IC batch system at the moment

An example command to run 100 jobs each producing 10000 events:

	python scripts/submit_ic_batch_jobs.py -i MG5_aMC_v2_6_7/BSM_hh_M600_6 --events_per_job 10000 --total_events 1000000

Or to run jobs from CMS-style gridpacks do e.g:

        for x in gridpacks/*0p007183*; do i=${x//gridpacks\//""}; python scripts/submit_ic_gridpack_batch_jobs.py -i $i; done

# merging LHE files

You can merge all lhe files into one large on using:

	python scripts/mergeLHEFiles.py batch_job_outputs/BSM_hh_M600_W6_Ntot_1000000_Njob_10000/events_all.lhe batch_job_outputs/BSM_hh_M600_W6_Ntot_1000000_Njob_10000/job_output_*/events_*.lhe

the individual files can then be removed:
	
	rm batch_job_outputs/BSM_hh_M600_W6_Ntot_1000000_Njob_10000/job_output_*/events_*.lhe


# running parton shower

Download and install Pythia8 from here: https://www.pythia.org/.
You should use option to install LHAPDF and PYTHON packages as well

To use Pythia8 in python you need to set the PYTHONPATH environment variable appropiatly, e.g:
    export PYTHONPATH=/vols/cms/dw515/HH_reweighting/HH_powheg/pythia8310/lib:$PYTHONPATH 

The script `shower_events.py` will run pythia8 using the CMS CP5 tune. Higgs bosons are decayed into taus. The inputs to the script are the lhe file and a pythia8 command file. The command file depends on whether you are showering events produced by POWHEG or MadGraph

To shower POWHEG events use `scripts/pythia_cmnd_file_powheg`, e.g:

    python scripts/shower_events.py -c scripts/pythia_cmnd_file_powheg -i /vols/cms/dw515/HH_reweighting/HH_powheg/POWHEG-BOX-V2/ggHH_v6/run_nnpdf_sm_v3/cmsgrid_final_all_reweighted_full.lhe -o output_powheg_pythia.root

To shower Madgraph events use `scripts/pythia_cmnd_file`, e.g

    python scripts/shower_events.py -c scripts/pythia_cmnd_file -i batch_job_outputs/HH_loop_sm_twoscalar_SM_Ntot_200000_Njob_10000/cmsgrid_final_all_reweighted_full.lhe -n 1000 -o output_mg_pythia.root
