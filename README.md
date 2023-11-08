# HH_reweighting
A repository to develope a ME-based reweighting tool for di-Higgs processes

## Setup 

Download and install Madgraph5 and MadAnalysis5 and download the TRSM files

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

## Using Madanalysis5

Run MadAnalysis5 on an already produced LHE file

	python2 MG5_aMC_v2_6_7/HEPTools/madanalysis5/madanalysis5/bin/ma5 -s scripts/ma5_sm_script 	
