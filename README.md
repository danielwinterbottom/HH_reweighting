# HH_reweighting
A repository to develope a ME-based reweighting tool for di-Higgs processes

## Setup 

Download and install Madgraph5 and MadAnalysis5 and download the TRSM files

	./setup_2p6.sh

Create directories for the SM and BSM processes with various combinations of diagrams excluded:

	cd MG5_aMC_v2_6_7
	python2 bin/mg5_aMC -f ../scripts/mg_script


## Using Madanalysis5

Run MadAnalysis5 on an already produced LHE file

	python2 MG5_aMC_v2_6_7/HEPTools/madanalysis5/madanalysis5/bin/ma5 -s scripts/ma5_sm_script 	
