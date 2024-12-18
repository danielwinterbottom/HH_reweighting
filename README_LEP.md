setup PYTHONPATH environment:

    export PYTHONPATH=/vols/cms/dw515/PYTHIA/pythia8310/lib/:$PYTHONPATH:/vols/cms/dw515/HH_reweighting/HHReweighter/:/vols/cms/dw515/HH_reweighting/HHReweighter/python::/vols/cms/dw515/HH_reweighting/python

process events:

    python scripts/shower_events_lep_study.py -c scripts/pythia_cmnd_file_lep_study -i HHReweighter/MG5_aMC_v3_5_5/ee_To_tata_noA/test_pack/events_100k.lhe -n 10
