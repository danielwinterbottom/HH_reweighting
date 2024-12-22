Following this paper:

https://arxiv.org/pdf/2405.09201

setup PYTHONPATH environment:

    export PYTHONPATH=/vols/cms/dw515/PYTHIA/pythia8310/lib/:$PYTHONPATH:/vols/cms/dw515/HH_reweighting/HHReweighter/:/vols/cms/dw515/HH_reweighting/HHReweighter/python::/vols/cms/dw515/HH_reweighting/python

process events:

    python scripts/shower_events_lep_study.py -c scripts/pythia_cmnd_file_lep_study -i HHReweighter/MG5_aMC_v3_5_5/ee_To_tata_noA/test_pack/events_100k.lhe -n 1
0

To force tau decays into a1 3-prongs use:

    15:onMode = off
    15:onIfMatch = 16 -211 211 -211

or to force decays into pions use:

    15:onMode = off
    15:onIfMatch = 16 -211

15:oneChannel = 1 1.0 0 211 -16

Also need to use:
    TauDecays:externalMode = 0
Otherwise correlations with beams are not accounted for properly

Note that it was noticed that the elements of Cij are not correct when decaying taus from MG using pythia which is still not understood. If taus are decayed using Madgraph then it works properly.

To decay taus in madgraph (from Z/gamma->tautau) do:

    import model sm-lepton_masses
    add model taudecay_UFO
    generate e+ e- > pi+ vt~ pi- vt / vt vt~ h
    output name_of_output
    launch

Note that sm-lepton_masses model is needed otherwise tau width is set to 0 and this has issues for the decay
Also note that I had issues producing gridpacks for the above which wasn't solved yet, but I was able to produce 100k events (not from gridpack)
