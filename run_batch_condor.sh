#!/bin/bash

input=$1
output=$2
skip=$3
rw_options=$4

export PYTHONPATH=/vols/cms/dw515/PYTHIA/pythia8310/lib/:$PYTHONPATH:/vols/cms/dw515/HH_reweighting/HHReweighter/:/vols/cms/dw515/HH_reweighting/HHReweighter/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vols/cms/dw515/PYTHIA/pythia8310/lib/
#export PYTHIA8DATA="/vols/cms/dw515/PYTHIA/pythia8310/share/Pythia8/xmldoc"

source /vols/cms/dw515/miniconda3/etc/profile.d/conda.sh
conda activate hh_regression

cd /vols/cms/dw515/HH_reweighting/HHReweighter

python ../scripts/shower_events_4b.py -c ../scripts/pythia_cmnd_file -i ../${input} -o ../${output} -n 2000 -s ${skip}
python ../scripts/reweight_showered_events.py -i ../${output} ${rw_options}

