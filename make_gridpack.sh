python make_reweight_card.py # add option to do different masses
cp reweight_card.dat MG5_aMC_v2_6_7/SM_hh/Cards/.
cd MG5_aMC_v2_6_7/SM_hh
#{ echo 0; echo set nevents 100; echo set gridpack True; } | python2 bin/generate_events SM_hh
cd ../../
mkdir -p gridpack_SM_hh
cd gridpack_SM_hh
tar -xvf ../MG5_aMC_v2_6_7/SM_hh/SM_hh_gridpack.tar.gz 
mkdir -p madevent/Events/pilotrun
cd madevent
cp Cards/reweight_card.dat Cards/reweight_card.dat.backup
{ echo reweight=OFF ; echo 0; echo set nevents 1 ; echo set gridpack False; echo set use_syst False ; } | bin/generate_events template_lhe
cp Events/template_lhe/unweighted_events.lhe.gz Events/pilotrun/.
sed -n '/^launch/q;p' Cards/reweight_card.dat.backup > Cards/reweight_card.dat
echo "launch" >> Cards/reweight_card.dat
echo "0" | ./bin/madevent --debug reweight pilotrun
cp Cards/reweight_card.dat.backup Cards/reweight_card.dat
