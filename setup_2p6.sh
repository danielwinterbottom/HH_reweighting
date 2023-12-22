wget https://launchpad.net/mg5amcnlo/lts/2.6.x/+download/MG5_aMC_v2.6.7.tar.gz -O MG5_aMC_v2.6.7.tar.gz
tar -xvf MG5_aMC_v2.6.7.tar.gz

cd MG5_aMC_v2_6_7

echo "install oneloop; install ninja ; install collier;" | python2 bin/mg5_aMC

wget https://gitlab.com/danielwinterbottom/twosinglet/-/archive/master/twosinglet-master.tar.gz?path=loop_sm_twoscalar -O loop_sm_twoscalar.tar.gz

tar -xvf loop_sm_twoscalar.tar.gz

mv twosinglet-master-loop_sm_twoscalar/loop_sm_twoscalar models/.
rm -r twosinglet-master-loop_sm_twoscalar
rm loop_sm_twoscalar.tar.gz

python2 bin/mg5_aMC -f ../scripts/mg_script

cp -r BSM_hh SM_hh 
cp ../Cards/param_card_SMlike.dat SM_hh/Cards/param_card.dat
cp ../Cards/reweight_card.dat SM_hh/Cards/.

cd ..
