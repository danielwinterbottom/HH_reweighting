wget https://launchpad.net/mg5amcnlo/3.0/3.5.x/+download/MG5aMC_LTS_v2.9.16.tar.gz -O MG5aMC_LTS_v2.9.16.tar.gz 
tar -xvf MG5aMC_LTS_v2.9.16.tar.gz

cd MG5_aMC_v2_9_16
# need to make this missing directory or it doesn't work
mkdir Template/loop_material/StandAlone/SubProcesses/MadLoop5_resources

echo install MadAnalysis5 --madanalysis5_tarball=https://launchpad.net/madanalysis5/trunk/v1.9/+download/MadAnalysis5_v1.9.60.tgz | python2 bin/mg5_aMC
echo "install oneloop; install ninja ; install collier;" | python2 bin/mg5_aMC

wget https://gitlab.com/apapaefs/twosinglet/-/archive/master/twosinglet-master.tar.gz?path=loop_sm_twoscalar -O loop_sm_twoscalar.tar.gz

tar -xvf loop_sm_twoscalar.tar.gz

mv twosinglet-master-loop_sm_twoscalar/loop_sm_twoscalar models/.
rm -r twosinglet-master-loop_sm_twoscalar
rm loop_sm_twoscalar.tar.gz

cd ..
