wget https://launchpad.net/mg5amcnlo/3.0/3.5.x/+download/MG5_aMC_v3.5.1.tar.gz -O MG5_aMC_v3.5.1.tar.gz 
tar -xvf MG5_aMC_v3.5.1.tar.gz

# make sure python3.7 is installed
version=$(python3 -V 3>&1 | grep -Po '(?<=Python )(.+)')
parsedVersion=$(echo "${version//./}")
if [[ "$parsedVersion" -lt "370" ]]
then 
    echo Invalid python3 version $version, installing version 3.7

    if ! [ -d $HOME/.pyenv ]; then
      echo $HOME/.pyenv
      curl https://pyenv.run | bash
    fi
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"   
    pyenv install -s 3.7.17
    pyenv global 3.7.17
fi

python3 -m venv env
echo \#!/bin/bash >> env.sh
echo source $PWD/env/bin/activate > env.sh
chmod +x env.sh

if [[ "$parsedVersion" -lt "370" ]]
then
    pyenv global system
fi

source env.sh
python3 -m pip install six

# install madanalysis5 by hand (not working at the moment)
cd MG5_aMC_v3_5_1/HEPTools
wget https://madanalysis.irmp.ucl.ac.be/raw-attachment/wiki/MA5SandBox/ma5_v1.9.10.tgz -O ma5_v1.9.10.tgz
tar -xvf ma5_v1.9.10.tgz 
cd madanalysis5
echo 1 | python2 bin/ma5
cd ../../

echo "install install ninja ; install collier;" | python3 bin/mg5_aMC
cd ..
