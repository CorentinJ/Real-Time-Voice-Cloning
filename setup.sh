conda_installed=$(conda list | grep 'conda: command not found')
if [ $conda_installed != '' ]; then
    wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/Downloads
    chmod +x ~/Downloads/Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh
    conda create -n RTVC python
    conda activate RTVC
fi

sudo apt -y install python3 python3-dev build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev python-pip libportaudio2
conda update
conda install pytorch -c pytorch
python3.7 -m pip install -r requirements.txt
echo "Finished installation"

# Fix for "This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem."
# sudo apt-get install libxkbcommon-x11-dev
