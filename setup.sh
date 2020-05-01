conda_installed=$(conda list | grep 'conda: command not found')
if [ '$conda_installed' != '' ]; then
    wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    Miniconda3-latest-Linux-x86_64.sh
    mv Miniconda3-latest-Linux-x86_64.sh ~/Downloads
fi

conda install pytorch
python3.7 -m pip install -r requirements.txt
sudo apt -y install libportaudio2
echo "Finished installation"

# Possible fix for "This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem."
# sudo apt-get install libxkbcommon-x11-dev

## Possible fix for webrtcvad failure
#sudo apt install python3 python3-dev build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev python-pip