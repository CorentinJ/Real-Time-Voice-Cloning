conda_installed=$(conda list | grep 'conda: command not found')
if [ '$conda_installed' != '' ]; then
    wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh
    rm Miniconda3-latest-Linux-x86_64.sh ~/Downloads
fi

conda install pytorch
sudo apt -y install libportaudio2 gcc
python3.7 -m pip install -r requirements.txt

## Future AMD setup (needs tensorflow api v2)
amd='FALSE'
if [ $amd == 'TRUE' ]; then
    sudo apt update
    sudo apt -y dist-upgrade
    sudo apt install libnuma-dev
    sudo reboot

    wget -q -O - https://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
    sudo apt update
    sudo apt install rocm-dkms
    sudo usermod -a -G video $LOGNAME
    echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf
    echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf
    sudo reboot
fi

echo "Finished installation"

## Possible fix for "This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem."
# sudo apt-get install libxkbcommon-x11-dev

## Possible fix for webrtcvad failure
#sudo apt install python3 python3-dev build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev python-pip