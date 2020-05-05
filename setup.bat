curl https://www.python.org/ftp/python/3.7.7/python-3.7.7-amd64.exe -o %userprofile%/Downloads/python-3.7.7-amd64.exe
%userprofile%/Downloads/python-3.7.7-amd64.exe

curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o %userprofile%/Downloads/Miniconda3-latest-Windows-x86_64.exe
%userprofile%/Downloads/Miniconda3-latest-Windows-x86_64.exe

conda -y install pytorch
start cmd /k %userprofile%/miniconda3/Scripts/activate base
cd /D "%~dp0"
pip install -r requirements.txt

plaidml-setup