if not exist %userprofile%/Downloads/python-3.7.7-amd64.exe (
    curl https://www.python.org/ftp/python/3.7.7/python-3.7.7-amd64.exe -o %userprofile%/Downloads/python-3.7.7-amd64.exe
    %userprofile%/Downloads/python-3.7.7-amd64.exe
)

if not exist %userprofile%/Downloads/Miniconda3-latest-Windows-x86_64.exe (
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o %userprofile%/Downloads/Miniconda3-latest-Windows-x86_64.exe
    %userprofile%/Downloads/Miniconda3-latest-Windows-x86_64.exe
)

if not exist %userprofile%/Downloads/vs_BuildTools.exe (
    curl https://download.visualstudio.microsoft.com/download/pr/5e397ebe-38b2-4e18-a187-ac313d07332a/00945fbb0a29f63183b70370043e249218249f83dbc82cd3b46c5646503f9e27/vs_BuildTools.exe -o %userprofile%/Downloads/vs_BuildTools.exe
    %userprofile%/Downloads/vs_BuildTools.exe
)

if not exist vocoder/saved_models (
    python -m pip install gdown
    gdown https://drive.google.com/uc?id=1n1sPXvT34yXFLT47QZA6FIRGrwMeSsZc
    python -c "import zipfile; zipfile.ZipFile('pretrained.zip').extractall()"
)

start cmd /k "%userprofile%/miniconda3/Scripts/activate base & conda install -y pytorch & exit"
cd /D "%~dp0"
pip install -r requirements.txt

:: plaidml-setup