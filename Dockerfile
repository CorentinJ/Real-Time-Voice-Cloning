FROM nvcr.io/nvidia/pytorch:21.08-py3

LABEL "coqui"="TTS"
LABEL version="1.0"
LABEL description="coqui TTS DevEnvironment"

# Ensure this docker file is built by VScode.
# Manual / user build of this Dockerfile is NOT supported.
ARG vscode
RUN if [[ -z "$vscode" ]] ; then printf "\nERROR: This Dockerfile NEEDS to be built with VScode !" && exit 1; else printf "VScode is detected: $vscode"; fi

RUN apt-get update \
# Install git and curl
&& apt-get install -y git \
&& apt-get install -y curl

# Install ZSH & OhMyZsh for development inside container
RUN apt-get install -y zsh \
&& sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# will throw error if not installed https://github.com/NVIDIA/NeMo/issues/841
#RUN pip install llvmlite --ignore-installed

#RUN pip install TTS

RUN apt-get install -y libportaudio2

RUN git clone https://github.com/CorentinJ/Real-Time-Voice-Cloning.git