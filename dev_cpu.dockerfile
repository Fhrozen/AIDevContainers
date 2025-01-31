FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
        build-essential \
        automake \
        autotools-dev \
        bc \
        libffi-dev \
        libtool \
        gnupg2 \
        libncurses5-dev \
        software-properties-common \
        unzip \
        wget \
        zip \
        zlib1g-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV TZ=Etc/UTC
RUN add-apt-repository ppa:git-core/ppa -y && \
    apt update && \
    apt install -y --no-install-recommends git-all && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --tries=3 -nv "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/miniconda && \
    /opt/miniconda/bin/conda config --prepend channels https://software.repos.intel.com/python/conda/ && \
    rm miniconda.sh

ENV PATH=/opt/miniconda/bin:${PATH}

# Run mains
RUN conda install -y python=3.10 pip && \
    conda install -y numpy matplotlib && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    conda clean -a -y && \
    rm -rf /root/.cache/pip
