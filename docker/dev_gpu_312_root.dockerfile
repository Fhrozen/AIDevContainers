ARG TAG_LABEL="12.6.3-base-ubuntu24.04"
FROM nvidia/cuda:${TAG_LABEL}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
        automake \
        autotools-dev \
        bc \
        build-essential \
        curl \
        espeak-ng \
        ffmpeg \
        git \
        gnupg2 \
        libffi-dev \
        libncurses5-dev \
        libsm6 \
        libtool \
        libxext6 \
        python3-full \
        python3-dev \
        python3-pip \
        sudo \
        software-properties-common \
        unzip \
        wget \
        zip \
        zlib1g-dev \
        && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get -y install --no-install-recommends \
        git-lfs \
        && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* && \
    mkdir -p /workspaces

WORKDIR /workspaces
ENV PATH=/workspaces/venv/bin:${PATH}
# Run mains
RUN python3 -m venv /workspaces/venv && \
    pip install torch torchvision torchaudio numpy matplotlib && \
    rm -rf /root/.cache/pip

ENV TZ=Etc/UTC
