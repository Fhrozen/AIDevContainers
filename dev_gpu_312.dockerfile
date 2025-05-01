ARG TAG_LABEL="12.6.3-base-ubuntu24.04"
FROM nvidia/cuda:${TAG_LABEL}

ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=user
ARG USER_ID=1000
ARG GROUP_ID=1000

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
    rm -rf /tmp/* 

RUN if [ -z "$(getent group ${GROUP_ID})" ]; then \
    groupadd -g ${GROUP_ID} "${USERNAME}"; \
    else \
    existing_group="$(getent group $GROUP_ID | cut -d: -f1)"; \
    if [ "${existing_group}" != "${USERNAME}" ]; then \
    groupmod -n "${USERNAME}" "${existing_group}"; \
    fi; \
    fi && \
    if [ -z "$(getent passwd $USER_ID)" ]; then \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} "${USERNAME}"; \
    else \
    existing_user="$(getent passwd ${USER_ID} | cut -d: -f1)"; \
    if [ "${existing_user}" != "${USERNAME}" ]; then \
    usermod -l "${USERNAME}" -d /home/"${USERNAME}" -m "${existing_user}"; \
    fi; \
    fi

RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/g' /home/${USERNAME}/.bashrc && \
    mkdir -p /workspaces && \
    chown -R ${USERNAME}:${USERNAME} /workspaces

USER ${USERNAME}
WORKDIR /workspaces
ENV PATH=/workspaces/venv/bin:${PATH}
# Run mains
RUN python3 -m venv /workspaces/venv && \
    pip install torch torchvision torchaudio numpy matplotlib && \
    rm -rf /home/${USERNAME}/.cache/pip

ENV TZ=Etc/UTC
