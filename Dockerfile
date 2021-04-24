FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

MAINTAINER Dominik Baack <dominik.baack@tu-dortmund.de>

ARG USER

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -qy --no-install-recommends\
        openssh-server\
        nis\
        git\
        vim\
        htop\
        rsync\
        emacs\
        build-essential\
        curl\
        wget\
        rsync\
        unzip\
        locales locales-all &&\
        locale-gen en_US.UTF-8 && locale-gen de_DE.UTF-8

COPY ./resource/nsswitch.conf /etc/nsswitch.conf

#Set Domainname: SFB876

# Install Anaconda
#ENV PATH="/opt/miniconda3/bin:$PATH" 
#RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh && \
#    HOME=/opt /usr/bin/bash Miniconda3-py38_4.9.2-Linux-x86_64.sh -b && \
#    rm Miniconda3-py38_4.9.2-Linux-x86_64.sh && \
#    conda update conda && \
#    conda update --all

#&& \
#    echo "# HostKey for protocol version 1" >  /etc/ssh/sshd_config.d/settings.conf && \
#    echo "HostKey /etc/ssh/ssh_host_key"    >> /etc/ssh/sshd_config.d/settings.conf && \
#    echo "# HostKeys for protocol version 2">> /etc/ssh/sshd_config.d/settings.conf && \
#    echo "HostKey /etc/ssh/ssh_host_rsa_key">> /etc/ssh/sshd_config.d/settings.conf && \
#    echo "HostKey /etc/ssh/ssh_host_dsa_key">> /etc/ssh/sshd_config.d/settings.conf && \
#    echo "HostKey /etc/ssh/ssh_host_ecdsa_key" >> /etc/ssh/sshd_config.d/settings.conf

RUN rm -f /etc/update-motd.d/* && \
echo "printf \"Hi $USER, welcome to your docker image\n Check https://projekte.itmc.tu-dortmund.de/projects/cluster/wiki/Running_Your_Software for a guide.\n Need anaconda? run 'source /opt/miniconda3/bin/activate' and you're good to go. \n\"" >> /etc/update-motd.d/10-help-text

COPY ./resource/startup.sh /startup.sh
RUN chmod +x /startup.sh

COPY ./resource/ssh/ssh_host* /etc/ssh/
RUN chmod 600 /etc/ssh/ssh_host* && chmod 644 /etc/ssh/ssh_host*.pub

ENV SSH_PORT=30100

# Change to non-root privilege
#USER dbaack
#WORKDIR /home/${USER}

ENTRYPOINT ["/startup.sh"]