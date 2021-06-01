FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends\
       openssh-server \
       nis \
       git \
       nano \
       build-essential \
       curl \
       wget \
       rsync \
       unzip \
       htop \
       tmux \
       cmake \
       less \
       default-jre

# Install Anaconda
ENV PATH="/opt/miniconda3/bin:$PATH"
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh && \
    HOME=/opt /usr/bin/bash Miniconda3-py38_4.9.2-Linux-x86_64.sh -b && \
    rm Miniconda3-py38_4.9.2-Linux-x86_64.sh && \
    conda update conda && \
    conda update --all
ADD nsswitch.conf /etc/nsswitch.conf
ARG USER
#RUN echo "PORT 2242" > /etc/ssh/sshd_config.d/customport.conf
RUN echo "printf \"YOU'RE TRAPPED IN A CONTAINER, DUMMY! \n\"" >> /etc/update-motd.d/10-help-text
RUN echo "printf \"Need anaconda? run 'source /opt/miniconda3/bin/activate' and you're good to go. \n\"" >> /etc/update-motd.d/10-help-text
RUN apt-get install -yf locales locales-all && locale-gen en_US.UTF-8 && locale-gen de_DE.UTF-8


ADD startup.sh /startup.sh
RUN chmod +x /startup.sh
#EXPOSE 2242
#ENTRYPOINT service nis start && service ssh start && su - $USER
#ENTRYPOINT service nis start && su - $USER source /opt/miniconda3/bin/activate
ENTRYPOINT ["/startup.sh"]
CMD ["/usr/bin/bash", "--login"]