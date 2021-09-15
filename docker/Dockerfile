
ARG TAG=1.8.1-cuda10.2-cudnn7-devel
ARG DS=https://github.com/microsoft/DeepSpeed.git
ARG DALLE=https://github.com/lucidrains/DALLE-pytorch.git

FROM pytorch/pytorch:$TAG

# Debian Deps
RUN apt-get -y update   \
 && apt-get -y install  \
 cmake                  \
 git                    \
 gcc                    \
 llvm-9-dev             \
 libaio-dev             \
 vim                    \
 wget

# DeepSpeed
RUN git clone $DS /tmp/DeepSpeed    \
 && cd /tmp/DeepSpeed               \
 && DS_BUILD_OPS=1                  \
    ./install.sh -r

# Dalle-pytorch
RUN pip install "git+$DALLE"

WORKDIR dalle
