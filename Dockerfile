FROM python:3
USER root

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i en_US en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN apt-get install -y vim less
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

COPY ./req.txt /req.txt
RUN pip install -r /req.txt

ENV USER_ID=1000 \
    GROUP_ID=1000 \
    USER_NAME=user
# ENV USER_NAME=user

RUN useradd -m $USER_NAME
# RUN usermod -u $USER_ID $USER_NAME && \
#     groupmod -g $GROUP_ID $USER_NAME

USER $USER_NAME

# ENV HOME=/home/user/
# RUN mkdir $HOME/pytorchStudy && chown $USER_ID:$GROUP_ID $HOME/pytorchStudy
# RUN mkdir $HOME/qwert
