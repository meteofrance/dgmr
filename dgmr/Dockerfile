FROM horovod/horovod:0.28.1

COPY mf.crt /usr/local/share/ca-certificates/mf.crt

RUN apt update && apt install -y software-properties-common && add-apt-repository ppa:ubuntugis/ppa
RUN apt update && apt install -y libgeos-dev git vim nano sudo libx11-dev tk python3-tk tk-dev libpng-dev libffi-dev dvipng texlive-latex-base openssh-server netcat libeccodes-dev libeccodes-tools

ARG USERNAME
ARG GROUPNAME
ARG USER_UID
ARG USER_GUID
ARG HOME_DIR

RUN pip install --upgrade pip
COPY requirements.txt /root/requirements.txt
RUN set -eux && pip install -r /root/requirements.txt

RUN set -eux && groupadd --gid $USER_GUID $GROUPNAME \
    # https://stackoverflow.com/questions/73208471/docker-build-issue-stuck-at-exporting-layers
    && mkdir -p $HOME_DIR && useradd -l --uid $USER_UID --gid $USER_GUID -s /bin/bash --home-dir $HOME_DIR --create-home $USERNAME \
    && chown $USERNAME:$GROUPNAME $HOME_DIR \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && echo "$USERNAME:$USERNAME" | chpasswd

WORKDIR $HOME_DIR
RUN curl -fsSL https://code-server.dev/install.sh | sh
