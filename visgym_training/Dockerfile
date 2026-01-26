FROM hiyouga/llamafactory:latest

# Basic setup
RUN apt-get update && apt-get install -y tmux
RUN pip install --no-cache-dir jupyterlab

# Install RP and related tools
RUN pip install einops asciinema
RUN pip install rp==0.1.1333

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# System packages
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    gnupg \
    curl \
    ffmpeg \
    rclone \
    nvtop \
    tree \
    git-lfs \
    shellinabox \
    zsh \
    iperf3 \
    stow


# Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true

# Code Server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# VSCode Extensions (installing as root for global access, or switch user if needed)
RUN code-server --install-extension ms-python.python || true
RUN code-server --install-extension ms-pyright.pyright || true
RUN code-server --install-extension ms-toolsai.jupyter || true

# Decord fix
RUN pip uninstall -y decord || true
RUN pip install decord --upgrade --force --no-cache

# Final touches
RUN pip install imgcat
RUN pip install rp --upgrade --no-cache

WORKDIR /home/clouduser
