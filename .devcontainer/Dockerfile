FROM mcr.microsoft.com/vscode/devcontainers/rust:latest

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# Installieren der notwendigen Abhängigkeiten für perf
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends linux-perf g++ cmake

RUN apt-get install -y make

RUN rustup toolchain install nightly && rustup override set nightly \
    && apt-get update && apt-get -y install --no-install-recommends libfmt-dev librange-v3-dev
