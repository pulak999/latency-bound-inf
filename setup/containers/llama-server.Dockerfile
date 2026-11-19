# One VLM tenant: llama-server from our own sm_110 build, run in its own container.
# The binary, the CUDA libraries and the model are mounted from the host at run time
# (see harness/engines/llama_server.py), so a container tenant runs exactly the
# binaries a host run does. The image only adds what the build links against and
# Ubuntu's base image lacks.
#   docker build -t lbi-llama-server:v1 -f setup/containers/llama-server.Dockerfile setup/containers
FROM ubuntu:24.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
