FROM python:3.11-slim

WORKDIR /app

RUN pip install --upgrade pip

COPY . .

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    chromium \
    && rm -rf /var/lib/apt/lists/*

######################
# Install Docker (I don't know why but it needed this at one point) 
RUN mkdir -p /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

RUN apt-get update && apt-get install -y docker-ce-cli && rm -rf /var/lib/apt/lists/*
######################

# Install forbiddenfruit from source (workaround required for some reason)
RUN pip install wheel setuptools
RUN git clone https://github.com/clarete/forbiddenfruit.git /tmp/forbiddenfruit
RUN cd /tmp/forbiddenfruit && python setup.py install && cd /app && rm -rf /tmp/forbiddenfruit

RUN pip install .

EXPOSE 2024

CMD ["langgraph", "dev", "--host", "0.0.0.0"]
