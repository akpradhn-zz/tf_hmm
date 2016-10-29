FROM ubuntu:14.04

MAINTAINER kesmarag@gmail.com

ENV TENSORFLOW_VERSION 0.11.0rc0

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3 \
        python3-dev \
        python3-numpy \
        python3-pip \
        python3-scipy \
	    python3-tk \
        python3-pandas \
        rsync \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN pip3 --no-cache-dir install \
        ipykernel \
        jupyter \
        pandas \
        matplotlib \
        && \
    python3 -m ipykernel.kernelspec

# Install TensorFlow CPU version from central repo
RUN pip3 --no-cache-dir install --upgrade \
    http://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-${TENSORFLOW_VERSION}-cp34-cp34m-linux_x86_64.whl
# --- ~ DO NOT EDIT OR DELETE BETWEEN THE LINES --- #

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Copy sample notebooks.
RUN mkdir /tf_hmm
RUN mkdir /tf_hmm/.logs
COPY tf_hmm.py /tf_hmm/tf_hmm.py
COPY toy_dataset.py /tf_hmm/toy_dataset.py
COPY notebook.ipynb /tf_hmm/notebook.ipynb
COPY test.py /tf_hmm/test.py

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /
RUN chmod +x /run_jupyter.sh

# TensorBoard script.
COPY run_tensorboard.sh /
RUN chmod +x /run_tensorboard.sh

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

WORKDIR "/tf_hmm"

CMD ["/run_jupyter.sh"]
