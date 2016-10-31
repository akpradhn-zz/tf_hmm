FROM kesmarag/research:py3

MAINTAINER Costas Smaragdakis <kesmarag@gmail.com>

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
