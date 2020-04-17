FROM tensorflow/tensorflow:nightly-py3
RUN pip3 install jupyter jupyterthemes pandas pyyaml h5py tensorflow_datasets
RUN jt -t onedork
