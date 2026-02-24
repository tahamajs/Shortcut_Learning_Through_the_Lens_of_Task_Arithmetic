FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace
COPY env/environment.yml /tmp/environment.yml
RUN pip install --no-cache-dir \
    numpy scikit-learn matplotlib seaborn umap-learn \
    transformers datasets detoxify joblib

COPY . /workspace
ENV PYTHONPATH=/workspace/src:$PYTHONPATH

CMD ["bash"]
