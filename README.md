# FAISS Retrieval Pipeline

## Overview

This library contains tools designed encapsulate the FAISS-embedding pipeline, giving users the ability to chunk and embed a dataset of text documents and perform queries to retrieve the most relevant documents.

## Installation and Dependencies

The library can be installed using

```
pip install git+https://github.com/hc-sc-ocdo-bdpd/faiss-search.git
```

This project is downstream from `file-processing-tools` and uses it as a dependency.
It also requires `llama-cpp` as an optional dependency to load embedding models from a `.gguf` file which. This must be installed using docker.
