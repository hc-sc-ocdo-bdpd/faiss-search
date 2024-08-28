# FAISS Retrieval Pipeline

<br>

## Overview

This library contains tools designed encapsulate the FAISS-embedding pipeline, giving users the ability to chunk and embed a dataset of text documents and perform queries to retrieve the most relevant documents.

<br>

## Installation and Dependencies

The library can be installed using

```
pip install git+https://github.com/hc-sc-ocdo-bdpd/faiss-search.git
```

Note that this project is downstream from `file-processing-tools` and uses it as a dependency.
It also requires `llama-cpp` as an optional dependency to load embedding models from a `.gguf` file which. This must be installed using docker.

<br>

# Getting Started

The faiss search library currently offers two imports: `SearchDirectory` and `DatasetVariability` that encapsulate the retrieval pipeline and generate statistics about the dataset, respectively. There is also extended functionality available for working with FAISS indexes.

<br>

## Search Directory

The `SearchDirectory` class contains the end-to-end functionality that can create a searchable database from a directory that can retrieve documents with similar semantic meanings to user inputted queries.

<br>

### File Structure

The functions contained within `SearchDirectory` are designed to manipulate existing data and store it in formats that lend themselves better search features. All of the files created will be stored in the folder specified when the `SearchDirectory` object is created. For example, if the following code is run it will create a `SearchDirectory` object that saves any files that it creates to `path/to/folder`.

```python
from file_processing import SearchDirectory
search = SearchDirectory("path/to/folder")
```

If there are already files contained in `path/to/folder`, such as a `.faiss` file, it will be able to read and load these files so that previous steps do not need to be recomputed.

Once all steps are completed, the following files will be contained in the specified folder:

| File | Generated by | Purpose |
| ---- | ------------ | ------- |
| `report.csv` | `report_from_directory()` | Contains text info and metadata from files in a given directory |
| `data_chunked.csv` | `chunk_text()` | Contains the chunked text data and corresponding file paths |
| `setup_data.json` | `chunk_text()` and `load_embedding_model()` | Contains the number of chunks and the name of the embedding model being used as well as if it's a `.gguf` model |
| `embedding_batches/` | `embed_text()` | Contains embedding batches in the form of `.npy` files |
| `embeddings.npy` | `embed_text()` | Contains the complete embeddings of the chunks data |
| `index.faiss` | `create_flat_index()`, `create_ivf_flat_index()`, and `create_hnsw_index()` | Contains the embeddings in the form of a searchable FAISS index |

<br>

### Generating Report
Creates - `report.csv`

If working with a directory of files, the `report_from_directory()` function generates a `report.csv` file that contains the text and metadata of any text-based files in that directory.

```python
from file_processing import SearchDirectory
search = SearchDirectory("path/to/folder")
search.report_from_directory("text/documents/directory/path")
```

<br>

### Chunking Text
Parameters - `input_file_path`, `document_path_column`, `document_text_column`, `chunk_size`, `chunk_overlap`

Creates - `data_chunked.csv`, `setup_data.json`

Many embedding models and LLMs have a limited context window. This means any large text files need to be broken down into chunks before being passed into these models. The `chunk_text()` method is used for this purpose.

It takes a `.csv` file containing a text field and a file path field as input. The `document_path_column` and `document_text_column` parameters are used to specify the column names in the `.csv`.

```python
from file_processing import SearchDirectory
search = SearchDirectory("path/to/folder")
search.chunk_text("path/to/csv/file.csv",
                  document_path_column="file path",
                  document_text_column="content")
```

Alternatively, if a `report.csv` file was already generated and is contained in the folder then no `.csv` file needs to be specified and the function will use the `report.csv` file as the input.

```python
from file_processing import SearchDirectory
search = SearchDirectory("path/to/folder")
search.report_from_directory("text/documents/directory/path")
search.chunk_text()
```

Both of these approaches will produce a chunked CSV file.

<br>

### Loading Embedding Model
Parameters - `model_name`, `gguf`

Creates - `setup_data.json`

The `load_embedding_model()` function is used to specify and load the embedding model that will be used in the text embedding and search steps. This function supports models that are hosted in the `sentence_transformers` library when `gguf` is set to `False` and will load a specified `.gguf` file when set to `True`.

```python
search.load_embedding_model("paraphrase-MiniLM-L3-v2")
```

<br>

### Text Embedding
Parameters - `row_start`, `row_end`, `batch_size`

Creates - `embedding_batches/`, `embeddings.npy`

The `embed_text()` function takes in the `data_chunked.csv` file and outputs an embedding file called `embeddings.npy`. Because the embeddings can be a time intensive computation, the embeddings are saved in batches of a specified size to the `embedding_batches/` folder in order to save progress. Once all of the chunk embeddings are saved in the `embedding_batches` folder the embeddings are combined and saved to `embeddings.npy`.

```python
from file_processing import SearchDirectory
search = SearchDirectory("path/to/folder")
search.report_from_directory("text/documents/directory/path")
search.chunk_text()
search.load_embedding_model("paraphrase-MiniLM-L3-v2")
search.chunk_text(batch_size=100)
```

<br>

### FAISS Index Creation
Parameters - `embeddings`

Creates - `index.faiss`

There are a few different functions to create FAISS indexes adopted from the `file_processing.faiss_index` library with the same functionality. The only difference is that the FAISS index is automatically saved to the folder. If the embeddings is not specified, it will check if `embeddings.npy` is contained in the folder and it will use that file.

```python
from file_processing import SearchDirectory
search = SearchDirectory("path/to/folder")
search.report_from_directory("text/documents/directory/path")
search.chunk_text()
search.load_embedding_model("paraphrase-MiniLM-L3-v2")
search.chunk_text(batch_size=100)
search.create_ivf_flat_index(nlist=16)
```

<br>

### Search Function
Parameters - `query`, `k`, `args`

The `search()` function takes in a query and returns the `k` closest matching chunks and corresponding file paths. The function also can take in arguments that can specify the FAISS index search hyperparameters.

```python
from file_processing import SearchDirectory
search = SearchDirectory("path/to/folder")
search.report_from_directory("text/documents/directory/path")
search.chunk_text()
search.load_embedding_model("paraphrase-MiniLM-L3-v2")
search.chunk_text(batch_size=100)
search.create_ivf_flat_index(nlist=16)
search.search("What is the meaning of life, the universe, and everything?", k=3, nprobe=2)
```

<br>

## FAISS Indexes

The FAISS index functionality is utilized by `SearchDirectory` but can also be called on its own if working directly with these indexes.

The `faiss_index` import offers a collection of functions that make it easy to interface with FAISS indexes. A `faiss_index` object can be created by either:

* calling one of the create index methods such as `faiss_index.create_flat_index(embeddings)` or `faiss_index.create_ivf_index(embeddings, nlist=16)`.
* loading an index from a `.faiss` file using `faiss_index.load_index("path/to/file.faiss")`.

Once an index is created it can be queried. This involves providing a query vector as an input and the index will return the nearest `k` vectors contained in the index (as found by that algorithm). Consider the example below to view the functionality:

```python
index = faiss_index.load_index("path/to/file.faiss")
nearest_three_vectors = index.query(query_vector, k=3)
```

For large numbers of documents, creating the index can take a while so it is often a good idea to save the file to be loaded in for future use. This can be done by specifying the file path when creating the index or by calling `save()`.

```python
# save the index when creating it
index = faiss_index.create_flat_index(embeddings, "path/to/save.faiss")
# save the index afetr creating it
index = faiss_index.create_flat_index(embeddings)
index.save("path/to/save.faiss")
```

The ability to create indexes is limited to a select number of common indexes. More complex indexes can still be loaded and queried as with the other indexes but does not come with the ability to adjust hyperparameters during the query.
