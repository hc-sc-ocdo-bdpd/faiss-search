import os
import pytest
import shutil
import numpy as np
from faiss_search import SearchDirectory

@pytest.fixture(scope="module")
def resource_folder():
    return "tests/resources/sample_text_files"

# Test chunking step

def test_empty_directory(tmp_path):
    SearchDirectory(tmp_path)
    assert not any(os.scandir(tmp_path))
    
def test_chunk_from_report(resource_folder, tmp_path):
    search = SearchDirectory(tmp_path)
    search.report_from_directory(resource_folder)
    assert os.path.exists(tmp_path / "report.csv")
    search.chunk_text()
    assert os.path.exists(tmp_path / "data_chunked.csv")
    assert os.path.exists(tmp_path / "setup_data.json")

def test_chunk_without_file(tmp_path):
    search = SearchDirectory(tmp_path)
    with pytest.raises(Exception):
        search.chunk_text()

def test_load_chunks_without_csv(tmp_path):
    search = SearchDirectory(tmp_path)
    with pytest.raises(Exception):
        search.chunk_text("tests/resources/directory_test_files/Test_excel_file.xlsx")

def test_load_chunks_with_name_issues(tmp_path):
    search = SearchDirectory(tmp_path)
    with pytest.raises(Exception):
        search.chunk_text("tests/resources/directory_test_files/2021_Census_English.csv")
    
@pytest.fixture()
def directory_with_chunks(resource_folder, tmp_path_factory):
    file_path = tmp_path_factory.mktemp("just_chunks")
    search = SearchDirectory(file_path)
    search.report_from_directory(resource_folder)
    search.chunk_text()
    return file_path

def test_load_with_chunks(directory_with_chunks):
    search = SearchDirectory(directory_with_chunks)
    assert search.n_chunks is not None

def test_load_chunks_different_column_names(directory_with_chunks, tmp_path):
    search1 = SearchDirectory(tmp_path)
    search1.chunk_text("tests/resources/document_search_test_files/report_modified.csv",
                       "path",
                       "content")
    search2 = SearchDirectory(directory_with_chunks)
    assert search2.n_chunks == search1.n_chunks
