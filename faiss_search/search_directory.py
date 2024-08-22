import os
import re
import json
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from file_processing import Directory

class SearchDirectory:
    def __init__(self, folder_path: str) -> None:
        """
        Initializes the SearchDirectory object with paths to data and model files.

        :param folder_path: Path to the folder containing data and setup files.
        """
        self.folder_path = folder_path
        # get chunking file path
        if os.path.exists(os.path.join(self.folder_path, "data_chunked.csv")):
            self.chunks_path = os.path.join(self.folder_path, "data_chunked.csv")
        else:
            self.chunks_path = None
        # get json data
        if os.path.exists(os.path.join(self.folder_path, "setup_data.json")):
            with open(os.path.join(self.folder_path, "setup_data.json"), 'r') as f:
                setup_data = json.load(f)
                self.encoding_name = setup_data['encoding_model']
                self.n_chunks = setup_data['number_of_chunks']
        else:
            self.n_chunks = None
            self.encoding_name = None

    def _get_text_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Splits the input text into smaller chunks with specified size and overlap.

        :param text: The text to be split into chunks.
        :param chunk_size: Number of characters in each chunk.
        :param chunk_overlap: Number of overlapping characters between chunks.

        :return: A list of text chunks.
        """
        chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk in splitter.split_text(text):
            chunks.append(chunk)
        return chunks
    
    def _save_to_json(self) -> None:
        """
        Saves the encoding model name and number of chunks to a JSON file in the folder.
        """
        setup_data = {
            'encoding_model': self.encoding_name,
            'number_of_chunks': self.n_chunks
        }
        with open(os.path.join(self.folder_path, "setup_data.json"), 'w') as f:
            json.dump(setup_data, f, indent=4)

    def report_from_directory(self, directory_path: str) -> None:
        """
        Generates a report from the specified directory and saves it as 'report.csv'.

        :param directory_path: Path to the directory to generate the report from.
        """
        directory = Directory(directory_path)
        directory.generate_report(
            report_file = os.path.join(self.folder_path,"report.csv"),
            split_metadata=True,
            include_text=True,
        )

    def chunk_text(self,
                   input_file_path: str = None,
                   document_path_column: str = "File Path",
                   document_text_column: str = "Text",
                   chunk_size: int = 1024,
                   chunk_overlap: int = 10) -> None:
        """
        Chunks the text data from a CSV file into smaller pieces and saves the result to 'data_chunked.csv'.

        :param input_file_path: Path to the CSV file containing text to chunk. If None, uses 'report.csv' in the folder.
        :param document_path_column: Column name for file paths in the CSV.
        :param document_text_column: Column name for text content in the CSV.
        :param chunk_size: Number of characters in each chunk.
        :param chunk_overlap: Number of overlapping characters between chunks.

        :raises FileNotFoundError: If no input file is specified and no report exists.
        :raises FileTypeError: If the input file is not a CSV.
        :raises KeyError: If specified columns are not found in the CSV.
        """

        # check if there is a report
        if input_file_path is None:
            if os.path.exists(os.path.join(self.folder_path, "report.csv")):
                input_file_path = os.path.join(self.folder_path, "report.csv")
            else:
                raise FileNotFoundError("No input file specified and no report provided. \
                                        Please provide a file path to a .csv or run 'report_from_directory'.")

        # load into a dataframe
        if input_file_path.lower().endswith('.csv'):
            df = pd.read_csv(input_file_path)
        else:
            raise FileTypeError(f"File path {input_file_path} is not a .csv file.")
        
        # check if the column names are valid
        if document_path_column not in df.columns:
            raise KeyError(f"'{document_path_column}' is not a column in {input_file_path}.")
        elif document_text_column not in df.columns:
            raise KeyError(f"'{document_text_column}' is not a column in {input_file_path}.")

        # Initialize an empty list to collect all rows
        all_new_rows = []

        # Get the total number of rows
        total_rows = len(df)
        print(f"Total rows (excluding header): {total_rows}")

        # Process each row with tqdm to show progress
        for index, row in tqdm(df.iterrows(), total=total_rows, desc="Processing rows"):
            file_path = row[document_path_column]
            content = row[document_text_column]
            
            # Get chunks for the current content
            chunks = self._get_text_chunks(content, chunk_size, chunk_overlap)
            
            # Create new rows for each chunk
            for chunk_text in chunks:
                new_row = {
                    'file_path': file_path,
                    'content': chunk_text
                }
                all_new_rows.append(new_row)

        # Create a new DataFrame from the collected new rows
        chunked_df = pd.DataFrame(all_new_rows)

        # Save the new DataFrame to a new CSV file
        chunked_df.to_csv(os.path.join(self.folder_path, 'data_chunked.csv'), index=False)
        self.chunks_path = os.path.join(self.folder_path, 'data_chunked.csv')
        self.n_chunks = len(chunked_df)
        self._save_to_json()

        print("Chunking complete and saved to 'data_chunked.csv'.")
