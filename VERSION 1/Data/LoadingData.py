# Import libraries
import pandas as pd
import zstandard as zstd
import io

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.load_data()

    def load_data(self):
        """
        Load data from a compressed .zst file and return a DataFrame.
        """
        # Create a Zstandard decompressor
        dctx = zstd.ZstdDecompressor()

        # Open the compressed file
        with open(self.file_path, 'rb') as compressed:
            # Create a stream reader to decompress the file
            with dctx.stream_reader(compressed) as reader:
                # Use TextIOWrapper to decode the binary stream
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                # Load data into DataFrame
                df = pd.read_csv(text_stream)
        return df

    def get_full_dataframe(self):
        """
        Return the entire dataset.
        """
        return self.df

    def backrank_and_m2(self):
        """
        Filter the dataset for puzzles with 'mateIn2' and 'backRankMate' themes.
        """
        contains_mate_in_2 = self.df['Themes'].str.contains('mateIn2', na=False)
        contains_backrank_mate = self.df['Themes'].str.contains('backRankMate', na=False)
        m2_and_backrank = self.df[contains_mate_in_2 & contains_backrank_mate].copy()
        return m2_and_backrank
