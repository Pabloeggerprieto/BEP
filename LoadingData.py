# Import libraries
import pandas as pd
import zstandard as zstd
import io



### Load data

# Path to .zst file
zst_file_path = 'Data/lichess_db_puzzle.csv.zst'

# Create a Zstandard decompressor
dctx = zstd.ZstdDecompressor()

# Open the compressed file
with open(zst_file_path, 'rb') as compressed:
    # Create a stream reader to decompress the file
    with dctx.stream_reader(compressed) as reader:
        # Use TextIOWrapper to decode the binary stream
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
        # Now you can read from text_stream using pandas
        df = pd.read_csv(text_stream)

def get_df():
    #
    # Entire dataset
    #
    return df

def m2_and_backrank():
    #
    # Mate-in-2 and backrank checkmates only
    # 
    contains_mate_in_2 = df['Themes'].str.contains('mateIn2', na=False)
    contains_backrank_mate = df['Themes'].str.contains('backRankMate', na=False)
    m2_and_backrank = df[contains_mate_in_2 & contains_backrank_mate].copy()
    return m2_and_backrank

    
