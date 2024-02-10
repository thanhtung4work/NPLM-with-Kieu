import re

import pandas as pd

def transform_row(row):
    """
    Cleans and transforms a given text row by performing the following steps:
    
    1. Removes leading digits and dots.
    2. Removes trailing dots, commas, and question marks.
    3. Replaces specific punctuation marks and symbols with spaces.
    4. Strips leading and trailing whitespaces.
    
    Parameters:
    - row (str): The input text row to be transformed.
    
    Returns:
    str: The cleaned and transformed text row.
    """
    # row = row.encode("utf-8")
    row = re.sub(r"^[0-9\.]+", "", row)
    
    row = re.sub(r"[\.,\?]+$", "", row)
    
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("  ", " ").replace("\\", "")
    
    row = row.strip()
    return row 

df = pd.read_csv("data/kieu.txt", sep="/", names=["row"], encoding="utf8").dropna()
df.head(10)

df["row"] = df.row.apply(transform_row)
df.head(10)
df.to_csv('data/data.txt', sep='\n', index=False)