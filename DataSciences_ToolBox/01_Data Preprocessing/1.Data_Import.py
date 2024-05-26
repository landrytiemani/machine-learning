# Purpose: This script demonstrates how to import data from various file formats (CSV, SPSS/SAV, and Excel/XLSX) into Pandas DataFrames.

import pandas as pd


# Import CSV Data
hiccups = pd.read_csv("./data/Hiccups.csv")
print("\nHiccups DataFrame:")
print(hiccups.info())  # Show DataFrame structure and data types


# Import SPSS/SAV Data (requires pyreadstat package)
try:
    import pyreadstat
    chickflick, _ = pyreadstat.read_sav("./data/ChickFlick.sav")  
    print("\nChickFlick DataFrame:")
    print(chickflick.info())
except ImportError:
    print("pyreadstat not found. Install with 'pip install pyreadstat' to read SAV files.")


# Import XLSX Data
texting = pd.read_excel("./data/Texting.xlsx")
print("\nTexting DataFrame:")
print(texting.info())
