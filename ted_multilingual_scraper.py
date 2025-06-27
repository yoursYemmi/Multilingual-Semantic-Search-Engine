import pandas as pd

# Load the Parquet file
df = pd.read_parquet(r"C:\Users\PRAVEEN\Downloads\train-00000-of-00001.parquet")  # adjust filename if needed

# Extract Hindi and English from the translation dictionary
df["hindi"] = df["translation"].apply(lambda x: x["hi"])
df["english"] = df["translation"].apply(lambda x: x["en"])

# Keep only these two columns
df = df[["hindi", "english"]]

# Optional: remove rows with missing or very short text
df = df[df["hindi"].str.len() > 10]
df = df[df["english"].str.len() > 10]

# Save to CSV
df.to_csv("ted_hi_en_parallel.csv", index=False, encoding="utf-8-sig")
print(f" Saved {len(df)} sentence pairs to ted_hi_en_parallel.csv")
