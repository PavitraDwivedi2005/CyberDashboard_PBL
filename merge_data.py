import pandas as pd
import glob

files = glob.glob("archive/*.csv")

df_list = []

for file in files:
    print(f"Loading {file}")
    df = pd.read_csv(file)

    df.columns = df.columns.str.strip()

    if "Label" in df.columns:
        df = df[df["Label"].str.contains("DoS|BENIGN", case=False, na=False)]

        df["label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

        df = df.drop(columns=["Label"])

        df_list.append(df)

if len(df_list) == 0:
    print("❌ No valid data found")
else:
    final_df = pd.concat(df_list, ignore_index=True)

    final_df = final_df.replace([float('inf'), -float('inf')], 0)
    final_df = final_df.dropna()

    final_df.to_csv("data.csv", index=False)

    print("✅ data.csv created successfully!")