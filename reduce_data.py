import pandas as pd

print("Loading large dataset...")

df = pd.read_csv("data.csv")

# Take smaller sample
df_small = df.sample(n=50000, random_state=42)

# Save new file
df_small.to_csv("data_small.csv", index=False)

print("✅ Smaller dataset created: data_small.csv")