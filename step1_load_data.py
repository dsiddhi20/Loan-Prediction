import pandas as pd

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print("✅ Train data shape:", train_df.shape)
print("✅ Test data shape:", test_df.shape)

print("\nTrain columns:")
print(train_df.columns)

print("\nTest columns:")
print(test_df.columns)
