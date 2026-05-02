import torch
import pandas as pd
from ModelFitter import ModelFitter

def main():
    df = pd.read_excel("CooksBranch.xlsx")
    # Filter for the columns used in the distribution boundary analysis
    df = df[df["tag"] < 38000]
    # Filter for Shortleaf and Loblolly pine
    df = df[df['spcode'].isin(['PINTAE', 'PINECH'])]
    print("Species Counts:")
    print(df["spcode"].value_counts())
    # Mapping PINTAE to 0 and PINECH to 1
    df['sp_binary'] = df['spcode'].map({'PINTAE': 0, 'PINECH': 1})
    x = df["y"].values 
    x = (x - x.min()) / (x.max() - x.min()) * 31
    tensor_x = torch.tensor(x, dtype=torch.float32)
    tensor_y = torch.tensor(df["sp_binary"].values, dtype=torch.float32)
    model = ModelFitter(tensor_x, tensor_y, 1)
    model.fit_change_point_model()
    model.plt_model()

if __name__ == "__main__":
    main()
