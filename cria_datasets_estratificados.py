# Cria datasets de treino e teste com Stratified K-Fold

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

df_bert = pd.read_parquet(
    "data/questoes_tokenized_bert.parquet", engine="fastparquet"
)

df_tratados = pd.read_parquet("data/questoes_tratadas.parquet")

df = df_bert.merge(df_tratados[["id", "texto_lem"]], on="id", how="left")
df["estrato"] = df["materia"] + " - " + df["topico"]

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

os.makedirs("./data/estratos", exist_ok=True)

for i, (train_idx, test_idx) in enumerate(skf.split(df, df["estrato"])):
    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]

    df_train = df_train.drop(columns=["estrato"])
    df_test = df_test.drop(columns=["estrato"])

    # Salvar cada fold em CSV (ou outro formato)
    df_train.to_parquet(f"./data/estratos/train_fold_{i+1}.parquet", index=False)
    df_test.to_parquet(f"./data/estratos/test_fold_{i+1}.parquet", index=False)

    print(f"Fold {i+1}: treino = {len(df_train)}, teste = {len(df_test)}")
