import pandas as pd
from transformers import BertTokenizer

# Dados
df = pd.read_parquet("./data/questoes_tratadas.parquet", engine="fastparquet")

# Tokenização
special_tokens = [
    "<num>",
    "<url>",
    "<delta>",
    "<graus>",
    "<raiz>",
    "<pi>",
    "<ohm>",
    "<lambda>",
    "<theta>",
    "<mu>",
    "<soma>",
]
tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
tokenizer.save_pretrained("./tokenizer-custom")

encodings = tokenizer(
    df["texto_clean_cased"].tolist(),
    truncation=True,
    padding="max_length",
    max_length=365,
)

df["input_ids"] = encodings["input_ids"]
df["attention_mask"] = encodings["attention_mask"]

df = df[["id", "materia", "topico", "input_ids", "attention_mask"]]

print("\nDataFrame após a tokenização:")
print(df.head())
df.to_parquet(
    "./data/questoes_tokenized_bert.parquet", engine="fastparquet", index=False
)
