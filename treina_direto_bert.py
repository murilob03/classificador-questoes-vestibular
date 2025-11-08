# Treina o BERT (ou DistilBERT) diretamente na tarefa de classificação de tópicos

import pandas as pd
import torch
import numpy as np

import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.utils import compute_class_weight

from transformers import BertForSequenceClassification, BertTokenizer
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback


class QuestoesDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.input_ids = torch.tensor(df["input_ids"].tolist())
        self.attention_mask = torch.tensor(df["attention_mask"].tolist())
        self.labels = torch.tensor(df["label"].tolist())

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

    def __len__(self):
        return len(self.labels)


for FOLD in range(1, 5):
    train_df = pd.read_parquet(
        f"./data/train_fold_{FOLD}.parquet", engine="fastparquet"
    )
    test_df = pd.read_parquet(f"./data/test_fold_{FOLD}.parquet", engine="fastparquet")
    train_df["label"] = train_df["topico"].astype("category").cat.codes
    test_df["label"] = test_df["topico"].astype("category").cat.codes

    label2id = {
        v: k
        for k, v in dict(
            enumerate(train_df["topico"].astype("category").cat.categories)
        ).items()
    }
    id2label = dict(enumerate(train_df["topico"].astype("category").cat.categories))

    train_dataset = QuestoesDataset(train_df)
    test_dataset = QuestoesDataset(test_df)

    model = BertForSequenceClassification.from_pretrained(
        "neuralmind/bert-base-portuguese-cased",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    tokenizer = BertTokenizer.from_pretrained("./tokenizer-custom")
    model.resize_token_embeddings(len(tokenizer))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average="macro")
        f1_weighted = f1_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="macro", zero_division=0)
        precision = precision_score(
            labels, predictions, average="macro", zero_division=0
        )

        return {
            "accuracy": acc,
            "macro-precision": precision,
            "macro-recall": recall,
            "macro-f1": f1_macro,
            "weighted-f1": f1_weighted,
        }

    # Cálculo dos pesos das classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"]),
        y=train_df["label"],
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float)

    def weighted_ce_loss(outputs, labels, num_items_in_batch: int):
        logits = outputs.logits
        weight = class_weights.to(logits.device)  # type: ignore
        return F.cross_entropy(logits, labels, weight=weight)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_threshold=0.005, early_stopping_patience=3
    )

    training_args = TrainingArguments(
        output_dir="./modelos-treinamento",  # Pasta para salvar o modelo
        eval_strategy="epoch",  # Avalia o modelo a cada época
        save_strategy="epoch",  # Salva o modelo a cada época
        save_total_limit=2,  # Mantém apenas os 2 últimos modelos salvos
        num_train_epochs=10,  # Número de épocas
        per_device_train_batch_size=12,  # Tamanho do batch de treino. Diminua se tiver erro de memória (ex: 4)
        per_device_eval_batch_size=24,  # Tamanho do batch de avaliação
        warmup_ratio=0.1,  # Passos de aquecimento do otimizador
        weight_decay=0.01,  # Regularização
        learning_rate=4e-5,  # Taxa de aprendizado
        logging_dir="./logs",  # Pasta para logs
        logging_steps=100,  # Exibe o progresso a cada 100 passos
        load_best_model_at_end=True,  # Carrega o melhor modelo no final do treino
        metric_for_best_model="macro-f1",  # Usa a macro-f1 para decidir o melhor modelo
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        compute_loss_func=weighted_ce_loss,
        callbacks=[early_stopping_callback],
    )

    trainer.train()

    scores = trainer.evaluate()

    for score_name, score_value in scores.items():
        print(f"{score_name}: {score_value}")

    trainer.save_model(f"./modelos/bert/diretao_fold_{FOLD}")
