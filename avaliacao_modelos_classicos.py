"""
Script: avaliacao_modelos_classicos.py
Descrição:
    Este script realiza a avaliação comparativa de modelos clássicos de aprendizado de máquina
    (SVM, Regressão Logística, Random Forest e Naive Bayes) na tarefa de classificação automática
    de questões de vestibulares por disciplina. O pipeline inclui vetorização TF-IDF, seleção
    de features via qui-quadrado e validação cruzada com 4 folds.

    Resultados consolidados são salvos em 'resultados.csv'.
"""

# =====================================================================
# Importações
# =====================================================================

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Modelos
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# Pré-processamento e pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

# Métricas
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


# =====================================================================
# Definição dos modelos
# =====================================================================

models = {
    "SVC": LinearSVC(C=0.316, class_weight="balanced", random_state=42),
    "Logistic Regression": LogisticRegression(
        C=10.0, class_weight="balanced", solver="lbfgs", random_state=42, max_iter=1000
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=800,
        class_weight="balanced_subsample",
        max_depth=30,
        random_state=42,
    ),
    "Naive Bayes": MultinomialNB(alpha=0.01),
}


# =====================================================================
# Pipeline base (vetorização + seleção de features + classificador)
# =====================================================================

vectorizer = TfidfVectorizer(
    max_features=None,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.8,
)

selector = SelectKBest(chi2, k=20000)

pipeline = Pipeline(
    [
        ("tfidf", vectorizer),
        ("chi2", selector),
        ("clf", None),  # placeholder do classificador
    ],
    memory="cache_dir",
)


# =====================================================================
# Função de validação cruzada
# =====================================================================

CLASS = "materia"


def validacao_cruzada(modelo, df_treino, df_teste, nome_modelo, fold):
    pipeline.set_params(clf=modelo)
    pipeline.fit(df_treino["texto_lem"], df_treino[CLASS])

    y_pred = pipeline.predict(df_teste["texto_lem"])
    y_true = df_teste[CLASS]

    return {
        "modelo": nome_modelo,
        "fold": fold,
        "acuracia": np.round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": np.round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_weighted": np.round(f1_score(y_true, y_pred, average="weighted"), 4),
        "recall_macro": np.round(recall_score(y_true, y_pred, average="macro"), 4),
        "precision_macro": np.round(
            precision_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
    }


# =====================================================================
# Execução das tarefas (4 folds, todos os modelos)
# =====================================================================

tasks = []
for FOLD in range(1, 5):
    df_treino = pd.read_parquet(
        f"./data/estratos/train_fold_{FOLD}.parquet"
    )
    df_teste = pd.read_parquet(f"./data/estratos/test_fold_{FOLD}.parquet")

    for nome_modelo, modelo in models.items():
        tasks.append((modelo, df_treino, df_teste, nome_modelo, FOLD))

results = Parallel(n_jobs=8, verbose=10)(
    delayed(validacao_cruzada)(modelo, df_treino, df_teste, nome_modelo, fold)
    for modelo, df_treino, df_teste, nome_modelo, fold in tasks
)


# =====================================================================
# Agregação e salvamento dos resultados
# =====================================================================

df_results = pd.DataFrame(results) # type: ignore
df_medias = (
    df_results.groupby("modelo")[
        ["acuracia", "f1_macro", "f1_weighted", "recall_macro", "precision_macro"]
    ]
    .mean()
    .reset_index()
    .round(4)
)

df_medias.to_csv("resultados.csv", index=False)
print(df_medias)
