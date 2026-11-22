import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    RUTA_DATASET_COMPLETO,
    RUTA_DATASET_REDUCIDO,
    FIG_DIR,
    RANDOM_STATE,
    K_FOLDS,
    CONFIGURACIONES_MLP
)


class MLP_Numpy:
    def __init__(self, tamanios_capas, activacion="relu", lr=0.01, momentum=0.9, seed=RANDOM_STATE):
        self.tamanios_capas = tamanios_capas
        self.activacion = activacion.lower()
        self.lr = lr
        self.momentum = momentum
        self.seed = seed
        self.inicializar_parametros()

    def inicializar_parametros(self):
        rng = np.random.default_rng(self.seed)
        self.pesos = []
        self.sesgos = []
        self.v_pesos = []
        self.v_sesgos = []

        for i in range(len(self.tamanios_capas) - 1):
            n_in = self.tamanios_capas[i]
            n_out = self.tamanios_capas[i + 1]

            if self.activacion == "relu":
                lim = np.sqrt(2.0 / n_in)
            else:
                lim = np.sqrt(2.0 / (n_in + n_out))

            W = rng.normal(0.0, lim, size=(n_in, n_out))
            b = np.zeros((1, n_out))

            self.pesos.append(W)
            self.sesgos.append(b)
            self.v_pesos.append(np.zeros_like(W))
            self.v_sesgos.append(np.zeros_like(b))

    def _activar(self, Z):
        if self.activacion == "relu":
            return np.maximum(0, Z)
        elif self.activacion == "tanh":
            return np.tanh(Z)
        return Z

    def _derivada_activacion(self, A, Z):
        if self.activacion == "relu":
            return (Z > 0).astype(float)
        elif self.activacion == "tanh":
            return 1.0 - (A ** 2)
        return np.ones_like(A)

    def _softmax(self, Z):
        Z_shift = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shift)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        A = X
        self.activaciones = [A]
        self.zs = []

        num_capas = len(self.pesos)
        for i in range(num_capas):
            Z = A @ self.pesos[i] + self.sesgos[i]
            self.zs.append(Z)

            if i == num_capas - 1:
                A = self._softmax(Z)
            else:
                A = self._activar(Z)
            self.activaciones.append(A)

        return A

    def calcular_loss(self, Y_prob, Y_onehot):
        eps = 1e-15
        Y_prob_clip = np.clip(Y_prob, eps, 1.0 - eps)
        return float(-np.mean(np.sum(Y_onehot * np.log(Y_prob_clip), axis=1)))

    def backward(self, Y_onehot):
        N = Y_onehot.shape[0]
        num_capas = len(self.pesos)
        dZ = self.activaciones[-1] - Y_onehot

        for i in reversed(range(num_capas)):
            A_prev = self.activaciones[i]
            dW = (A_prev.T @ dZ) / N
            db = np.sum(dZ, axis=0, keepdims=True) / N

            if i > 0:
                W = self.pesos[i]
                dA_prev = dZ @ W.T
                Z_prev = self.zs[i - 1]
                dZ = dA_prev * self._derivada_activacion(A_prev, Z_prev)

            self.v_pesos[i] = self.momentum * self.v_pesos[i] + self.lr * dW
            self.v_sesgos[i] = self.momentum * self.v_sesgos[i] + self.lr * db
            self.pesos[i] -= self.v_pesos[i]
            self.sesgos[i] -= self.v_sesgos[i]

    def entrenar(self, X_train, Y_train, X_val, Y_val, epochs=350, batch_size=16):
        N = X_train.shape[0]
        hist_tr = []
        hist_val = []
        rng = np.random.default_rng(self.seed)

        for ep in range(epochs):
            indices = rng.permutation(N)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            for inicio in range(0, N, batch_size):
                fin = inicio + batch_size
                xb = X_shuffled[inicio:fin]
                yb = Y_shuffled[inicio:fin]
                self.forward(xb)
                self.backward(yb)

            loss_tr = self.calcular_loss(self.forward(X_train), Y_train)
            hist_tr.append(loss_tr)

            if X_val is not None:
                loss_val = self.calcular_loss(self.forward(X_val), Y_val)
                hist_val.append(loss_val)

        return hist_tr, hist_val

    def predecir(self, X):
        return np.argmax(self.forward(X), axis=1)


def to_one_hot(y, num_clases=10):
    y = np.array(y, dtype=int)
    N = len(y)
    onehot = np.zeros((N, num_clases), dtype=float)
    onehot[np.arange(N), y] = 1.0
    return onehot


def calcular_metricas_multiclase(y_true, y_pred, num_clases=10):
    accuracy = np.mean(y_true == y_pred)
    cm = np.zeros((num_clases, num_clases), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    f1_list = []
    for c in range(num_clases):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

    return accuracy, np.mean(f1_list), cm


def crear_k_folds_estratificados(y, k=5, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y = np.array(y)
    clases_unicas = np.unique(y)
    folds = [[] for _ in range(k)]

    for c in clases_unicas:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        partes = np.array_split(idx, k)
        for i in range(k):
            folds[i].extend(partes[i])

    return [np.array(f) for f in folds]


def estandarizar_train_val(X_tr, X_val):
    mu = np.mean(X_tr, axis=0)
    sigma = np.std(X_tr, axis=0)
    sigma_segura = np.where(sigma == 0, 1e-12, sigma)
    return (X_tr - mu) / sigma_segura, (X_val - mu) / sigma_segura


def ejecutar_experimentos_kfold(df, nombre_experimento, config):
    X_crudo = df.drop(columns=["Clase"]).to_numpy(dtype=np.float64)
    y_crudo = df["Clase"].to_numpy(dtype=int) - 1

    folds = crear_k_folds_estratificados(y_crudo, k=K_FOLDS, seed=RANDOM_STATE)
    accuracies = []
    f1_scores = []
    perdidas_tr_todas = []
    perdidas_val_todas = []
    cm_total = np.zeros((10, 10), dtype=int)

    tamanios_capas = [X_crudo.shape[1]] + config["capas_ocultas"] + [10]

    for fold_idx in range(K_FOLDS):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(K_FOLDS) if i != fold_idx])

        X_train_fold, X_val_fold = estandarizar_train_val(X_crudo[train_idx], X_crudo[val_idx])
        y_train_fold = y_crudo[train_idx]
        y_val_fold = y_crudo[val_idx]

        mlp = MLP_Numpy(
            tamanios_capas=tamanios_capas,
            activacion=config["activacion"],
            lr=config["lr"],
            seed=RANDOM_STATE + fold_idx
        )

        hist_tr, hist_val = mlp.entrenar(
            X_train_fold, to_one_hot(y_train_fold, 10),
            X_val_fold, to_one_hot(y_val_fold, 10),
            epochs=config["epochs"],
            batch_size=16
        )

        y_pred = mlp.predecir(X_val_fold)
        acc, f1, cm = calcular_metricas_multiclase(y_val_fold, y_pred, 10)

        accuracies.append(acc)
        f1_scores.append(f1)
        cm_total += cm
        perdidas_tr_todas.append(hist_tr)
        perdidas_val_todas.append(hist_val)

    prom_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    prom_f1 = np.mean(f1_scores) * 100

    return {
        "nombre": f"{nombre_experimento}_{config['nombre']}",
        "acc_mean": prom_acc,
        "acc_std": std_acc,
        "f1_mean": prom_f1,
        "cm": cm_total,
        "loss_tr": np.mean(perdidas_tr_todas, axis=0),
        "loss_val": np.mean(perdidas_val_todas, axis=0)
    }


def graficar_comparativa(resultados_fase2, resultados_fase4, mejor_config_nombre):
    plt.figure(figsize=(10, 5))
    plt.plot(resultados_fase2["loss_tr"], label="Fase 2 (29 vars) - Train Loss", color="#1f77b4", linestyle="--")
    plt.plot(resultados_fase2["loss_val"], label="Fase 2 (29 vars) - Val Loss", color="#1f77b4")
    plt.plot(resultados_fase4["loss_tr"], label="Fase 4 (20 vars) - Train Loss", color="#ff7f0e", linestyle="--")
    plt.plot(resultados_fase4["loss_val"], label="Fase 4 (20 vars) - Val Loss", color="#ff7f0e")

    plt.title(f"Curvas de Aprendizaje (Loss) - {mejor_config_nombre}", fontsize=12, fontweight="bold")
    plt.xlabel("Épocas de Entrenamiento")
    plt.ylabel("Pérdida (Entropía Cruzada)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "curvas_loss_fase2_vs_fase4.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(resultados_fase4["cm"], annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=range(1, 11), yticklabels=range(1, 11))
    plt.title(f"Matriz de Confusión Acumulada 5-Fold (Fase 4: {resultados_fase4['acc_mean']:.1f}% Acc)", fontsize=11, fontweight="bold")
    plt.xlabel("Clase Predicha")
    plt.ylabel("Clase Verdadera")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "matriz_confusion_fase4.png", dpi=300)
    plt.close()


def ejecutar_todos_los_experimentos():
    df_completo = pd.read_excel(RUTA_DATASET_COMPLETO)
    if "Longitud" in df_completo.columns:
        df_completo = df_completo.drop(columns=["Longitud"])

    df_reducido = pd.read_excel(RUTA_DATASET_REDUCIDO)

    mejores_fase2 = None
    mejores_fase4 = None
    mejor_acc = -1.0
    mejor_cfg_nombre = ""

    for cfg in CONFIGURACIONES_MLP:
        res_f2 = ejecutar_experimentos_kfold(df_completo, "Fase2", cfg)
        res_f4 = ejecutar_experimentos_kfold(df_reducido, "Fase4", cfg)

        if res_f4["acc_mean"] > mejor_acc:
            mejor_acc = res_f4["acc_mean"]
            mejores_fase2 = res_f2
            mejores_fase4 = res_f4
            mejor_cfg_nombre = cfg["nombre"]

    graficar_comparativa(mejores_fase2, mejores_fase4, mejor_cfg_nombre)
    return mejores_fase2, mejores_fase4


if __name__ == "__main__":
    ejecutar_todos_los_experimentos()
