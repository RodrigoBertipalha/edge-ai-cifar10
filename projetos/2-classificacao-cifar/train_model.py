import os
import sys
import io
import ssl

# Forçar treino apenas em CPU (antes de importar tensorflow)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Configurar stdout/stderr para UTF-8 (evita erros de codificação no Windows)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# O download do CIFAR-10 usa HTTPS; em alguns ambientes Windows o certificado local falha.
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf

# Garantir que nenhuma GPU seja visível
tf.config.set_visible_devices([], "GPU")

# ---------------------------------------------------------------------------
# Projeto 2 — Classificação CIFAR-10
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset CIFAR-10 via tf.keras.datasets.cifar10
#   2. Normalizar as imagens para [0, 1] (shape (32, 32, 3))
#   3. Separar um conjunto de validação
#   4. Incluir data augmentation (ex: layers.RandomFlip, RandomRotation, RandomZoom)
#      aplicada ao conjunto de treino
#   5. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   6. Treinar com EarlyStopping monitorando a perda de validação
#   7. Exibir a acurácia de validação final no terminal
#   8. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

SEED = 42
BATCH_SIZE = 128
EPOCHS = 25
VAL_SIZE = 5000
NUM_CLASSES = 10

tf.keras.utils.set_random_seed(SEED)


def load_data():
    (x_train_full, y_train_full), (_, _) = tf.keras.datasets.cifar10.load_data()

    x_train_full = x_train_full.astype("float32") / 255.0
    y_train_full = y_train_full.astype("int32").reshape(-1)

    # Split explícito treino/validação
    x_val = x_train_full[-VAL_SIZE:]
    y_val = y_train_full[-VAL_SIZE:]
    x_train = x_train_full[:-VAL_SIZE]
    y_train = y_train_full[:-VAL_SIZE]

    return (x_train, y_train), (x_val, y_val)


def build_model():
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.1, seed=SEED),
            tf.keras.layers.RandomZoom(0.1, seed=SEED),
        ],
        name="data_augmentation",
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(32, 32, 3)),
            data_augmentation,

            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="cifar10_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    (x_train, y_train), (x_val, y_val) = load_data()
    print(f"Treino: {x_train.shape[0]} amostras | Validação: {x_val.shape[0]} amostras")

    model = build_model()
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1,
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        verbose=2,
    )

    # Avaliar o modelo já com os melhores pesos restaurados
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f"\nAcurácia final de validação: {val_accuracy:.4f}")
    print(f"Perda final de validação:    {val_loss:.4f}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.h5")
    model.save(model_path, save_format="h5")
    print(f"Modelo salvo em: {model_path}")


if __name__ == "__main__":
    main()
