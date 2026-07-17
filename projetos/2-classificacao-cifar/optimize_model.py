import os
import sys
import io

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Configurar stdout/stderr para UTF-8 (evita erros de codificação no Windows)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import tensorflow as tf

# ---------------------------------------------------------------------------
# Projeto 2 — Otimização do Modelo (CIFAR-10)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------


def format_size(num_bytes):
    kb = num_bytes / 1024.0
    if kb < 1024:
        return f"{kb:.2f} KB"
    return f"{kb / 1024:.2f} MB"


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    h5_path = os.path.join(script_dir, "model.h5")
    tflite_path = os.path.join(script_dir, "model.tflite")

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"model.h5 não encontrado em {h5_path}. "
            "Rode train_model.py antes de otimizar."
        )

    print(f"Carregando modelo Keras de: {h5_path}")
    model = tf.keras.models.load_model(h5_path)
    print("Modelo carregado com sucesso.")

    print("\nConvertendo modelo para TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Dynamic Range Quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    print("Conversão concluída.")

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    h5_size = os.path.getsize(h5_path)
    tflite_size = os.path.getsize(tflite_path)
    reduction = (1 - tflite_size / h5_size) * 100 if h5_size > 0 else 0.0

    print(f"\nModelo TFLite salvo em: {tflite_path}")
    print(f"Tamanho do model.h5:     {format_size(h5_size)}")
    print(f"Tamanho do model.tflite: {format_size(tflite_size)}")
    print(f"Redução de tamanho:      {reduction:.2f}%")
    print("\nOtimização concluída com sucesso.")


if __name__ == "__main__":
    main()
