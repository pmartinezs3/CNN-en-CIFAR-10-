#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Módulo 7 – Sesión 5
CNN con regularización y optimización avanzada (CIFAR‑10 / Fashion‑MNIST)

Incluye
- 2+ capas convolucionales con regularización L2 y Dropout (opcional L1)
- Optimizador seleccionable (Adam, RMSprop, SGD+momentum)
- Callbacks de optimización: EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
- Registro y gráfico del Learning Rate por época
- Curvas de loss/accuracy, matriz de confusión y grilla de errores

"""

import os
os.environ.pop("KERAS_BACKEND", None)  

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks

# Reproducibilidad
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============================
# Parámetros principales
# =============================
DATASET = "cifar10"    
IMG_SIZE = 32
BATCH_SIZE = 128
EPOCHS = 30
VAL_SPLIT = 0.2

# Regularización
L2 = 1e-4               
L1 = 0.0                
DROPOUT1 = 0.25
DROPOUT2 = 0.5

# Optimizador
OPTIMIZER = "adam"      

# =============================
# Utilidades
# =============================
class LrLogger(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.lrs = []
    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.lrs.append(lr)


def plot_history_and_lr(history, lr_logger, title_prefix="CNN"):
    plt.figure()
    plt.title(f"{title_prefix} – Accuracy")
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Época"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()

    plt.figure()
    plt.title(f"{title_prefix} – Pérdida")
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Época"); plt.ylabel("Pérdida"); plt.legend(); plt.tight_layout()

    if lr_logger and lr_logger.lrs:
        plt.figure()
        plt.title(f"{title_prefix} – Learning Rate")
        plt.plot(lr_logger.lrs)
        plt.xlabel("Época"); plt.ylabel("LR"); plt.tight_layout()


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", aspect="equal")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="Etiqueta real", xlabel="Predicción")
    plt.xticks(rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.title("Matriz de confusión")
    plt.tight_layout()


def show_errors_grid(x, y_true, y_pred, class_names, n=16):
    wrong = np.where(y_true != y_pred)[0]
    if wrong.size == 0:
        print("No hay errores para mostrar.")
        return
    idx = wrong[:n]
    cols = 4
    rows = int(np.ceil(len(idx)/cols))
    plt.figure(figsize=(cols*3, rows*3))
    for k, i in enumerate(idx):
        plt.subplot(rows, cols, k+1)
        plt.imshow(x[i].astype("uint8"))
        t = class_names[y_true[i]]
        p = class_names[y_pred[i]]
        plt.title(f"gt: {t}\npred: {p}", color="red", fontsize=9)
        plt.axis("off")
    plt.suptitle("Errores de clasificación")
    plt.tight_layout()


# =============================
# Datos
# =============================

def load_dataset(name="cifar10"):
    name = name.lower()
    if name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        x_train = x_train.astype("float32"); x_test = x_test.astype("float32")
    elif name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        class_names = ["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
        # pasar a 3 canales y 32x32
        x_train = np.repeat(x_train[...,None], 3, axis=-1).astype("float32")
        x_test  = np.repeat(x_test[...,None], 3, axis=-1).astype("float32")
        x_train = tf.image.resize(x_train, (IMG_SIZE, IMG_SIZE)).numpy()
        x_test  = tf.image.resize(x_test,  (IMG_SIZE, IMG_SIZE)).numpy()
    else:
        raise ValueError("DATASET debe ser 'cifar10' o 'fashion_mnist'")

    y_train = y_train.flatten(); y_test = y_test.flatten()

    # split train/val
    n = x_train.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    val_size = int(n*VAL_SPLIT)
    val_idx, train_idx = idx[:val_size], idx[val_size:]

    x_tr, y_tr = x_train[train_idx], y_train[train_idx]
    x_val, y_val = x_train[val_idx], y_train[val_idx]

    # normalización 0-255 -> 0-1
    x_tr /= 255.0; x_val /= 255.0; x_test = x_test/255.0

    # almacenar versión uint8 reescalada para grillas
    x_test_vis = (x_test*255.0).clip(0,255).astype("uint8")

    # augmentations ligeras en train
    def _augment(img, label):
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).map(_augment, num_parallel_calls=tf.data.AUTOTUNE).shuffle(4096, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds   = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(lambda a,b:(tf.image.resize(a,(IMG_SIZE,IMG_SIZE)), b), num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(lambda a,b:(tf.image.resize(a,(IMG_SIZE,IMG_SIZE)), b), num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names, x_test_vis, y_test


# =============================
# Modelo
# =============================

def make_cnn(num_classes):
    reg = regularizers.l1_l2(l1=L1, l2=L2) if (L1>0 or L2>0) else None
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(DROPOUT1)(x)

    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(DROPOUT1)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(DROPOUT2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="cnn_regopt")


def get_optimizer(name="adam", lr=1e-3):
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.9)
    if name == "sgd_mom":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    raise ValueError("OPTIMIZER debe ser 'adam', 'rmsprop' o 'sgd_mom'")


# =============================
# Entrenamiento y evaluación
# =============================

def train_and_eval():
    train_ds, val_ds, test_ds, class_names, x_test_vis, y_test_raw = load_dataset(DATASET)
    num_classes = len(class_names)

    model = make_cnn(num_classes)
    opt = get_optimizer(OPTIMIZER, LR_INIT)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 

    lr_logger = LrLogger()
    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        lr_logger,
        callbacks.ModelCheckpoint("best_cnn.keras", monitor="val_loss", save_best_only=True)
    ]

    print("Entrenando…")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb, verbose=2)

    plot_history_and_lr(history, lr_logger, title_prefix=f"CNN · {DATASET.upper()} · {OPTIMIZER}")

    print("Evaluando en test…")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test → loss: {test_loss:.4f} | acc: {test_acc:.4f}")

    # Predicciones completas y matriz de confusión
    y_true, y_pred = [], []
    for batch_imgs, batch_lbls in test_ds:
        probs = model.predict(batch_imgs, verbose=0)
        y_true.extend(batch_lbls.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
    y_true = np.array(y_true); y_pred = np.array(y_pred)

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()
    plot_confusion_matrix(cm, class_names)

    # Grilla de errores
    show_errors_grid(x_test_vis, y_true, y_pred, class_names, n=16)



    return {"model": model, "history": history, "test_acc": float(test_acc), "cm": cm, "comment": comentario}


if __name__ == "__main__":
    results = train_and_eval()


# In[3]:


comentario = f""" Resultados y discusión — Sesión 5 (CNN regularizada en CIFAR-10, optimizador Adam)

Evolución del entrenamiento
La accuracy de entrenamiento y validación avanza de forma paralela y estable hasta ~0.83 y ~0.82 respectivamente. La brecha entre curvas es pequeña e incluso la validación supera levemente al entrenamiento en varios puntos, lo cual es coherente con el uso de Dropout, L2 y data augmentation: el modelo ve ejemplos “más difíciles” durante el entrenamiento y generaliza bien. La pérdida de validación desciende con oscilaciones naturales, cerrando cerca de 0.62.

Efecto del scheduler de tasa de aprendizaje
ReduceLROnPlateau redujo el learning rate cuando la validación se estancó: de 0.001 a 0.0005 alrededor de la época 17 y luego a 0.00025 cerca de la 24. Tras cada bajada se observa un pequeño salto de calidad en validación, signo de que el ajuste fino ayudó a consolidar la convergencia. El entrenamiento completó 30 épocas; EarlyStopping no se activó porque la validación siguió mejorando de manera intermitente.

Rendimiento en test
Test accuracy = 0.8200, consistente con la validación final, lo que confirma buena generalización con la configuración actual de regularización y optimización.

Matriz de confusión y patrones de error
Las clases con mejor desempeño son ship (899 aciertos), automobile (927), truck (863), frog (860), horse (855) y airplane (850), con diagonales fuertes. Los errores más frecuentes aparecen entre pares visualmente similares o con fondos confusos: cat ↔ dog (cat como dog: 166; dog como cat: 109), bird ↔ airplane (bird como airplane: 55), deer ↔ horse (horse como deer: 43; deer como horse: 23) y airplane ↔ ship (airplane como ship: 40). La grilla de errores muestra que el bajo tamaño 32×32 y el ruido de fondo contribuyen a esas confusiones.

Qué funcionó

La combinación L2 + Dropout contuvo el sobreajuste, manteniendo juntas las curvas train/val.

ReduceLROnPlateau permitió mejorar cuando la métrica se estancó, mostrando ganancias tras cada reducción de LR.

Batch Normalization estabilizó las actualizaciones y facilitó entrenar con tasas de aprendizaje relativamente altas al inicio.

Mejoras sugeridas

Probar SGD con momentum y un plan de LR tipo coseno o One-Cycle; suele mejorar la generalización en CIFAR-10.

Aumentar moderadamente la capacidad (más filtros en el segundo bloque o un bloque adicional) y reforzar el augmentation con recortes aleatorios, traslaciones, Cutout/Mixup.

Ajustar el peso L2 en el rango 1e-4 a 3e-4 y reducir ligeramente Dropout2 (por ejemplo a 0.4) para ver si sube la accuracy sin perder generalización.

Añadir label smoothing ligero (p. ej., 0.1) y monitorizar accuracy por clase; poner atención especial a cat y dog.

Reportar top-2 accuracy; dado que los errores se concentran entre clases cercanas, la métrica top-k ayuda a reflejar mejor la calidad del modelo.
"""


# In[ ]:




