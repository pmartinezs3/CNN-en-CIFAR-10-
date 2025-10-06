# CNN · CIFAR‑10 · Regularización y Optimización Avanzada

Proyecto de clasificación de **CIFAR‑10** con una **CNN** que incluye **BatchNorm**, **L2**, **Dropout**, 
**data augmentation** y callbacks (**EarlyStopping**, **ReduceLROnPlateau**, **ModelCheckpoint**).
Se registran las curvas de *accuracy/loss*, el **learning rate** por época, la **matriz de confusión** y una grilla de **errores**.


## Requisitos
- Python 3.10+
- TensorFlow 2.14+ (CPU o GPU), NumPy, Matplotlib

```bash
# Crear entorno (Windows)
python -m venv .venv
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Ejecutar
```bash
python Actividad5.py
```
El script descarga automáticamente **CIFAR‑10**, entrena por ~30 épocas y guarda las figuras en `reports/`.

### Figuras incluidas
- `reports/acc.png` — Accuracy train/val.
- `reports/loss.png` — Pérdida train/val.
- `reports/lr.png` — Evolución del learning rate.
- `reports/cm.png` — Matriz de confusión en test.
- `reports/errors.png` — Mosaico de ejemplos mal clasificados.

## Notas
- Si usas GPU, instala además: `pip install tensorflow[and-cuda]` 

## Licencia
MIT © Pamela Martinez
