# D7047E Lab 3 — Image Captioning

End-to-end image captioning on Flickr8k: VGG16 features + LSTM + concatenation fusion.

## Structure

```
.
├── task1_image_features.ipynb     # VGG16 fc2 feature extraction → features.pkl
├── Task2_RNN.ipynb                # Vocabulary builder → vocab.pkl
├── Task3_Model_Training.ipynb     # model training → model.pt
├── data/
│   ├── features.pkl               # 8091 images × 4096-dim (VGG16 fc2)
│   └── vocab.pkl                  # 2973 words
└── outputs/
    ├── model.pt                   # trained checkpoint (best val_loss)
    ├── loss_curve.png
    └── training_log.csv
```

## How to run

1. Run `task1_image_features.ipynb` (needs `kaggle.json` to download Flickr8k).
2. Run `Task2_RNN.ipynb` to build `vocab.pkl`.
3. Run `Task3_Model_Training.ipynb` to train (PyTorch, MPS/CUDA/CPU auto-detect).

## Results

- Architecture: VGG16 fc2 (4096) + LSTM(256) + concatenation fusion
- Best val_loss: 3.24 (epoch 2, early-stopped at epoch 5)
- Best val accuracy: 37%
- Trainable params: 3.2 M
