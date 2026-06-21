# CIFAR-10 Enhanced CNN Classifier

A custom hybrid CNN architecture trained on CIFAR-10 achieving **90.93% test accuracy**.

## Architecture
- Residual Connections (He et al., ResNet CVPR 2016)
- Squeeze-and-Excitation Attention (Hu et al., SENet CVPR 2018)
- Depthwise Separable Convolutions (Howard et al., MobileNets 2017)
- Global Average Pooling head
- MixUp Augmentation + Cosine LR Schedule + Label Smoothing

## Results
| Metric | Value |
|--------|-------|
| Test Accuracy | 90.93% |
| Top-3 Accuracy | 98.71% |
| Macro F1 | 0.91 |
| Best Epoch | 86 / 100 |
| Overfit Gap | 3.4% (healthy) |

## Per-Class Performance
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Airplane | 0.90 | 0.92 | 0.91 |
| Automobile | 0.96 | 0.94 | 0.95 |
| Bird | 0.89 | 0.88 | 0.88 |
| Cat | 0.89 | 0.77 | 0.82 |
| Deer | 0.90 | 0.91 | 0.90 |
| Dog | 0.86 | 0.87 | 0.87 |
| Frog | 0.91 | 0.95 | 0.93 |
| Horse | 0.92 | 0.93 | 0.92 |
| Ship | 0.94 | 0.96 | 0.95 |
| Truck | 0.93 | 0.95 | 0.94 |

## Training Configuration
- **Optimizer:** SGD + Nesterov Momentum (0.9)
- **LR Schedule:** Linear Warmup (5 epochs) + Cosine Annealing
- **Loss:** Categorical Cross-Entropy (label smoothing=0.1)
- **Augmentation:** Random Flip, Crop, Brightness + MixUp (α=0.4)
- **Hardware:** Google Colab T4 GPU
- **Framework:** TensorFlow 2.15 / Keras

## Project Structure
CIFAR10_ENHANCEDCNN/

├── streamlit_app/

│   ├── app.py

│   ├── requirements.txt

│   ├── assets/

│   │   ├── training_curves.png

│   │   ├── confusion_matrix.png

│   │   └── sample_predictions.png

│   └── models/           ← download from releases

├── graphs/

│   ├── training_curves.png

│   ├── confusion_matrix.png

│   └── lr_schedule.png

└── logs/

└── training_log.csv

## Setup & Run Locally
```bash
cd streamlit_app
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python -m streamlit run app.py
```

## Note on Model Weights
Model weights are not included in this repository due to file size.
Download from Google Drive or retrain using the provided architecture.

## References
1. He et al., *Deep Residual Learning*, CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
2. Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
3. Howard et al., *MobileNets*, 2017. [arXiv:1704.04861](https://arxiv.org/abs/1704.04861)
4. Zhang et al., *MixUp*, ICLR 2018. [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
5. Ioffe & Szegedy, *Batch Normalization*, ICML 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)

## Author
**Hussain Samdani**
- GitHub: [Hussain-Innovator](https://github.com/Hussain-Innovator)
- LinkedIn: [hussain56](https://linkedin.com/in/hussain56)