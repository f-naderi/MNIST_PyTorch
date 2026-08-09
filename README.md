```markdown
# MNIST Handwritten Digit Recognition

A deep learning project for recognizing handwritten digits (0-9) using a Convolutional Neural Network (CNN) built with PyTorch.


## Project Structure

Explainable_AI/
├── model.py                  # Model architecture (CNN)
├── main.py                   # Entry point for training and evaluation
├── utils.py                  # Helper functions, data loading
├── train.py                  # Training script
├── evaluate.py               # Evaluation (confusion matrix, ROC curves, PR curves)
├── MNIST-PyTorch.ipynb       # Interactive Jupyter notebook
└── requirements.txt          # Python dependencies


## Installation

```bash
git clone https://github.com/f-naderi/MNIST_PyTorch.git
cd MNIST_PyTorch
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train the model

```bash
python3 main.py
```
Options:

* --epochs: Number of epochs (default: 10)
* --batch_size: Batch size (default: 64)
* --lr: Learning rate (default: 0.001)

### Evaluate 

```bash
python3 main.py --no_train --load_model "model/mnist_model.pth"
```


## Requirements

* Python 3.10+
* PyTorch
* Matplotlib
* NumPy
* Requests
* Scikit-learn
* Seaborn
* Tqdm

## References

- MNIST Dataset: LeCun et al. (1998)

```
