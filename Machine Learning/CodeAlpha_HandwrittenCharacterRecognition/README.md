# Handwritten Character Recognition

**CodeAlpha Machine Learning Internship - Task 3**
Author: Farhan Shaikh

Recognise handwritten **digits and letters** with a Convolutional Neural Network (CNN), then use the trained
model on your own photos - and, as a bonus, on whole words.

## Datasets

| `--dataset` | Content | Classes | Images (train / test) |
|---|---|---|---|
| `mnist` (default) | handwritten digits | 10 | 60,000 / 10,000 |
| `emnist-digits` | EMNIST digits | 10 | 240,000 / 40,000 |
| `emnist-letters` | letters A-Z (upper/lower case merged) | 26 | 124,800 / 20,800 |
| `emnist-balanced` | digits + letters | 47 | 112,800 / 18,800 |
| `emnist-byclass` | digits + upper + lower case | 62 | 697,932 / 116,323 |
| `custom` | your own `.npz` (`x_train, y_train, x_test, y_test`) | any | any |

MNIST downloads automatically (about 11 MB). The EMNIST options use the small `emnist` package, which downloads the
EMNIST archive once (about 535 MB) and caches it; it also fixes the transposed orientation of the raw EMNIST images.
`digits-demo` (scikit-learn's tiny 8x8 digits) exists only as an offline smoke test of the code.

## Approach

```
28x28 image -> rescale 0-1 -> random rotate / shift / zoom (training only)
  -> 2 x [Conv 32] -> MaxPool -> Dropout 0.25
  -> 2 x [Conv 64] -> MaxPool -> Dropout 0.25
  -> Flatten -> Dense 256 -> Dropout 0.4 -> softmax over the characters
```

- Every convolution is followed by **Batch Normalisation** and ReLU (about 0.87 million parameters for 10 classes).
- **Data augmentation** (small rotations, shifts and zooms) shows the network each character in many slightly different
  forms, which reduces over-fitting; it switches itself off at prediction time.
- Training uses Adam, **early stopping** on validation accuracy and a **learning-rate schedule**
  (the rate is halved when validation loss stalls).
- A 10 % validation split is held out of the training data; the official test set is touched only once, at the end.
- The model takes raw 0-255 pixels and rescales them internally, so training and prediction cannot disagree about scaling.

Evaluation: test accuracy, per-class precision / recall / F1, confusion matrix, a gallery of the most confident
mistakes and training curves.

## Results

Verified in my sandbox on **a 5,000-image sample of real MNIST** (3,600 train / 400 validation / 1,000 test images,
15 epochs on a CPU): **97.8 % test accuracy**, every digit between 95 % and 100 % recall
(`outputs/mnist_5k_sample/`). That is a small, quick check - training on the full 60,000 MNIST images
(`python train.py --dataset mnist`) gives more data per class and typically lands around 99 %; your run prints
the exact figure at the end of training and writes the plots to `outputs/mnist/`.
EMNIST letters are harder because many characters look alike (O / 0, I / l / 1, S / 5, ...), so expect a lower accuracy there.

## Run it

```bash
pip install -r requirements.txt

python train.py --dataset mnist                    # digits
python train.py --dataset emnist-letters           # letters A-Z (downloads EMNIST once)
python train.py --dataset emnist-balanced --epochs 30
python train.py --dataset mnist --limit 10000      # quick experiment on 10,000 training images
```

On a CPU, full MNIST takes a few minutes per epoch; a free GPU (for example Google Colab) is much faster for EMNIST.
Training writes `models/<dataset>_cnn.keras` and `outputs/<dataset>/` (classification report, confusion matrix,
training curves, misclassified examples, sample predictions).

### Recognise your own handwriting

```bash
python predict.py my_digit.png                                   # model trained on MNIST
python predict.py letter.png --dataset emnist-letters --top 5
python predict.py photo.jpg --save-processed what_the_model_sees.png
```

Write **one** character, big and thick (marker, or a thick brush in Paint), dark ink on plain paper. `preprocess.py` turns
the picture into the MNIST format automatically: it flips dark-on-light to white-on-black, crops to the ink, scales it
to fit a 20x20 box (a slightly larger one for EMNIST), and centres it on a 28x28 canvas. Use `--save-processed` to see exactly what the
network receives - handy for explaining why a photo is misread.

### Bonus: words and numbers

```bash
python recognize_word.py number.png --dataset mnist
python recognize_word.py word.png --dataset emnist-letters --save result.png
```

`recognize_word.py` cuts a line of writing into characters wherever there is a vertical gap, classifies each character
with the CNN, and inserts a space where a gap is much wider than the usual letter spacing. In my tests on real MNIST digits
glued into a line it read "31415" and "27 08" correctly (`outputs/mnist_5k_sample/test_number.png`).

## Extending to words and sentences (CRNN)

Cutting at gaps fails for joined-up (cursive) handwriting, where letters touch. The standard solution is a **CRNN**:

1. a CNN (like this one, without the final Flatten) turns the word image into a sequence of feature columns,
2. a bidirectional LSTM reads that sequence left to right,
3. **CTC loss** learns the alignment between image columns and letters, so no manual character segmentation is needed.

It is trained on a word-level dataset such as IAM Handwriting. The CNN in `model.py` is the natural starting point for step 1.

## Project structure

```
data_loader.py     MNIST / EMNIST / custom dataset loading
model.py           the CNN
train.py           train, evaluate, save plots and model
preprocess.py      photo -> 28x28 MNIST-style image (no TensorFlow needed)
predict.py         recognise one character from an image file
recognize_word.py  bonus: words / numbers by segmentation
outputs/           sample results from my 5,000-image MNIST check
```
