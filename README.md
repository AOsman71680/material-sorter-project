# AI Material Sorter

A laptop runs an AI model that classifies an object into one of four material
classes. An STM32 microcontroller will later drive the physical sorting
mechanism.

**Current status: Tasks 1-3.** The repository contains the validated
configuration system, webcam image collection, raw-dataset validation, the
group-aware train/validation/test split with manifests, and image preprocessing
with training-only augmentation. Model training, evaluation, prediction, STM32
serial communication, and motor control are not implemented yet.

## Material classes and cups

| Class     | Default cup | Notes                                                          |
|-----------|-------------|----------------------------------------------------------------|
| `metal`   | Cup 1       |                                                                |
| `plastic` | Cup 2       |                                                                |
| `paper`   | Cup 3       |                                                                |
| `other`   | Cup 4       | Glass, wood, rubber, fabric, ceramics, and other materials     |

Multi-material objects will be labeled by a consistent predominant-material
policy (to be defined with the dataset).

`uncertain` is an internal runtime state used when a prediction's confidence is
below `inference.confidence_threshold`. It is **not** a fifth class and has no
cup. The config loader rejects it as a class name.

## Setup (Windows 10/11, Python 3.11+)

Run these in PowerShell from the project folder:

```powershell
python --version                      # should print 3.11 or newer
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you set the project up before Task 2, run the last command again to install
OpenCV. Use the regular `opencv-python` package (not `opencv-python-headless`),
because the preview window needs OpenCV's GUI support.

`requirements.txt` includes PyTorch and torchvision for training (Task 4). The
default `pip install` on Windows installs the CPU build, which is enough for this
project. If you have an NVIDIA GPU and want CUDA training, install the matching
CUDA build from the selector at https://pytorch.org/get-started/locally/ before
running `pip install -r requirements.txt` (the selector gives the exact command
for your CUDA version). Check what you have with:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`True` means CUDA is available. `training.device: auto` uses it automatically.

If PowerShell blocks the activation script, run this once and retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

In Command Prompt, activate with `.venv\Scripts\activate.bat` instead.

## Run the tests

```powershell
python -m unittest discover -s tests -v
```

The tests use only `unittest`, so no extra install is needed. You may see
OpenCV messages such as `PNG input buffer is incomplete` while they run; those
come from tests that deliberately create corrupt images. The webcam tests use
fake cameras and windows, so they do not need a camera. If you prefer pytest, `python -m pip install pytest` then
`python -m pytest` runs the same tests.

## Collect images (webcam)

```powershell
python -m src.data_collection
python -m src.data_collection --camera-index 1     # use another camera
python -m src.data_collection --config config\config.yaml
```

The camera index comes from `camera.index` in the config; `--camera-index`
overrides it for one run. Images are saved to
`<paths.dataset_dir>\<class>\` (default `data\dataset\metal\` etc.) as
`<class>_<YYYYmmdd_HHMMSS_microseconds>.jpg`. Existing files are never
overwritten.

| Key              | Action                                                  |
|------------------|---------------------------------------------------------|
| `1`              | Select `metal` (classes are numbered in config order)   |
| `2`              | Select `plastic`                                        |
| `3`              | Select `paper`                                          |
| `4`              | Select `other`                                          |
| `Space` or `C`   | Capture and save the current frame to the selected class |
| `Q` or `Esc`     | Quit (closing the window also quits)                    |

Behavior:

- No class is selected at start, and Space does nothing until you pick one, so
  an image can't be saved under the wrong label by accident.
- The window shows the selected class and the image count for each class. The
  text is drawn on the preview only; saved images are the raw camera frame.
- If the camera can't be opened, or stops delivering valid frames for 3
  seconds, the program prints an error and exits. Frames that fail to read are
  never saved.

Labeling tips: keep one consistent rule for multi-material objects (label by
the predominant material), vary angle, lighting and background, and keep the
class counts reasonably balanced.

## Validate the dataset

```powershell
python -m src.dataset
python -m src.dataset --max-imbalance 3 --min-images 100
```

This is read-only: it never modifies, moves, or deletes any file. The exit code
is `0` when there are no errors (warnings are allowed), `1` when the dataset has
errors, and `2` for a bad config or arguments.

| Finding                                              | Severity |
|------------------------------------------------------|----------|
| Dataset folder or a class folder is missing          | Error    |
| A class has no valid images                          | Error    |
| Corrupted, unreadable, or 0-byte image               | Error    |
| Identical image content in two different classes     | Error    |
| Identical image content twice in the same class      | Warning  |
| Largest/smallest class ratio above the limit (default 2x) | Warning |
| Class has fewer than the recommended images (default 50) | Warning |
| Non-image files, sub-folders, or folders that aren't configured classes | Warning |

Supported image types are `.jpg`, `.jpeg`, `.png` and `.bmp`. An image counts as
valid if OpenCV can decode it. Duplicates are detected by exact file content
(SHA-256), so re-saved or resized copies are not caught. Invalid images are
reported but never removed; fix or delete them yourself, then re-run.

Example output:

```
Class     Valid  Invalid   Share  Status
metal        60        0   30.0%  ok
plastic      25        1   12.5%  has invalid images
...
ERRORS (2):
  - invalid image plastic/broken.jpg: cannot be decoded as an image (corrupted or not an image)
  - identical image content in different classes (label conflict): metal/img_001.jpg, other/mislabeled.jpg
WARNINGS (2):
  - class imbalance: 'metal' has 60 images vs 'plastic' with 25 (2.4x, limit 2x)
  ...
Result: INVALID (fix the errors above; no files were changed)
```

## Prepare the dataset (train / validation / test split)

### Raw dataset layout

The raw images live in `paths.dataset_dir` (default `data\dataset`), with one
folder per class. The folder names must match the classes exactly:

```
data\dataset\metal\      *.jpg  *.jpeg  *.png  *.bmp
data\dataset\plastic\
data\dataset\paper\
data\dataset\other\
```

Splitting only reads these files. It never moves, renames, deletes, or changes
them.

### Run it

```powershell
python -m src.prepare_dataset                  # validate, split, write manifest and report
python -m src.prepare_dataset --dry-run        # print the split report, write nothing
python -m src.prepare_dataset --overwrite      # replace earlier outputs (old files are kept as .bak)
python -m src.prepare_dataset --config config\config.yaml
```

Exit codes: `0` success (warnings allowed), `1` invalid dataset, impossible split,
or existing outputs without `--overwrite`, `2` bad config or arguments.

Before any split is made, the dataset is validated with the same checks as
`python -m src.dataset`. Any error (a missing or empty class, a corrupted or
undecodable image, the same image in two classes) stops the run with exit code
`1`. Invalid images are never dropped silently. Fix or remove them, then run
again.

### Split ratios and seed

Set these in `config\config.yaml`:

```yaml
split:
  train: 0.70     # defaults; must sum to 1
  val: 0.15
  test: 0.15
random_seed: 42
```

The same images, settings, and grouping file always give the same split and
byte-identical output files. A different seed gives a different split. Each
class is split separately (stratified), so the class proportions stay close to
the ratios.

### Why group-aware splitting prevents data leakage

Suppose 30 photos show the same plastic bottle. If the photos were split at
random, some would go to training and some to testing. The model would then
see almost the same bottle in both sets, so the test accuracy would look high
even though the model has not learned to recognise new bottles. This is data
leakage, and it inflates the reported accuracy.

To avoid it, all photos of one physical object must be in the same split.
Splitting identifies these groups in two ways:

- **Object groups** come from the grouping metadata file (below).
- **Identical files** are always kept in one group. Byte-identical copies are
  detected automatically. Re-saved or resized photos of the same object are not
  detected, which is why the metadata file is needed.

### Grouping metadata file

The file is set by `dataset.group_metadata`. The default is
`data\group_metadata.csv`, which is outside the dataset folder. It is a CSV with
one row per image:

```csv
path,object_id,session_id
metal/metal_20261024_103501_123456.jpg,bottle-03,2026-10-24-a
metal/metal_20261024_103512_654321.jpg,bottle-03,2026-10-24-a
plastic/plastic_20261024_104010_000001.jpg,cup-07,2026-10-24-b
```

- `path`: the image path relative to the dataset folder. Forward slashes and
  Windows backslashes are both accepted.
- `object_id` (required): every photo of one physical object gets the same ID.
  The ID may not contain `|`, and the same ID may not be used for two classes.
- `session_id` (optional): recorded in the manifest for later analysis. It does
  not affect grouping.

**Limitation:** the webcam collector (`python -m src.data_collection`) saves
only the class and a timestamp in each file name. It does not record object or
session IDs. Create this CSV yourself, for example in Excel, while you label
the photos. Until you do, the split still works, but leakage prevention is off,
and the report says so.

### Outputs

| File | Contents |
|------|----------|
| `data\splits\manifest.csv` | One row per image: `relative_path`, `material_class`, `split`, `group_id`, `object_id`, `session_id` |
| `data\splits\manifest.json` | Random seed, ratios, classes, grouping status, a hash of the grouping file, image counts, and warnings. No timestamps and no absolute paths. |
| `data\splits\split_report.txt` | The full validation report, split table, and warnings |

Paths in the manifest are relative to `paths.dataset_dir` and use forward
slashes, so the same manifest works on any computer. Training and evaluation
scripts should read the manifest instead of splitting again, so every script
uses the same images in each split.

Existing outputs are never replaced silently. If they exist, the script stops
with exit code `1`. With `--overwrite`, each old file is renamed to
`<name>.<timestamp>.bak` first.

### Reading the warnings

| Message | Meaning and action |
|---------|--------------------|
| `no grouping metadata file found` | Leakage prevention is **off**. Test accuracy may be optimistic. Create the grouping CSV. |
| `N image(s) have no row in the grouping metadata` | Those images are treated as separate objects. Add them to the CSV. |
| `class 'x' has only N image(s) in the 'val' split` | A split is small (below 10 images). Collect more photos. |
| `class 'x': 'train' holds 85% ... because whole groups must stay together` | One object has many photos and dominates a split. Add more distinct objects. |
| `class 'x' has one object ... so it was split by image` | The class has fewer than 3 objects, so it was split image by image. Near-copies of one object are in train, val and test, so accuracy for that class is optimistic. Add more objects for a real test. |
| `class 'x' has only N separate group(s)` (error) | Even splitting by image is impossible: the class has fewer than 3 distinct images. |
| `conflicting labels in one group` (error) | One `object_id` (or identical files) appears in two classes. Fix the CSV or the folder. |

## Preprocessing and augmentation

Implemented in `src\preprocessing.py`. Each image goes through these steps:

1. Load as RGB. OpenCV reads BGR, so the channels are converted. File names
   with non-ASCII characters work on Windows.
2. Resize the shorter side to `image.width` x `image.height` (224 x 224 by
   default), keeping the aspect ratio, then take the centre crop. Keep objects
   near the centre of the frame, because edges can be cropped.
3. Scale pixels to 0 to 1, then apply the ImageNet mean and standard deviation
   used by the pretrained models.
4. Return a `float32` array in channel-first (CHW) order, which PyTorch expects.

**Validation, test, and inference images are not augmented.** They use
`preprocess_eval`, which is deterministic.

**Training images** use `preprocess_training`, which adds random rotation,
shift, zoom, and brightness/contrast changes before normalisation. The strength
is set in the `augmentation` section of `config\config.yaml`. Each setting is a
range, and the defaults are small on purpose:

```yaml
augmentation:
  max_rotation_degrees: 10.0   # rotation in [-10, +10] degrees
  max_translation: 0.05        # shift up to 5% of the image size
  min_scale: 0.9               # zoom between 0.9x and 1.1x
  max_scale: 1.1
  brightness: 0.15             # brightness changed by up to 15%
  contrast: 0.15               # contrast changed by up to 15%
```

The augmentation does not change hue or swap colour channels, so the material
cues (colour, gloss, texture) are kept. The random generator is seeded, so an
augmented image can be reproduced.

Model training itself is not implemented yet (Task 4). The training code must
call `preprocess_training` for the train split and `preprocess_eval` for the
validation, test, and inference inputs, so all three use the same pipeline.

## Train the model (transfer learning)

Training uses the manifest from `python -m src.prepare_dataset`. It never splits
the dataset itself. It reads only the `train` and `val` rows, and the `test` rows
are not loaded. Final test evaluation is a separate step.

The default model is **MobileNetV3 Small** with ImageNet-pretrained weights. Its
final layer is replaced with a new 4-output layer for `metal`, `plastic`,
`paper`, and `other`. Pretrained weights are downloaded once (about 10 MB) and
cached by torch. Use `--no-pretrained` to train from random weights.

### Run training

```powershell
python -m src.train                              # default settings, outputs to paths.model_dir (models\)
python -m src.train --epochs 5 --device cpu      # quick test on the CPU
python -m src.train --no-pretrained              # random initial weights (no download)
python -m src.train --output-dir models\run2     # keep this run separate
python -m src.train --output-dir models\run2 --overwrite   # replace an earlier run (old files kept as .bak)
```

Exit codes: `0` success, `1` error (missing manifest, data problem, CUDA not
available, download failed, or outputs already exist without `--overwrite`),
`2` bad config or arguments.

Progress is printed once per epoch:

```
training on cpu | train 1192 images, val 285 images | mobilenet_v3_small, pretrained=True
epoch   1/30  train loss 0.9120 acc 0.640  |  val loss 0.7344 acc 0.715  (best)  58.2s
epoch   2/30  train loss 0.6011 acc 0.772  |  val loss 0.6890 acc 0.740  (best)  57.9s
...
early stopping: no validation-loss improvement for 5 epoch(s)
best epoch 14: val loss 0.5012, val accuracy 0.836
```

(The numbers above are illustrative, not from a real run.)

### How training works

- **Augmentation** is applied to training images only, using the Task 3 settings
  in `augmentation`. Validation images use the deterministic `preprocess_eval`.
- **Validation** runs after every epoch. Each epoch adds one row to
  `training_history.csv`.
- **Best checkpoint:** `best_model.pth` is the epoch with the lowest validation
  *loss*. Loss is used rather than accuracy because it is smoother on small
  datasets. Each new best replaces the file.
- **Early stopping:** training stops once the validation loss has not improved for
  `training.early_stopping_patience` epochs. `last_model.pth` is saved from the
  final state, which is the state at the stopping point.
- **Reproducibility:** `random_seed` seeds Python, NumPy, and torch, and makes
  the shuffling and augmentation repeatable. On the same machine and settings,
  two runs produce the same history. On CUDA, results can still differ slightly.

### Outputs

| File | Contents |
|------|----------|
| `best_model.pth` | Checkpoint with the lowest validation loss. Holds the weights, architecture, class order, input size, normalisation, epoch, and validation metrics. |
| `last_model.pth` | Checkpoint from the final epoch that ran. |
| `training_history.csv` | `epoch, train_loss, train_accuracy, val_loss, val_accuracy, learning_rate, seconds, is_best` |
| `training_summary.json` | Settings used, device, torch version, manifest counts, best epoch and metrics, whether it stopped early, and `test_split_used: false` |

Checkpoints are saved as dictionaries with the class order recorded. Loading one
with a different class order is rejected, so labels cannot be mixed up. Load
them with `src.model.load_checkpoint`. The inference code is not part of Task 4.

### Notes on the first run

- The first run with `pretrained: true` downloads the ImageNet weights. If the
  download fails, you get a clear message that suggests `--no-pretrained`.
- Speed depends on your hardware. In the test environment (2 CPU cores, no GPU),
  one epoch over 336 training and 80 validation images at 224 px took about 7 s.
  Use `--epochs` for a short first run to measure your own time per epoch.
- A training loss that becomes `NaN` stops the run with an error. Lower
  `training.learning_rate` and try again.

## Evaluate on the test split (Task 5)

This step measures the trained model on the **test split**, the images the model
never saw during training or validation. It loads a checkpoint, runs it over every
test image once, and writes metrics and per-image predictions. It never trains,
fine-tunes, or changes the model, and it never creates a new split.

You need a checkpoint first. Create one with Task 4 (`python -m src.train`). That
writes `models\best_model.pth`. If the file is missing, the evaluation stops with
an error and does not fall back to another file.

### Run the evaluation

```powershell
python -m src.evaluate --checkpoint models\best_model.pth
python -m src.evaluate --checkpoint models\best_model.pth --output-dir models\eval_run2
python -m src.evaluate --checkpoint models\best_model.pth --device cpu
python -m src.evaluate --checkpoint models\best_model.pth --overwrite
```

Options:

| Option | Meaning |
|--------|---------|
| `--checkpoint` | Checkpoint to evaluate. Default: `models\best_model.pth`, and the script prints which file it used. |
| `--manifest` | Manifest CSV. Default: `dataset.manifest` from the config. |
| `--output-dir` | Results folder. Default: `models\evaluation`. |
| `--device` | `auto`, `cpu`, or `cuda`. Default: `training.device` from the config. |
| `--overwrite` | Replace existing results. The old files are renamed to `<name>.<timestamp>.bak`. Without this option, existing results are never replaced. |
| `--config` | Config file. Default: `config\config.yaml`. |

The evaluation reads the class order, architecture, image size, and normalisation
from the checkpoint, and checks them against the config. If they differ, the
script stops with an error. It also checks the manifest. It stops if the test
split is empty, a test image is missing or unreadable, or a test group also
appears in train or val. No test image is skipped silently.

Exit codes: `0` success, `1` evaluation error (missing or incompatible checkpoint,
invalid test data, or existing results without `--overwrite`), `2` bad config or
arguments. The `--device` option only accepts `auto`, `cpu`, or `cuda`.

A run prints a short summary:

```
Test split evaluation
  checkpoint     models/best_model.pth
  test images    64 (correct 16, incorrect 48)
  accuracy       0.2500
  macro F1       0.1000   weighted F1 0.1000
  device         cpu
```

The numbers above come from a small synthetic check, not a real model.

### What each output file contains

| File | Contents |
|------|----------|
| `evaluation_summary.json` | Checkpoint file name and SHA-256 hash, architecture, class order, test image count, correct and incorrect counts, accuracy, macro and weighted averages, per-class metrics, missing classes, device, timestamp, and notes. |
| `classification_report.csv` | One row per class with precision, recall, F1, and support. Then macro avg, weighted avg, and accuracy rows. |
| `confusion_matrix.csv` | Rows are the **actual** class, columns are the **predicted** class, in the order metal, plastic, paper, other. |
| `predictions.csv` | One row per test image: relative path, actual class, predicted class, `correct`, and the softmax probability for each class (`prob_metal`, and so on). |
| `confusion_matrix.png` | The same confusion matrix as a labelled plot with counts in each cell. Rows are actual, columns are predicted. |

The probabilities are the softmax of the model's outputs over the four classes,
so each row sums to 1. They show how confident the model was about each class.
They are not calibrated, so do not read them as reliable confidence. Task 5 does
not use a confidence threshold.

### How to read the metrics

- **Confusion matrix:** the diagonal holds correct predictions. Each off-diagonal
  cell counts one kind of mistake. For example, a row `paper` with a value in the
  `metal` column means paper was mistaken for metal. Read along a row to see where
  one actual class goes. Read down a column to see what was predicted as one class.
- **Precision:** of the images predicted as a class, the fraction that really
  belong to it. Low precision means false alarms for that class.
- **Recall:** of the images that really belong to a class, the fraction the model
  found. Low recall means that material is often missed.
- **F1:** the harmonic mean of precision and recall. It is low when either one is
  low.
- **Support:** the number of true test images of each class. Per-class numbers
  from small support are unstable.
- **Macro average:** the unweighted mean over the four classes. A weak class pulls
  it down equally.
- **Weighted average:** the mean weighted by support. It follows the class mix of
  the test set.
- **Zero denominators:** when a class is never predicted, its precision is 0.0. When
  a class has no test images, its recall and F1 are 0.0. Those classes are then
  not measured at all, and the notes in the summary say so. Macro averages
  include these zeros, so a missing class lowers them.

### Recognise an incomplete or unrepresentative test set

Check these before you believe any number:

- **A class has no test images.** The summary lists it in `missing_classes_in_test`,
  and the notes say the model has not been evaluated on that class.
- **Few images per class.** The notes warn below 30 images per class, and the
  per-class figures then move a lot with one image.
- **The model never predicts a class.** Its precision and F1 are 0.0 by convention.
- **Repeated photos of the same object.** Grouping prevents that object from being
  split across sets, but a test set made of a few objects still tests only those
  objects.
- **One lighting setup, background, or camera session.** The test set then does
  not show how the model behaves elsewhere.
- **Strong class imbalance.** Overall accuracy can look good by being right on the
  largest class.

### Do not use the test results to tune the model

The test split is only meaningful as a final, one-time check. Changing the model
(a different architecture, epoch, learning rate, augmentation, or threshold)
because of test results makes the test set part of the training process. The
score then over-states real performance. Make all decisions with the validation
split. Use the test split once, when the model is final. If you do re-evaluate
later after a change, report that the test set was reused.

### Important limitation

High test accuracy on a small or unrepresentative dataset does not guarantee
reliable real-world material sorting. Real conditions include lighting, camera
position, background, object shape, dirt, and the mix of materials on the
platform, and many of those may not appear in this dataset. Treat these numbers as
evidence about this test set only. They say nothing about performance on other
objects, sessions, or hardware.

## Predict one image (Task 6)

This step predicts the material of a single image with a trained checkpoint. It
prints the predicted class and the probability of each of the four classes. It is
**inference only**: it does not train, change the checkpoint, or control any
hardware. Sorting decisions are not part of this step.

### Installation

No extra packages are needed. The command uses the packages already in
`requirements.txt` (PyTorch, torchvision, OpenCV, and PyYAML). Install them as
described in the setup section. You also need a checkpoint, such as
`models\best_model.pth` from Task 4 (`python -m src.train`).

### Predict one image (PowerShell)

```powershell
python -m src.predict --image path\to\object.jpg
python -m src.predict --image path\to\object.jpg --checkpoint models\best_model.pth
python -m src.predict --image "C:\Photos\my object 01.jpg" --checkpoint models\best_model.pth
python -m src.predict --image path\to\object.jpg --device cpu
python -m src.predict --image path\to\object.jpg --json
python -m src.predict --image path\to\object.jpg --threshold 0.8
```

Put the path in quotes when it contains spaces. File names with non-ASCII
characters work.

| Option | Meaning |
|--------|---------|
| `--image` | Image file to classify (required). JPG, PNG, BMP, and any other format OpenCV can read. |
| `--checkpoint` | Checkpoint to use. Default: `models\best_model.pth`. The script prints the file it used. |
| `--device` | `auto`, `cpu`, or `cuda`. Default: `training.device` from the config. |
| `--json` | Print JSON instead of text, for scripts. Errors are also printed as JSON with an `error` field. |
| `--threshold` | Optional, between 0 and 1. Adds a `low_confidence` flag. See below. |
| `--config` | Config file. Default: `config\config.yaml`. |

### Example output

```
Image:            object.jpg
Predicted:        plastic
Max probability:  0.812345
Probabilities:
  metal    0.051012
  plastic  0.812345  <- predicted
  paper    0.090143
  other    0.046500
Model:            mobilenet_v3_small
Checkpoint:       models/best_model.pth
Device:           cpu

Probabilities are softmax outputs of the model. They are not calibrated and are not guaranteed real-world probabilities.
```

(The numbers above are illustrative.)

### What the output means

- **Predicted:** the class with the highest probability. It is always one of
  `metal`, `plastic`, `paper`, or `other`.
- **Max probability:** that highest probability.
- **Probabilities:** one value per class, computed with softmax over the model's
  four raw output scores (logits). They are between 0 and 1 and add up to 1.
  The class order comes from the checkpoint, and the program checks it against
  the config.
- **Checkpoint and architecture:** which model produced the result. The JSON
  output also includes the SHA-256 hash of the checkpoint file, so a result can
  be tied to one exact file.

### Optional threshold

With `--threshold 0.8`, the output adds `low_confidence: true` when the highest
probability is below 0.8, and `false` otherwise. This is a **display flag only**.
A low-confidence result is still the same four classes. There is no fifth
"unknown" class, and no decision is made from the flag. Task 6 does not include
accept or reject logic. Choosing a threshold for real sorting needs separate
evidence and is left for a later safety task.

### Common errors

| Message | Cause and fix |
|---------|---------------|
| `image not found` | Check the path. Quote it if it has spaces. |
| `... is not a readable image file` | The file is corrupt, empty, or not an image. Try opening it in another program. |
| `checkpoint not found ... No other checkpoint is used in its place` | The file in `--checkpoint` (or `models\best_model.pth`) does not exist. Create it with `python -m src.train`, or give the correct path. |
| `cannot read checkpoint` | The file is damaged or is not a training checkpoint. Retrain or use a different file. |
| `checkpoint architecture ... differs from model.architecture` | The checkpoint was trained with another architecture. Restore the matching config, or use a checkpoint trained with the current one. |
| `checkpoint was trained at 64x64 but the config image size is 32x32` | Same idea: the input size changed after training. Restore the config used for training. |
| `checkpoint classes ... do not match` | The class order in the checkpoint differs from `classes`. Do not use it; the labels would be wrong. |
| `CUDA was requested but is not available` | No NVIDIA GPU or no CUDA build of PyTorch. Use `--device cpu`. |
| `threshold must be strictly between 0 and 1` | Give a value such as `0.8`. |

Exit codes: `0` success, `1` prediction error (image or checkpoint), `2` bad
config or arguments.

### Limitations of softmax confidence

- **Not calibrated.** A probability of 0.9 does not mean the prediction is right
  9 times out of 10. Check calibration on the test split before quoting numbers.
- **Only the four trained classes exist.** The model has no "unknown" output. An
  object from a material it has never seen is still assigned one of the four
  classes, often with high confidence.
- **Not a proof.** A single prediction is one model's opinion about one photo.
  Lighting, background, and camera position can change the result.
- **A threshold does not catch everything.** Low confidence helps with some
  mistakes, but a wrong prediction can still be made with high confidence.
- **Performance depends on the training data.** The results are only as good as
  the dataset the model was trained on. Test-split results (Task 5) describe that
  test set, not every real object.

## Live webcam prediction (Task 7)

This step opens the webcam and shows the trained model's prediction on the live
picture: the predicted class, its confidence, and the probability of each of the
four classes. It reuses the Task 6 prediction code and the webcam helpers from
data collection. It is an **informational preview only**. Nothing is saved, nothing
is sent to any hardware, and nothing is sorted.

### What you need

- A trained checkpoint, normally `models\best_model.pth`. Make it with the
  training step (`python -m src.train`, Task 4). This program does not train or
  change the checkpoint.
- A webcam. The program checks the camera when it starts.
- No GPU is required. CPU inference works; add `--device cuda` only if you have a
  CUDA-capable GPU and a CUDA build of PyTorch.

### Run it (Windows PowerShell)

```powershell
python -m src.webcam_predict
python -m src.webcam_predict --checkpoint models\best_model.pth
python -m src.webcam_predict --camera 1
python -m src.webcam_predict --device cpu
python -m src.webcam_predict --threshold 0.8
python -m src.webcam_predict --inference-every-n-frames 5
python -m src.webcam_predict --checkpoint models\best_model.pth --camera 0 --device cpu --threshold 0.8 --inference-every-n-frames 3
```

### Run it (Linux)

```bash
python -m src.webcam_predict --checkpoint models/best_model.pth
python -m src.webcam_predict --camera 0 --device cpu
```

### Options

| Option | Meaning |
|--------|---------|
| `--checkpoint` | Checkpoint to use. Default: `best_model.pth` in `paths.model_dir`. The program prints the file it uses. |
| `--camera` | Camera index. Default: `camera.index` from the config (normally `0`). Try `1` if `0` is another device. |
| `--device` | `auto`, `cpu`, or `cuda`. Default: `training.device` from the config. |
| `--threshold` | Optional, between 0 and 1. When the top probability is below it, the window shows a `LOW CONFIDENCE` line. Off when not given. |
| `--inference-every-n-frames` | Run the model on every Nth camera frame (default `1`). The last prediction stays on screen until the next one, so a slower device still shows a smooth picture. |
| `--config` | Config file. Default: `config\config.yaml`. |

The `inference.confidence_threshold` value in the config (0.70) is **not** applied
to this preview. The threshold is used only when you pass `--threshold`.

### Quit

Click the video window, then press `q` or `Esc`. Closing the window with its
close button also stops the program. The camera is released and the window is
closed on every exit path, including errors and Ctrl+C.

### What the window shows

- **Predicted:** the highest-probability class (`metal`, `plastic`, `paper`, or `other`).
- **Confidence:** that highest probability, as a percentage. It comes from the
  softmax of the model's four outputs. It is not calibrated.
- **Four probability lines:** one per class. The predicted class is marked with `>`.
- **LOW CONFIDENCE (below ...):** only when `--threshold` is given and the top
  probability is below it. It is a display flag. There is no fifth class.
- **Footer:** "Model output only - not verified on real materials" and
  "Informational only - no sorting action".

### Errors

| Message | Cause and fix |
|---------|---------------|
| `checkpoint not found ... No other checkpoint is used in its place` | Train a model first, or pass the correct `--checkpoint`. The camera is not opened. |
| `checkpoint architecture ...` / `classes ... do not match` / `image size` | The checkpoint does not match the config. Restore the config used for training. The camera is not opened. |
| `CUDA was requested but is not available` | Use `--device cpu`. |
| `could not get frames from camera index N` | The camera is missing, in use by another app, or blocked. On Windows, check Settings > Privacy > Camera. Try another `--camera` index. |
| `camera stopped delivering valid frames for 3 s` | The camera stopped sending frames while the program ran (unplugged, or taken by another app). Restart it. |
| `threshold must be strictly between 0 and 1` | Use a value such as `0.8`. |
| `--inference-every-n-frames must be at least 1` | Use `1` or more. |
| `OpenCV error ... install the GUI build` | The window could not be created. Install `opencv-python` (not `opencv-python-headless`). |

Exit codes: `0` normal exit (`q`, `Esc`, closed window, or Ctrl+C), `1` runtime
error (checkpoint, camera, frame read, or CUDA), `2` bad config or arguments.

### Limitations

- **Real-world accuracy is unknown.** The classifier is only as good as its
  training data (how many images, how varied, how many objects and sessions). The
  test-split numbers from Task 5 describe that test set only. Do not treat the
  live preview as proof that a material is recognised correctly.
- **Only four classes exist.** An unfamiliar object is still given one of the
  four classes, often with high confidence. A threshold does not catch every
  unfamiliar or ambiguous object.
- **Probabilities are not calibrated.** A 90% value does not mean the prediction
  is right 9 times in 10.
- **Lighting, background, distance, and camera angle change the result.** The
  model sees the centre square of the frame, so objects near the edges can be cut
  off.
- **Prediction is informational only.** Predictions must not drive a sorter, a
  trapdoor, a motor, or any other hardware until a later task has shown that the
  predictions are stable and validated.
- **Not verified with a physical camera here.** The automated tests use fake
  cameras and windows. Test the live window on your own computer before relying
  on it.

## Build a dataset from template photos (synthetic)

This step builds a training dataset from five photos: one object photo per
material and one photo of the empty platform. It isolates each object and pastes
it onto the empty platform many times, with different positions, rotations,
sizes, hues, lighting, shadows, blur, and noise. The output goes into
`data\dataset\<class>\`, in the same layout the collection tool uses, so the
existing validation, split, training, and evaluation steps can use it.

**Synthetic data is only a stand-in for real photos.** Validation numbers on it
overstate real accuracy. Read the limits below before using the results.

### Template folder

Put at least five photos in `data\template\` (the folder next to `data\dataset\`). The
file names matter, not the extensions (`.jpg`, `.png`, `.bmp`, and `.jpeg` all
work; capitals do not matter):

| File | Contents |
|------|----------|
| `Metal.jpg` | One metal object on the platform |
| `Plastic.jpg` | One plastic object on the platform |
| `Paper.jpg` | One paper object on the platform |
| `Organic.jpg` | One organic object on the platform (saved as class `other`) |
| any other name, e.g. `empty.jpg` | The same platform with nothing on it |

**More than one face of the same object.** Add extra photos of the same object
(another side, a different pose) with the class name followed by a separator or a
number: `Metal_2.jpg`, `Metal 3.png`, `Metal-side.jpg`, `Metal2.jpg`. Works for
`Plastic`, `Paper` and `Organic` too. All faces of a class count as one object, so
they stay in the same train/validation/test split. A name that merely starts with
a class name, like `metallic.jpg`, is not a face; it is treated as the empty
platform photo, and having two such photos is an error.

All of the photos must be the same size and taken with the same camera, framing,
and platform. An object must cover between 0.2% and 50% of the frame. The platform
colour is read from the photo's outer edge, so keep the platform visible along
the edges of the frame.

### Run it (Windows PowerShell)

```powershell
python -m src.synthesize_dataset
python -m src.synthesize_dataset --per-template 300 --seed 7
python -m src.synthesize_dataset --template-dir C:\Photos\template --overwrite
```

| Option | Meaning |
|--------|---------|
| `--template-dir` | Folder with the template photos. Default: `data\template`. |
| `--per-template` | Images generated per class, split evenly across its faces (the first faces get any remainder). Must be at least the number of faces. Default: `200` (800 images in total). |
| `--seed` | Random seed. The same seed and settings produce the same images. Default: `42`. |
| `--overwrite` | Move an existing `data\dataset\` and grouping file to `.<time>.bak` copies first. Without it, the command refuses to replace existing images. |
| `--config` | Config file. Default: `config\config.yaml`. |

### Outputs

- `data\dataset\metal\`, `plastic\`, `paper\`, `other\`: the generated JPEGs
  (named `synth_<photo name>_<number>.jpg`).
- `data\group_metadata.csv`: marks every generated image with the template object
  it came from (all faces of a class share one). All variants of one object are then kept in the same split
  (train, validation, or test), so near-copies of one object cannot leak into the
  test split.
- `data\synthetic_report.json`: the settings, seed, object area found in each
  template, the images made from each face, and the notes above.

The templates are never changed. Generation is written to a temporary folder
first, so a failed run leaves your current dataset in place.

### How it works

1. **Isolate:** the object photo is compared with the empty platform photo after
   matching exposure on the platform's border. A threshold finds the object,
   which is refined with GrabCut and cleaned up (largest piece, holes filled).
2. **Compose:** the cut-out object is placed on the empty platform at a random
   spot, rotated by a random angle, scaled, and given a random hue, saturation,
   and brightness shift. A soft shadow is added. The whole scene then gets random
   exposure, colour temperature, contrast, light falloff, optional slight blur,
   and sensor noise. The JPEGs are saved at random quality between 85 and 95.

Settings such as the hue range (`max_hue_shift_degrees`, default ±30°) and the
scale range (`min_scale`, `max_scale`) are fixed in `src/synthesize_dataset.py`
(`SynthesisSettings`). They are not in `config.yaml` yet.

### Limits you must read

- **One object per material means no real test of new objects.** Every variant
  of a template object lands in one split. With one object per class (however many faces), the
  split tool (`python -m src.prepare_dataset`) cannot split by object, so it
  splits that class image by image and prints a warning. Near-copies of the same
  object then appear in train, validation and test, so the validation and test
  numbers are optimistic. A class with three or more objects is still split by
  object, and each class is decided on its own. Collect more objects per class
  for a real test.
- **Synthetic images are a rough stand-in.** Real shadows, reflections, edges,
  and lighting differ from the pasted versions. The model may do well on the
  synthetic images and poorly on real ones.
- **Cut-outs are approximate.** Shadows near the object, and any object that
  is dark, transparent, reflective, or close in colour to the platform, can
  leave part of the object out or include some background. On the test
  photos used during development, the cut-out overlapped the true object by
  79–100%.
- **Test with your own photos first.** Check a contact sheet of the output
  (a few images per class) before training on it.

## Live sorting with the STM32 (serial)

This step reads the USB camera, classifies the object on the platform with the
trained model, and sends one digit command to the STM32 over serial. The STM32 confirms the
command and reports each step of the sort. **Do not connect the STM32 until the dry run has been
checked with real objects.** A wrong prediction sends an object to the wrong bin.

### Setup

1. Install the serial library once: `pip install pyserial` (it is also in
   `requirements.txt`).
2. Open `config\serial.txt` and fill in the two values:

   ```
   port = COM3
   baud = 115200
   ack_timeout = 2
   ```

   `port` is the COM port the STM32 appears on (check Device Manager). `baud` must
   match the STM32 firmware. `ack_timeout` is how long to wait for the reply, in
   seconds (default 2).

3. Put the trained model at `models\best_model.pth`, and the five template photos
   at `data\template\` (the empty platform photo is needed for detecting objects).

### Dry run first (no serial port opened)

```powershell
python -m src.live_sorter --dry-run
```

The window shows the live picture. When an object is confidently recognised, the
text `[dry run] would send 1 (metal, 93.0%)` is printed. Nothing is sent to the
STM32. Check several objects in several positions before going further.

### Run with the STM32 connected

```powershell
python -m src.live_sorter
python -m src.live_sorter --threshold 0.85 --stable-frames 8
python -m src.live_sorter --camera 1 --device cpu
```

| Option | Meaning |
|--------|---------|
| `--threshold` | Send only when the top probability is at least this. Default: `inference.confidence_threshold` from the config (0.70). |
| `--stable-frames` | The same class must be predicted this many frames in a row before sending. Default: 5. |
| `--camera` | Camera index. Default: `camera.index` from the config. |
| `--device` | `auto`, `cpu`, or `cuda`. |
| `--serial-config` | Serial settings file. Default: `config\serial.txt`. |
| `--dry-run` | Print the codes only. The serial port is never opened. |

Press `q` or `Esc` in the window to stop.

### If an empty platform is read as "object present"

The sorter decides "object" when more than 2% of the picture differs from your
empty-platform photo. That photo must come from the same camera, position and
lighting as the live view. To see what your camera measures, with the platform
**empty**, run:

```powershell
python -m src.check_empty
python -m src.check_empty --camera 0 --seconds 8
```

It prints the share of changed pixels, says whether the empty platform is
recognised, and saves `data\empty_check.png` (empty photo, live frame, and the
pixels that differ). If it reports a problem, retake the empty photo with the
camera the sorter uses and replace the empty file in `data\template\`.

### Protocol

The commands match the STM32 firmware. Every command ends with a newline (`\n`).

| Material | Command sent | Firmware name in the reply |
|----------|--------------|----------------------------|
| metal    | `1\n`        | `METAL` |
| plastic  | `2\n`        | `PLASTIC` |
| paper    | `3\n`        | `PAPER` |
| organic  | `4\n`        | `ORGANIC` (the project class is `other`) |

The codes come from `cups` in `config\config.yaml`, which must stay 1 to 4.

One sort, as the firmware answers it:

1. `REQUESTED:<NAME>` confirms the command. A different name is an error.
2. The firmware may print the colour reading (`METAL`, `PLASTIC`, `PAPER`, or `ORGANIC`).
3. `POSITION CONFIRMED`, then `GATE OPEN`, then `GATE CLOSED`.
4. `SORTING COMPLETE` ends the sort. `POSITION TIMEOUT` or `STOPPED` also end it,
   and the script reports them without treating them as a crash.

Two other messages are used. At start-up the script sends `STATUS\n` and waits for
`LAST_SAVED:<name>`. When the script exits, it sends `STOP\n` and waits for `STOPPED`.

The firmware replies `BUSY` when a sort is still running and `INVALID COMMAND` for an
unknown byte. Either one stops the script.

**Organic needs a measured range.** The firmware's yellow colour range for ORGANIC is
still empty, so organic objects are not detected until those values are filled in.
Metal, plastic, and paper are unaffected.

### How it decides when to send

- **One send per object.** After a send, nothing else is sent until the platform
  is empty again.
- **Object detection.** The frame is compared with the empty platform photo. If
  enough pixels differ, an object is on the platform. Strong lighting changes
  can look like an object, so keep the lighting steady.
- **Stable and confident.** The same class must be predicted for `--stable-frames`
  frames in a row, with a probability at or above the threshold. Otherwise nothing
  is sent.

### Errors and safety stops

- If the STM32 does not reply in time, replies with the wrong name, sends `BUSY` or
  `INVALID COMMAND`, or the reply is cut off before its newline, **the script stops
  at once**. It sends `STOP`, closes the port, and does not send again.
- `POSITION TIMEOUT` and `STOPPED` are not crashes. The script reports them and
  carries on with the next object.
- A serial port that cannot be opened (wrong port, in use by another program) stops
  the script before the camera starts.
- Settings problems in `serial.txt` (blank port or baud, unknown key, bad number)
  are reported with the line number and exit with code 2.

Exit codes: `0` normal stop (`q`, `Esc`, closed window, or Ctrl+C), `1` runtime
error (model, camera, serial, template folder, or a bad reply), `2` bad config or
arguments.

### Limits

- **Accuracy is unknown.** The model was trained on synthetic images. Until it has
  been tested on new, real photos of your objects, a wrong material can be sent.
  The camera decides the material. The STM32 colour sensor only checks the bin in
  front of it.
- **Only four classes exist.** Anything that is not one of them still gets one of
  the four codes if the model is confident. A confidence threshold does not catch
  every unfamiliar object.
- **Presence detection is simple.** It compares the frame with one empty photo, so
  a change in lighting, a moved camera, or a hand in view can count as an object.
- **Not verified on hardware here.** The tests use fake cameras and a fake serial
  port, and a loopback test used a pseudo-terminal. The real STM32, real camera,
  and Windows COM ports have not been tested.

## Sorter window (camera, detected material, and the colour-sensor bin)

`src.sorter_gui` does the same job as `src.live_sorter` (same `config\serial.txt`,
same codes, same decision rules), with a window that shows three things:

- **The camera picture.**
- **Detected material:** a bar with the material and its probability, for example
  `PLASTIC  91%`. It says `Unsure` when the probability is below the threshold.
- **Colour sensor is over:** four segments, METAL, PLASTIC, PAPER and ORGANIC. The
  segment for the bin under the sensor right now is lit. It says `nothing in front
  of the sensor` when the sensor sees no colour.

```powershell
python -m src.sorter_gui --dry-run
python -m src.sorter_gui
python -m src.sorter_gui --camera 1 --threshold 0.85
```

It takes the same options as `live_sorter` (`--checkpoint`, `--camera`, `--device`,
`--threshold`, `--stable-frames`, `--serial-config`, `--dry-run`, `--config`).
Close the window, or press q or Esc, to stop. The STM32 is sent `STOP` and the port
is closed.

**The bin bar needs the new firmware.** Flash
`firmware\stm32_sorter_bluepill\stm32_sorter_bluepill.ino` to the Blue Pill. It
prints `BIN:METAL`, `BIN:PLASTIC`, `BIN:PAPER`, `BIN:ORGANIC` or `BIN:NONE` whenever
the colour under the sensor changes, also while idle. With the older firmware the bar
says `waiting for the colour sensor...`. Everything else still works.

Notes:

- The window needs tkinter. The standard Python installer for Windows includes it.
- A sort runs in the background, so the camera and the bars keep updating while the
  STM32 works. Only one sort runs at a time. A new object is not sent until the
  previous sort has finished.
- The firmware reads the colour once every 300 ms while idle. Each read takes a
  fraction of a second, so commands can take up to about half a second to start.
- If the camera stops giving frames for 3 seconds, the window closes with an error.
- If the STM32 sends an unexpected reply, the window stops sending and shows the
  reason in the status lines. Close it and fix the cause before running again.

## Train on all the dataset images (no split)

Use this step when each material is a single object, so the split step cannot
run (see the synthetic dataset section). It trains on every image in
`data\dataset\<class>\` and saves the model after the last epoch. There is no
validation set and no test set, so **this step does not measure accuracy**.

```powershell
python -m src.train_all
python -m src.train_all --epochs 15 --overwrite
python -m src.train_all --no-pretrained
```

| Option | Meaning |
|--------|---------|
| `--epochs` | Number of passes over the images. Default: `training.epochs` from the config (30). |
| `--device` | `auto`, `cpu`, or `cuda`. |
| `--no-pretrained` | Start from random weights instead of the ImageNet download. |
| `--overwrite` | Replace existing model files. The old files are kept as `.bak` copies. |
| `--output-dir` | Where to save. Default: `paths.model_dir` (`models\`). |

Outputs in `models\`:

- `best_model.pth`: the checkpoint that `predict`, `webcam_predict`, and
  `live_sorter` load.
- `training_history.csv`: one row per epoch.
- `training_summary.json`: settings and the notes above.

The same augmentation, model, and optimiser as `src.train` are used. The
printed accuracy is on training images, so it is not a test score. Training on
CPU takes time: with 800 images at 224 pixels, expect roughly 15 minutes for 30
epochs on a typical laptop CPU (much faster with an NVIDIA GPU).

**What you cannot tell from this step:** whether the model works on new photos
of the same objects in new orientations. For that, take a separate set of real
photos that are never used to make synthetic images, and test on them (not built
yet).

## Configuration

All settings live in [`config/config.yaml`](config/config.yaml). Every key is
required; there are no hidden defaults.

| Section / key                      | Meaning                                                    |
|------------------------------------|------------------------------------------------------------|
| `classes`                          | Exactly `metal`, `plastic`, `paper`, `other` (order = label index) |
| `cups`                             | Class → cup ID (unique positive integers)                  |
| `paths.dataset_dir`, `paths.model_dir` | Folders for data and trained models (must not be nested in each other) |
| `dataset.group_metadata`, `dataset.manifest`, `dataset.report` | Grouping CSV (optional), manifest CSV, and split report. None may be inside `paths.dataset_dir`. |
| `image.width`, `image.height`      | Model input size in pixels (32–2048)                       |
| `augmentation.*`                   | Training-only augmentation ranges (rotation, shift, zoom, brightness, contrast) |
| `split.train`, `split.val`, `split.test` | Each in (0, 1); must sum to 1                        |
| `random_seed`                      | Non-negative integer                                       |
| `training.batch_size`, `training.epochs` | Positive integers                                    |
| `training.learning_rate`           | In (0, 1]                                                  |
| `training.weight_decay`            | In [0, 1); AdamW weight decay                              |
| `training.early_stopping_patience` | Positive integer; epochs without validation-loss improvement before stopping |
| `training.num_workers`             | Integer 0–16; data-loading processes (0 is safest on Windows) |
| `training.device`                  | `auto` (CUDA if available), `cpu`, or `cuda`               |
| `training.pretrained`              | `true` or `false` (ImageNet weights)                       |
| `model.architecture`               | `mobilenet_v3_small` (default), `mobilenet_v2`, `efficientnet_b0`, or `resnet18` |
| `inference.confidence_threshold`   | Strictly between 0 and 1                                   |
| `camera.index`                     | Non-negative integer (0 = default webcam)                  |

Relative paths resolve against the project root. Paths do not have to exist yet.

Loading validates strictly. Missing keys, unknown keys (e.g. typos), duplicate
YAML keys, wrong types (`"32"`, `yes`, `true` where a number is expected), and
out-of-range values all raise `ConfigError` naming the offending key.

To change the cup mapping, edit `cups` and keep every ID unique:

```yaml
cups:
  metal: 4
  plastic: 3
  paper: 2
  other: 1
```

Using the config from Python:

```python
from src.config import load_config

config = load_config()               # reads config/config.yaml
print(config.cup_for("plastic"))     # 2
print(config.confidence_threshold)   # 0.7
```

## Project layout

```
config/config.yaml        Validated settings
src/config.py             Config loading and validation
src/data_collection.py    Webcam image collection
src/dataset.py            Dataset validation, manifest reader
src/prepare_dataset.py    Group-aware train/val/test split, manifest and report
src/preprocessing.py      Image loading, resizing, augmentation, normalisation
src/model.py              Transfer-learning model and checkpoint save/load
src/train.py              Training loop, validation, early stopping, outputs
src/evaluate.py           Test-split evaluation, metrics, result files
src/predict.py            Single-image prediction (inference only)
src/webcam_predict.py     Live webcam prediction preview (informational only)
src/synthesize_dataset.py Template-based synthetic dataset (isolate object, compose variants)
src/live_sorter.py        Live camera sorting: classify, send one byte to the STM32, check reply
src/train_all.py          Train on all dataset images, no split (saves best_model.pth from the last epoch)
config/serial.txt         Serial port and baud rate for the STM32
tests/test_config.py      Config tests
tests/test_dataset.py     Dataset validation tests
tests/test_prepare_dataset.py   Split, grouping, manifest and CLI tests
tests/test_preprocessing.py     Preprocessing and augmentation tests
tests/test_train.py       Model, checkpoint, manifest, early-stopping and CPU training tests
tests/test_evaluate.py    Test-split selection, metrics, outputs and error tests
tests/test_predict.py     Single-image prediction, softmax, errors and output tests
tests/test_webcam_predict.py    Live preview: mapping, overlay, loop keys, camera failures, CLI (fake camera/window)
tests/test_synthesize_dataset.py   Template discovery, isolation, compositing, outputs, overwrite and CLI
tests/test_live_sorter.py Serial settings, protocol, presence, one-send-per-object, stop on bad reply, CLI
tests/test_train_all.py   Training on all images, last-epoch checkpoint, overwrite and CLI
tests/test_data_collection.py   Collection tests (fake camera/window)
tests/helpers.py          Shared test helpers
requirements.txt          PyYAML, opencv-python
```
