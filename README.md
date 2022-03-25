This repository contains the source code used to produce the results of the following paper:

Paul Engstler, Matthias Keicher, David Schinz, Kristina Mach, Alexandra
S. Gersing, Sarah C. Foreman, Sophia S. Goller, Juergen Weissinger, Jon
Rischewski, Anna-Sophia Dietrich, Benedikt Wiestler, Jan S. Kirschke,
Ashkan Khakzar, and Nassir Navab; **Interpretable Vertebral Fracture Diagnosis**

The code comprises two central contributions of the paper:
* Jupyter notebook to determine the activations of detector units highly correlated with a positive prediction with [Network Dissection](http://netdissect.csail.mit.edu/) (`netdissect_tasks.ipynb`)
* Concept visualization system that gives user insight into the decision-making process of the neural network on a detector unit level (`app.py`)

For a live demo of our concept visualization system, please see this [Hugging Face Space](https://huggingface.co/spaces/paulengstler/interpretable-vertebral-fracture-diagnosis).

Note that this repository contains a fork of [Network Dissection](https://github.com/CSAILVision/NetDissect) in the folder `netdissect`. It contains our extensions to support three-dimensional data and additional export options.

## Training and Evaluation
### Requirements
Please observe `requirements.txt` for Python package requirements as well as `packages.txt` for system dependencies.

### Data
Part of the dataset we used to train the final model is publicly available. Please consult the page of the [VerSe dataset](https://github.com/anjany/verse) for details. Note that this dataset does not contain individual vertebrae that are required for training the model. As we used a proprietary pipeline to generate those, we regret being unable to publish them.

Once the spine images have been processed, create a file called `fxall_labels.csv` that is read by the datamodule. Place all vertebrae (or their corresponding directories) in a folder called `raw`.

The datamodule (as configured in `config.yaml`) queries the following columns of `fxall_labels.csv`:
  * **verse_id**: identifier of the vertebra
  * **fx**: Fracture indicator
  * **level_idx**: Numerical index of the vertebra level (T1: 8)
  * **path**: relative path to the npy file, contained in 'raw' folder
  * **split_1**: split specified by the configuration ('training', 'validation', 'test')

Thus, an example row might be: `verse765, 0, 8, verse20/dataset-02validation/sub-verse765/sub-verse765_8.npy, training`

### Training
To train the model please run `train.py`. The training process as well as the model itself are configured through modification of `config.yaml`. If saving is enabled, the model will be stored in a directory called `saved_models`. Information about the training process as well as metrics may be tracked to [Weights & Biases](https://wandb.ai/).

### Investigating Detector Units
The Jupyter notebook `netdissect_tasks.ipynb` contains functionality to compute the top images for each detector unit as well as the units that are highly correlated with a positive prediction. The images contain a highlighted area indicating the corresponding activation of a detector unit, and can be exported either as a collection of sliced views or a collage of single slices. It is possible to consider only positive or false positive examples.

Furthermore, the notebook also produces the positive correlation rank and activation threshold for each unit to be inlined in the concept visualization system script `app.py`.

### Concept Visualization System
To run the self-contained concept visualization system, please run `streamlit run app.py`. Please observe that it contains inlined information about detector units that were acquired with the Jupyter notebook `netdissect_tasks.ipynb`.