# Graph Neural Network for Graph Classification [DL Hackaton]

**Giuseppe D'Addario** 2177530, **Lorenzo Benucci** 2219690.

---

This project implements a framework for graph classification tasks. It **features Graph Neural Network** (GNN) architectures, including **GNNtransformers** and the advanced loss function `GCOD`.  

Two distinct strategies are proposed:

* `GIN` model for C and D datasets
* `GINE + transformer` model for A and B datasets





## Overview of the Method

Initial experiments were conducted using simple `GIN` models, but the resulting accuracy and F1 scores remained low, suggesting that these models lacked sufficient generalization capability. Subsequently, hybrid approaches combining a `GIN` and a `Transformer` were explored, yielding significantly improved performance on datasets C and D. Finally, a more advanced model was implemented, integrating `GINEConv` and `Transformer` layers within a unified architecture. This model also incorporated **batch normalization**, a **linear classifier** with **LeakyReLU** activation, and **dropout** for regularization.

---
#### <u>GCOD</u>
To mitigate overfitting due to noise, we found that using the **GCOD** loss instead of the standard *Cross-Entropy* significantly improves performance.

*   **How it works:**
    1.  Maintains a learnable parameter `u` for each training sample, representing the model's confidence that the sample is "clean".
    2.  Computes class `centroids` based on embeddings of (presumably) cleaner samples.
    3.  Calculates `soft_labels` based on embedding similarity to centroids.
    4.  **L1 (Classification Loss):** A modified cross-entropy where logits are adjusted by `u` and true labels, and the target is the `soft_label`.
    5.  **L2 (u-Update Loss):** Encourages `u` to align with model predictions vs. true labels.
    6.  **L3 (Regularization Loss):** A DKL-based regularizer using `u` and model's confidence in the true class, weighted by `(1 - atrain_overall_accuracy)`.
*   `atrain_overall_accuracy`: Global accuracy on the training set, used to modulate GCOD's behavior.



## Procedure

1. **Preprocessing Data:**
   - The datasets are loaded and preprocessed into graph structures.
   - Graphs are represented by node features, edge indices, and edge attributes.


2. **Training:**
   - The model is randomly initialized and starts the training. The convergence of the accuracy and loss is noted around 80-100 epochs training; going on would mean overfitting on the train set.
   - For training on a dataset:
     ```bash
     python main.py --train_path ../A/train.json.gz --test_path ../A/test.json.gz --num_epochs 100
     ```
   - the trained models' weights are saved. F1 score metric is used to update, and choose, the best model.

3. **Inference and Predictions:**
   - The two trained models can be used to predict on the test sets.
   - To make test set labelling with predictions:
     ```bash
     python main.py --test_path ../A/test.json.gz 
     ```

## Code Structure

The script inscludes a `requirements.txt` containing all dependencies needed to run the code. Other files are organized as below:

- **`main.py`:** The main script for training, evaluating, and predicting with the model.
- **`loadData.py`:** Handles data loading, preprocessing, and dataset preparation.
- **`conv.py`:** Implements convolutional layers and custom graph convolution operations.
- **`models.py`:** Defines the architecture of the neural network models used in the project.
- **`train.py`:** Script and functions responsible for training the model.
- **`evaluation.py`:** Provides evaluation methods and metrics to assess model performance.
- **`loss.py`:** Contains the loss functions used for model training.
- **`statistics.py`:** Contains functions to compute and report statistical metrics and performance summaries.

---






