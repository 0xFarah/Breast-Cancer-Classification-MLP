# Breast-Cancer-Classification-MLP
This repository contains a pure Python and NumPy implementation of a Multi-Layer Perceptron (MLP) neural network to classify breast cancer tumors as Malignant or Benign


## Dataset Information: `wdbc.data`

The core of this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset**. 
* **Source:** UCI Machine Learning Repository.
* **Format:** `wdbc.data` (Comma-separated values).
* **Content:** 569 instances, each with 32 attributes. 
    * **Column 1:** Patient ID (ignored during training).
    * **Column 2:** Diagnosis (M = Malignant, B = Benign).
    * **Columns 3-32:** 30 real-valued numerical features (radius, texture, perimeter, area, etc.).

### How to use the Data:
1. Ensure the file is named exactly `wdbc.data`.
2. **If using Google Colab:** The script will automatically prompt you to upload the file when you run the first cell.
3. **If running locally:** Place the `wdbc.data` file in the same directory as the script.


##  Code Walkthrough

The project is structured to handle the machine learning pipeline from raw data to evaluation results:

### 1. Data Preprocessing (`get_clean_data`)
Before feeding data into the neural network, the code performs:
* **Label Encoding:** Converts categorical labels ('M', 'B') into numerical values (1, 0).
* **Standardization:** Uses `StandardScaler` to scale features so they have a mean of 0 and a standard deviation of 1. This is crucial for the convergence of the MLP.
* **Stratified Splitting:** Splits data into 80% Training and 20% Testing while maintaining the same ratio of malignant/benign cases in both sets.

### 2. MLP Architecture
The network is built from scratch with a flexible architecture:
* **Input Layer:** 30 nodes (matching the dataset features).
* **Hidden Layer:** One hidden layer where the number of nodes ($k$) can be varied to observe performance changes.
* **Output Layer:** 1 node with a **Sigmoid** activation function for binary probability.



### 3. Training Mechanism
The training loop implements the **Backpropagation** algorithm:
* **Forward Pass:** Computes pre-activations and activations for each layer.
* **Loss Calculation:** Uses the **Squared-Error** function as the objective to minimize.
* **Backward Pass:** Computes gradients using the chain rule.
* **Weight Updates:** Adjusts weights and biases based on the specified **Learning Rate** ($\alpha$).
![Hidden Nodes Accuracy](<img width="1389" height="490" alt="Accuracy and loss plot" src="https://github.com/user-attachments/assets/12cb2b38-5c8e-4e16-a045-48a704d8f58a" />)

### 4. Advanced Evaluation
The code doesn't just train once; it performs:
* **Learning Rate Analysis:** Compares how different rates ($1.0, 0.5, 0.1, 0.01$) affect the loss over time.
* **Hidden Node Analysis:** Evaluates accuracy across different hidden layer sizes ($5, 10, 15, 20, 25, 30$) to find the optimal complexity.



###  Visualizations
The script generates plots to visualize:
1. **Training/Test Loss** per epoch.
2. **Learning Rate Comparison** to see which rate converges fastest.
3. **Accuracy vs. Hidden Nodes** to identify underfitting or overfitting.
