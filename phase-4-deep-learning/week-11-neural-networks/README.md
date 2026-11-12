# Week 11 — Neural Networks
### Task: Introduction to Deep Learning
**Next Gen Software Hub Pvt. Ltd. — AI & ML Internship Program**

---

## 🎯 Learning Objectives
By the end of this week, students will master the core foundations of Deep Learning:
- **Artificial Neurons (Perceptrons):** Inputs, weights, bias, and summation junction.
- **Neural Networks:** Input layer, hidden layer(s), and output layer topologies.
- **Activation Functions:** Mathematical role of non-linearity (Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax).
- **Loss Functions:** Quantifying prediction error (Binary Cross-Entropy, Categorical Cross-Entropy, Mean Squared Error).
- **Forward Propagation:** Matrix multiplications mapping input vectors to output predictions.
- **Backpropagation:** Computing partial derivatives via the Chain Rule.
- **Gradient Descent & Optimizers:** Updating weights iteratively (SGD, Adam, learning rate schedules).
- **Epochs and Batches:** Full dataset passes vs mini-batch gradient updates.

---

## 🛠️ Practical Assignment
Build a basic neural network for a classification problem using **PyTorch** or **TensorFlow/Keras**.

### Requirements:
1. Load a multi-feature classification dataset (e.g. `datasets/sensor_fault_detection.csv`).
2. Construct an Artificial Neural Network (ANN) architecture with at least 2 hidden layers.
3. Apply appropriate activation functions (ReLU for hidden layers, Sigmoid/Softmax for output).
4. Implement a forward and backward training loop tracking loss across epochs and mini-batches.
5. Generate training vs validation loss and accuracy graphs.
6. Evaluate the model on test data using Accuracy, Precision, Recall, and Confusion Matrix.

---

## 📋 Submission Deliverables
- **Jupyter Notebook (`starter_notebook.ipynb`):** Clean, runnable code with markdown commentary.
- **Model Architecture:** Diagram or programmatic summary of layers, parameters, and shapes.
- **Training Graphs:** Exported plots showing Loss and Accuracy curves across training epochs.
- **Evaluation Results:** Metric scorecard and confusion matrix.
