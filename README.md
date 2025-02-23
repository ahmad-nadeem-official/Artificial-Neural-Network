🌟 **Diabetes Prediction using Artificial Neural Network (ANN)** 🌟
===================================================================

This project uses **Artificial Neural Networks (ANN)** to predict the likelihood of diabetes in patients based on medical data. The model is trained using the **Keras library** and tested on a real-world **diabetes dataset**. Let's dive into the details of the project! 🚀

📝 **Table of Contents**
------------------------

1.  Project Overview
2.  Technologies Used
3.  Dataset
4.  Model Architecture
5.  Training & Testing
6.  Results & Accuracy
7.  How to Run
8.  Conclusion

* * *

💡 **Project Overview**
-----------------------

This project aims to predict whether a person has diabetes based on a set of features such as age, blood pressure, glucose levels, etc. The model is built using an **Artificial Neural Network (ANN)**, with multiple hidden layers, dropout for regularization, and batch normalization for better performance.

* * *

⚙️ **Technologies Used**
------------------------

*   **Python** 🐍
*   **Keras** (Deep Learning Framework) 🧠
*   **Scikit-learn** (Machine Learning) 📊
*   **Pandas** (Data Manipulation) 🧮
*   **NumPy** (Numerical Computing) 🔢
*   **Matplotlib & Seaborn** (Data Visualization) 📈
*   **TensorFlow** (Backend for Keras) 💻

* * *

📊 **Dataset**
--------------

*   **Source**: The dataset used in this project is a **diabetes dataset** that contains various health-related features to predict the presence of diabetes.
*   **Columns**: The dataset includes features like:
    *   Glucose
    *   Blood Pressure
    *   Skin Thickness
    *   Insulin
    *   BMI (Body Mass Index)
    *   Diabetes Pedigree Function
    *   Age
    *   Outcome (target: 1 if diabetic, 0 if non-diabetic)

* * *

🧠 **Model Architecture**
-------------------------

The Artificial Neural Network (ANN) has the following layers:

1.  **Input Layer**: Takes in 8 features (input\_dim=8).
2.  **Hidden Layers**:
    *   5 hidden layers with progressively increasing units (6 → 8 → 10 → 12 → 14).
    *   ReLU activation function for non-linearity.
    *   Dropout (0.2) for regularization to prevent overfitting.
    *   Batch Normalization for stabilizing training.
3.  **Output Layer**: A single neuron with **sigmoid activation** for binary classification (0 or 1).

* * *

🏋️‍♀️ **Training & Testing**
-----------------------------

### **Model Compilation**:

*   **Optimizer**: Adam optimizer for efficient gradient descent.
*   **Loss Function**: Binary Cross-Entropy, suitable for binary classification.
*   **Metrics**: Accuracy metric to evaluate the model performance.

### **Model Training**:

python

Copy

`ann.fit(x_train, y_train, batch_size=32, epochs=10)` 

The model is trained for 10 epochs with a batch size of 32. The training process evaluates the network using the validation set.

### **Prediction**:

After training, the model predicts diabetes presence for the test dataset:

python

Copy

`prd = ann.predict(x_test)` 

The model's prediction is then converted into a binary outcome (0 or 1) based on a threshold of 0.5.

* * *

🏅 **Results & Accuracy**
-------------------------

After evaluating the model, here is the **confusion matrix** and **accuracy score** for the test data:

### **Confusion Matrix:**

lua

Copy

`[[100   0]
 [ 52   2]]` 

*   **True Positives (TP)**: 100
*   **False Positives (FP)**: 0
*   **False Negatives (FN)**: 52
*   **True Negatives (TN)**: 2

### **Accuracy Score:**

Model

Accuracy

ANN (Deep Learning)

**66.23%**

* * *

🚀 **How to Run**
-----------------

1.  **Clone this repository**:
    
    bash
    
    Copy
    
    `git clone https://github.com/ahmad-nadeem-official/Artificial-Neural-Network.git` 
    
2.  **Install dependencies**:
    
    bash
    
    Copy
    
    `pip install -r requirements.txt` 
    
3.  **Run the script**:
    
    bash
    
    Copy
    
    `python diabetes_prediction_ann.py` 
    

Make sure to adjust the dataset path if you're using your local version.

* * *

📝 **Conclusion**
-----------------

This project demonstrates how Artificial Neural Networks can be applied to predict diabetes based on various health features. While the accuracy of the model currently stands at 66.23%, there is potential for improvement with more data, better hyperparameter tuning, and advanced techniques like feature engineering or different architectures.

Feel free to modify the model, experiment with different neural network configurations, or apply this methodology to other binary classification problems. 🌟

* * *

📢 **Contact Information**
--------------------------

For any questions or feedback, feel free to reach out:

*   **Email**: ahmadnadeem095@gmail.com