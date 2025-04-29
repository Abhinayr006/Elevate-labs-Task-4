# Task 4: Binary Classification Using Logistic Regression

This project represents the fourth task of my internship at *Elevate Labs*, focusing on developing a binary classification model using **Logistic Regression** to predict tumor malignancy.

## Objective

The goal of this task was to create a binary classifier with logistic regression, assess its performance using various evaluation metrics, and adjust the decision threshold to optimize results.

## Technologies and Libraries Used

- **Python**: Core programming language for implementation
- **Scikit-learn**: For model building, preprocessing, and evaluation
- **Pandas**: For data manipulation and analysis
- **Matplotlib**: For visualizing the ROC curve and sigmoid function
- **Seaborn**: For enhanced data visualization (if applicable)

## Workflow and Implementation Steps

1. **Dataset Acquisition**  
   I utilized the *Breast Cancer Wisconsin (Diagnostic) Dataset* to classify tumors as malignant (M) or benign (B). The dataset was sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)).

2. **Data Preprocessing**  
   - Removed irrelevant columns: `id` and `Unnamed: 32` (if present in the dataset).  
   - Encoded the target variable `diagnosis` into binary format: Malignant (M) = 1, Benign (B) = 0.  
   - Split the dataset into training (80%) and testing (20%) sets using `train_test_split`.  
   - Standardized the feature values using `StandardScaler` to ensure consistent scaling.

3. **Model Development**  
   - Implemented a Logistic Regression model using Scikit-learn's `LogisticRegression` class.  
   - Trained the model on the standardized training data.  
   - Generated predictions on the test set for evaluation.

4. **Model Assessment**  
   The model's performance was evaluated using the following metrics:  
   - **Confusion Matrix**: To visualize true positives, true negatives, false positives, and false negatives.  
   - **Precision, Recall, and F1-Score**: To measure the balance between correct positive predictions and missed positives.  
   - **ROC-AUC Score**: To assess the model's ability to distinguish between classes.  
   - **ROC Curve**: Plotted to visualize the trade-off between true positive rate and false positive rate.

5. **Threshold Optimization**  
   - Experimented with different decision thresholds (e.g., 0.3) to adjust the classification boundary.  
   - Analyzed the impact of threshold changes on precision, recall, and overall model performance.

## Project Files

- `ElevateLabsTask-4.ipynb`: Jupyter Notebook containing the complete implementation, including data preprocessing, model training, evaluation, and threshold tuning.  
- `Breast_Cancer_Wisconsin.csv`: The dataset used for this task.  
- `README.md`: Project documentation summarizing the task and findings.

## Dataset Details

- **Source**: Breast Cancer Wisconsin (Diagnostic) Dataset from UCI Machine Learning Repository.  
- **Description**: The dataset contains features computed from digitized images of fine needle aspirates (FNA) of breast masses, used to predict whether a tumor is malignant or benign.

## Performance Metrics

The following metrics were used to evaluate the logistic regression model:  
- **Confusion Matrix**: Summarized prediction outcomes.  
- **Precision**: Measured the proportion of correct positive predictions.  
- **Recall**: Measured the proportion of actual positives correctly identified.  
- **F1-Score**: Harmonic mean of precision and recall for balanced evaluation.  
- **ROC-AUC Score**: Quantified the model's discriminative power.  
- **ROC Curve**: Visualized the model's performance across different thresholds.

## Key Takeaways

- Gained hands-on experience in applying Logistic Regression for binary classification tasks.  
- Learned to evaluate models comprehensively using confusion matrices, precision, recall, F1-score, and ROC-AUC.  
- Understood the importance of threshold tuning to achieve a balance between precision and recall based on the problem's requirements.
