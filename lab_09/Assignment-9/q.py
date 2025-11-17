import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math # Used for np.exp

# --- Configuration ---
# Set to True to display EDA plots (can be time-consuming)
SHOW_EDA_PLOTS = True 
# Set a random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("--- Lab Assignment: Logistic Regression from Scratch ---")

# --- 1. Load Data ---
print("\n--- 1. Loading Data ---")
try:
    # The file is specified as semicolon-delimited
    df = pd.read_csv('bank-full.csv', delimiter=';')
    print("Data loaded successfully.")
    print("First 5 rows:")
    print(df.head())
    print("\nData Info:")
    df.info()
except FileNotFoundError:
    print("Error: 'bank-full.csv' not found. Please make sure the file is in the same directory.")
    exit()

# --- 2. Exploratory Data Analysis (EDA) ---
print("\n--- 2. Exploratory Data Analysis (EDA) ---")

# Summary statistics for numerical features
print("\nNumerical Features Summary:")
print(df.describe())

# Distribution of the target variable 'y'
print("\nTarget Variable 'y' Distribution:")
print(df['y'].value_counts())
print(f"Percentage 'yes': {df['y'].value_counts(normalize=True)['yes'] * 100:.2f}%")
print(f"Percentage 'no': {df['y'].value_counts(normalize=True)['no'] * 100:.2f}%")

if SHOW_EDA_PLOTS:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='y', data=df)
    plt.title('Distribution of Target Variable (y)')
    plt.savefig('eda_target_distribution.png')
    plt.close() # Close plot to prevent inline display clutter
    print("Saved plot: eda_target_distribution.png (Target variable is imbalanced)")

    # Example: Histogram of a numerical feature
    plt.figure(figsize=(8, 5))
    sns.histplot(df['age'], bins=30, kde=True)
    plt.title('Age Distribution')
    plt.savefig('eda_age_distribution.png')
    plt.close()
    print("Saved plot: eda_age_distribution.png")

    # Example: Bar plot of a categorical feature
    plt.figure(figsize=(12, 6))
    sns.countplot(y='job', data=df, order = df['job'].value_counts().index)
    plt.title('Job Distribution')
    plt.tight_layout()
    plt.savefig('eda_job_distribution.png')
    plt.close()
    print("Saved plot: eda_job_distribution.png")

# --- 3. Data Preprocessing ---
print("\n--- 3. Data Preprocessing ---")

# Create a copy to avoid modifying the original dataframe
data_processed = df.copy()

# 3.1. Convert target variable 'y' to binary (1 for 'yes', 0 for 'no')
data_processed['y'] = data_processed['y'].map({'yes': 1, 'no': 0})
print("Converted target 'y' to 1 (yes) and 0 (no).")

# 3.2. Convert binary categorical features to 0/1
binary_cols = ['default', 'housing', 'loan']
for col in binary_cols:
    data_processed[col] = data_processed[col].map({'yes': 1, 'no': 0})
print(f"Converted binary columns {binary_cols} to 1/0.")

# 3.3. Identify numerical and categorical columns for further processing
# 'day' and 'month' are cyclical, but for simplicity, we'll treat 'day' as numerical
# and 'month' as categorical.
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

print(f"\nNumerical columns: {numerical_cols}")
print(f"Categorical columns (to be one-hot encoded): {categorical_cols}")


# --- 4. Train-Test Split (From Scratch) ---
print("\n--- 4. Train-Test Split (From Scratch) ---")

# 4.1. Randomly shuffle the data
# We use .sample(frac=1) to shuffle. random_state ensures reproducibility.
shuffled_df = data_processed.sample(frac=1, random_state=RANDOM_SEED)

# 4.2. Calculate split index for 80:20
split_idx = int(len(shuffled_df) * 0.80)

# 4.3. Perform the split
train_df = shuffled_df.iloc[:split_idx]
test_df = shuffled_df.iloc[split_idx:]

print(f"Total samples: {len(data_processed)}")
print(f"Training samples: {len(train_df)} (80.0%)")
print(f"Test samples: {len(test_df)} (20.0%)")

# 4.4. Separate features (X) and target (y)
y_train = train_df['y']
X_train = train_df.drop('y', axis=1)

y_test = test_df['y']
X_test = test_df.drop('y', axis=1)

# --- 4.5. Preprocessing (continued after split to prevent data leakage) ---

print("\n--- Continuing Preprocessing (post-split) ---")

# 4.5.1. One-Hot Encoding for categorical features
print("Applying one-hot encoding to categorical features...")
X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True)
X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=True)

# Align columns: Ensure test set has same columns as train set
X_train_cat_cols = X_train_cat.columns
X_test_cat = X_test_cat.reindex(columns=X_train_cat_cols, fill_value=0)
print(f"Created {len(X_train_cat_cols)} dummy variables.")

# 4.5.2. Standardization for numerical features (from scratch)
print("Applying standardization (Z-score scaling) to numerical features...")
X_train_num = pd.DataFrame(index=X_train.index)
X_test_num = pd.DataFrame(index=X_test.index)
scaler_stats = {}

for col in numerical_cols:
    # Calculate mean and std *only* from training data
    mean = X_train[col].mean()
    std = X_train[col].std()
    
    # Store stats (e.g., for future use)
    scaler_stats[col] = {'mean': mean, 'std': std}
    
    # Apply transformation to both train and test
    # Handle std=0 (constant feature) by setting scaled value to 0
    if std > 0:
        X_train_num[col] = (X_train[col] - mean) / std
        X_test_num[col] = (X_test[col] - mean) / std
    else:
        X_train_num[col] = 0
        X_test_num[col] = 0

print("Scaling complete.")

# 4.5.3. Combine all processed features
print("Combining all processed features...")
X_train_final = pd.concat([X_train_num, X_train[binary_cols], X_train_cat], axis=1)
X_test_final = pd.concat([X_test_num, X_test[binary_cols], X_test_cat], axis=1)

print("Final feature set shape (Train):", X_train_final.shape)
print("Final feature set shape (Test):", X_test_final.shape)

# 4.5.4. Convert to NumPy arrays for the algorithm
# Our from-scratch model will work with NumPy
X_train_np = X_train_final.values
y_train_np = y_train.values

X_test_np = X_test_final.values
y_test_np = y_test.values

print("Converted data to NumPy arrays.")

# --- 5. Logistic Regression (From Scratch) ---
print("\n--- 5. Logistic Regression (From Scratch) ---")

class LogisticRegressionFromScratch:
    """
    A simple implementation of Logistic Regression using Gradient Descent.
    
    Parameters:
    lr (float): The learning rate.
    n_iters (int): The number of iterations for gradient descent.
    """
    
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """
        Private helper function to compute the sigmoid activation.
        Clips values to avoid overflow in exp().
        """
        z_clipped = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z_clipped))

    def fit(self, X, y):
        """
        Train the logistic regression model using batch gradient descent.
        
        X (np.array): Training features (n_samples, n_features)
        y (np.array): Training target (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        print(f"Starting training for {self.n_iters} iterations...")
        
        # Gradient Descent
        for i in range(self.n_iters):
            # 1. Compute linear model: z = X.w + b
            linear_model = np.dot(X, self.weights) + self.bias
            
            # 2. Apply sigmoid function
            y_predicted_proba = self._sigmoid(linear_model)
            
            # 3. Calculate gradients
            # Gradient of weights
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted_proba - y))
            # Gradient of bias
            db = (1 / n_samples) * np.sum(y_predicted_proba - y)
            
            # 4. Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Optional: Print cost every 100 iterations to check progress
            if i % 100 == 0:
                cost = self._compute_cost(y, y_predicted_proba)
                print(f"Iteration {i}: Cost = {cost:.4f}")

    def _compute_cost(self, y, y_pred_proba):
        """
        Compute the Log Loss (Binary Cross-Entropy) cost.
        Includes epsilon to avoid log(0).
        """
        epsilon = 1e-9
        n_samples = len(y)
        cost = -(1/n_samples) * np.sum(
            y * np.log(y_pred_proba + epsilon) + 
            (1 - y) * np.log(1 - y_pred_proba + epsilon)
        )
        return cost

    def predict_proba(self, X):
        """
        Predict class probabilities for new data.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Predict class labels (0 or 1) based on a threshold.
        """
        probabilities = self.predict_proba(X)
        # Apply threshold to get binary class
        return [1 if i > threshold else 0 for i in probabilities]

print("LogisticRegressionFromScratch class defined.")

# --- 6. Model Training and Evaluation ---
print("\n--- 6. Model Training and Evaluation ---")

# 6.1. Instantiate and train the model
# We use a relatively small learning rate and a good number of iterations.
# These values may need tuning for optimal performance.
model = LogisticRegressionFromScratch(lr=0.1, n_iters=1000)
model.fit(X_train_np, y_train_np)

print("Model training complete.")

# 6.2. Make predictions on the test set
y_pred_np = model.predict(X_test_np)

# 6.3. Evaluation Metrics (From Scratch)
print("\n--- Evaluation Metrics (From Scratch) ---")

def calculate_metrics(y_true, y_pred):
    """
    Calculates Confusion Matrix, Accuracy, Precision, Recall, and F1-Score
    from scratch.
    """
    # Ensure inputs are numpy arrays for easier boolean indexing
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 1. Confusion Matrix
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    print("Confusion Matrix:")
    print(f"           Predicted 0 | Predicted 1")
    print(f"Actual 0:  {TN:^11} | {FP:^11}")
    print(f"Actual 1:  {FN:^11} | {TP:^11}")
    
    # Helper for safe division
    def safe_division(numerator, denominator):
        return numerator / denominator if denominator != 0 else 0
    
    # 2. Accuracy
    total = TP + TN + FP + FN
    accuracy = safe_division(TP + TN, total)
    
    # 3. Precision (How many selected 'yes' were correct?)
    precision = safe_division(TP, TP + FP)
    
    # 4. Recall (How many actual 'yes' did we find?)
    recall = safe_division(TP, TP + FN)
    
    # 5. F1-Score (Harmonic mean of Precision and Recall)
    f1 = safe_division(2 * (precision * recall), (precision + recall))
    
    print("\nCalculated Metrics:")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-Score:   {f1:.4f}")
    
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Calculate and display metrics for our model
metrics = calculate_metrics(y_test_np, y_pred_np)

print("\n--- End of Lab Assignment ---")