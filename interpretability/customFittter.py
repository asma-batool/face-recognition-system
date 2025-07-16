"""
A custom module for fitting statistical distributions to feature data based on
a unique set of performance-driven and statistical criteria.

The idea behind the execution:
-------------------------------
REVISED LOGIC:

1. Known Distribution Types:
   The fitting process considers two main families of distributions.

   a. Standard Continuous Distributions:
      This category now includes Normal, Student's t, Chi-squared, Gamma,
      and Exponential distributions. This broader set increases the likelihood
      of finding a good fit for various data shapes.

   b. Polynomial Approximation (Non-Normal):
      This method fits a polynomial to the data's Empirical Cumulative
      Distribution Function (ECDF) using Ordinary Least Squares (OLS).
      Instead of a flawed PGF analogy, the fit's quality is now judged
      by its R-squared score, a standard measure of regression fit.

2. Iterative Fitting and Stopping Conditions:
   The core of this module is an iterative fitting process with robust
   stopping conditions to find an optimal data transformation.

   - For Standard Distributions:
     The process uses the K-S test p-value for a statistical goodness-of-fit
     and a custom "Zaid Score" (0.5*Precision + 0.5*F1) to measure the
     feature's utility for a downstream classification task.

   - For Polynomials:
     The fitting process stops when the improvement in the R-squared score
     between successive polynomial degrees becomes marginal (e.g., < 1%).
     This prevents overfitting while finding a sufficiently complex model.

3. Expected Output:
   The module provides clear, visual feedback, plotting the best-fitting
   distributions and polynomial models with relevant metrics (K-S p-value,
   Zaid Score, R-squared) to allow for an informed, human-in-the-loop
   evaluation of the best data transformation.
"""


# distribution_fitter.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, f1_score, r2_score
from collections import deque
import warnings

# Suppress warnings from fitting that may not converge
warnings.filterwarnings('ignore', category=RuntimeWarning)


def calculate_zaid_score(y_true, y_pred, class_labels):
    """Calculates the Zaid Score: 50% Precision + 50% F1-Score."""
    # Use 'macro' average to handle class imbalance
    precision = precision_score(
        y_true, y_pred, labels=class_labels, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=class_labels,
                  average='macro', zero_division=0)
    score = 0.5 * precision + 0.5 * f1
    return score, precision, f1


# --- Main Analysis Class ---
class DistributionFitter:
    def __init__(self, X_train, y_train, labels):
        self.X_train = X_train
        self.y_train = y_train
        self.labels = labels
        # Simple classifier for Zaid Score
        self.knn = KNeighborsClassifier(n_neighbors=5)
        print("✅ DistributionFitter Initialized.")
        print(f"Training data shape: {self.X_train.shape}")

    def _get_zaid_score_for_feature(self, feature_data):
        """Trains a KNN on a single feature and returns the Zaid Score."""
        # Reshape for sklearn
        feature_data_reshaped = feature_data.reshape(-1, 1)
        self.knn.fit(feature_data_reshaped, self.y_train)
        y_pred = self.knn.predict(feature_data_reshaped)
        class_labels = np.unique(self.y_train)
        return calculate_zaid_score(self.y_train, y_pred, class_labels)

    def plot_fit(self, data, dist, params, title, ax):
        """Helper function to plot data histogram against a fitted distribution PDF."""
        ax.hist(data, bins=50, density=True, alpha=0.7,
                color='g', label='Data Histogram')
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 200)

        # Ensure pdf is calculated correctly even if dist is a frozen distribution
        if isinstance(dist, stats.rv_continuous):
            pdf = dist.pdf(x, **params)
        else:  # Handle case where dist might be a frozen instance
            pdf = dist.pdf(x)

        ax.plot(x, pdf, 'k', linewidth=2, label='Fitted PDF')
        ax.set_title(title)
        ax.legend()

    def fit_standard_distributions(self, feature_data, feature_index):
        """
        Task 1: Fit standard distributions and check statistical goodness-of-fit.
        """
        print(
            f"\n--- Fitting Standard Distributions to Feature {feature_index} ---")

        # A wider range of common continuous distributions
        distributions_to_try = [
            ('Normal', stats.norm),
            ('Student\'s T', stats.t),
            ('Chi-Squared', stats.chi2),
            ('Gamma', stats.gamma),
            ('Exponential', stats.expon)
        ]

        results = []

        for name, dist in distributions_to_try:
            try:
                # Fit the distribution to the data. `fit` returns the distribution parameters (e.g., df, loc, scale)
                params = dist.fit(feature_data)

                # K-S Test for goodness of fit
                ks_stat, p_value = stats.kstest(
                    feature_data, dist.name, args=params)

                # Use the fitted parameters to create a "frozen" distribution for CDF transformation
                frozen_dist = dist(*params)
                transformed_feature = frozen_dist.cdf(feature_data)
                zaid_score, precision, f1 = self._get_zaid_score_for_feature(
                    transformed_feature)

                print(
                    f"Testing: {name}... K-S p-value: {p_value:.4f}, Zaid Score: {zaid_score:.4f}")

                # Save result if K-S test is meaningful (p > 0.01)
                if p_value > 0.01:
                    results.append({
                        'name': name,
                        'dist': frozen_dist,  # Store the frozen distribution
                        'params': params,
                        'p_value': p_value,
                        'zaid_score': zaid_score,
                        'data': feature_data
                    })

            except Exception as e:
                # Some fits may fail to converge, which is expected
                print(f"Skipping {name} due to error: {e}")
                continue

        print("\n--- Final Results for Standard Distributions ---")
        # Sort by the K-S p-value in descending order (best fits first)
        results.sort(key=lambda x: x['p_value'], reverse=True)

        if not results:
            print("❌ No standard distribution passed the K-S test threshold (p > 0.01).")
            return

        # Display the top 3 best fits
        best_results = results[:3]
        print(
            f"Displaying the best {len(best_results)} fit(s) based on K-S p-value.")
        fig, axes = plt.subplots(1, len(best_results),
                                 figsize=(15, 5), squeeze=False)
        for i, res in enumerate(best_results):
            title = f"{res['name']}\nZaid Score: {res['zaid_score']:.3f}, K-S p-val: {res['p_value']:.3f}"
            self.plot_fit(res['data'], res['dist'], {}, title,
                          axes[0, i])  # Pass empty params dict
        fig.suptitle(
            f'Best Standard Distribution Fits for Feature {feature_index}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def fit_polynomial(self, feature_data, feature_index):
        """
        Task 2: Fit polynomial to the ECDF using OLS and stop when R-squared plateaus.
        """
        print(
            f"\n--- Fitting Polynomial Distribution to Feature {feature_index} ---")

        # Create empirical CDF (ECDF) which is the target for OLS
        data_sorted = np.sort(feature_data)
        y_ecdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)

        last_3_results = deque(maxlen=3)
        prev_r_squared = -np.inf

        for degree in range(1, 10):
            print(f"\n-- Testing Polynomial Degree: {degree} --")

            # 1. Fit OLS model to the ECDF
            poly_features = PolynomialFeatures(
                degree=degree, include_bias=False)
            X_poly = poly_features.fit_transform(data_sorted.reshape(-1, 1))

            model = LinearRegression()
            model.fit(X_poly, y_ecdf)

            # 2. Evaluate the fit using R-squared
            y_pred = model.predict(X_poly)
            r_squared = r2_score(y_ecdf, y_pred)
            print(f"  R-squared: {r_squared:.4f}")

            # Store results
            result = {
                'degree': degree, 'model': model, 'poly_features': poly_features,
                'r_squared': r_squared, 'data_sorted': data_sorted,
                'y_ecdf': y_ecdf, 'y_pred': y_pred
            }
            last_3_results.append(result)

            # 3. Check Stopping Condition: Stop when R-squared improvement is marginal
            r_squared_improvement = r_squared - prev_r_squared
            print(f"  Improvement in R-squared: {r_squared_improvement:.4f}")
            if r_squared_improvement < 0.01 and degree > 1:
                print(
                    "✅ STOPPING CONDITION MET: R-squared improvement is marginal. Returning previous results.")
                # Pop the last result as we want the one *before* the marginal improvement
                last_3_results.pop()
                break

            prev_r_squared = r_squared

        print("\n--- Final Results for Polynomial Fitting ---")
        print("Displaying the last (up to) 3 fits before stopping.")

        if not last_3_results:
            print("❌ No polynomial fits were generated.")
            return

        fig, axes = plt.subplots(1, len(last_3_results), figsize=(
            15, 5), squeeze=False, sharey=True)
        for i, res in enumerate(last_3_results):
            ax = axes[0, i]
            ax.plot(res['data_sorted'], res['y_ecdf'], 'o',
                    label='Empirical CDF', markersize=4, alpha=0.6)
            ax.plot(res['data_sorted'], res['y_pred'], color='r',
                    linewidth=2, label='Fitted Polynomial')
            title = f"Degree {res['degree']}\nR-squared = {res['r_squared']:.3f}"
            ax.set_title(title)
            ax.set_xlabel("Feature Value")
            ax.legend()
        axes[0, 0].set_ylabel("Cumulative Probability")

        fig.suptitle(
            f'Best Polynomial Fits for Feature {feature_index}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


# --- Main execution block ---
if __name__ == "__main__":
    try:
        # Load data from previous steps
        X_train_lbp = np.load("X_train_lbp_features.npy")
        y_train = np.load("y_train.npy")
        labels = np.load("labels.npy")
    except FileNotFoundError:
        print("❌ Error: Make sure 'X_train_lbp_features.npy', 'y_train.npy', and 'labels.npy' are present.")
        print("Run featureExtraction.py first.")
        exit()

    # LBP creates 256 features. We will analyze one of them as a sample.
    FEATURE_INDEX_TO_ANALYZE = 128

    if X_train_lbp.shape[1] <= FEATURE_INDEX_TO_ANALYZE:
        print(
            f"❌ Error: The selected feature index ({FEATURE_INDEX_TO_ANALYZE}) is out of bounds.")
        print(
            f"Please choose an index between 0 and {X_train_lbp.shape[1] - 1}.")
        exit()

    selected_feature_vector = X_train_lbp[:,
                                          FEATURE_INDEX_TO_ANALYZE].astype(float)
    # A small amount of jitter can help break ties and prevent issues with some distribution fits
    if np.std(selected_feature_vector) > 0:
        selected_feature_vector += np.random.normal(
            0, 1e-6, selected_feature_vector.shape)

    # Initialize and run the fitter
    fitter = DistributionFitter(X_train_lbp, y_train, labels)

    # Run Task 1: Fit Standard Distributions
    fitter.fit_standard_distributions(
        selected_feature_vector, FEATURE_INDEX_TO_ANALYZE)

    # Run Task 2: Fit Polynomial
    fitter.fit_polynomial(selected_feature_vector, FEATURE_INDEX_TO_ANALYZE)
