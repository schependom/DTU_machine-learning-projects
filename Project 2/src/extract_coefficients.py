import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def generate_latex_coefficients(
    pipeline: Pipeline, label_encoder: LabelEncoder, feature_names: list
):
    """
    Extracts coefficients from a trained LogisticRegression pipeline
    and prints LaTeX tables for both scaled and unscaled coefficients.

    Args:
        pipeline: The trained scikit-learn Pipeline object.
        label_encoder: The fitted LabelEncoder object used on the target.
        feature_names: A list of the feature names in the order they
                       appear in the ColumnTransformer.
    """

    print("--- Extracting Coefficients ---")

    # 1. --- Get model components ---
    try:
        # Access the logistic regression model step
        logreg = pipeline.named_steps["logreg"]
        # Access the preprocessor step
        preprocessor = pipeline.named_steps["preprocessor"]
        # Access the 'num' transformer within the preprocessor
        # This assumes your ColumnTransformer's numerical transformer is named 'num'
        scaler = preprocessor.named_transformers_["num"]
    except KeyError as e:
        print(f"Error: Could not find a pipeline step or transformer.")
        print(f"Make sure your steps are named 'logreg' and 'preprocessor',")
        print(f"and the transformer in ColumnTransformer is named 'num'.")
        print(f"Details: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # 2. --- Get coefficients, names, and scaler stats ---
    coefs = logreg.coef_
    intercepts = logreg.intercept_
    class_names = label_encoder.classes_

    if len(feature_names) != coefs.shape[1]:
        print(
            f"Error: Number of feature names ({len(feature_names)}) does not match number of coefficients ({coefs.shape[1]})."
        )
        print("Expected feature names:", feature_names)
        return

    # Get the mean and scale (std dev) from the fitted StandardScaler
    std_devs = scaler.scale_
    means = scaler.mean_

    # 3. --- Create Table 1: Standardized Coefficients ---
    # These coefficients relate to a 1-standard-deviation change in the input
    df_scaled = pd.DataFrame(
        np.vstack([intercepts, coefs.T]),
        index=["Intercept"] + feature_names,
        columns=class_names,
    )

    # 4. --- Create Table 2: Unscaled Coefficients ---
    # These coefficients relate to a 1-unit change in the *original* input data
    unscaled_coefs = coefs / std_devs
    unscaled_intercept = intercepts - np.dot(coefs, means / std_devs)

    df_unscaled = pd.DataFrame(
        np.vstack([unscaled_intercept, unscaled_coefs.T]),
        index=["Intercept"] + feature_names,
        columns=class_names,
    )

    # 5. --- Print LaTeX tables ---
    print("\n" + "=" * 80)
    print("--- LaTeX Table 1: Standardized Coefficients (for 1-std-dev change) ---")
    print("--- Use this to compare relative feature importance. ---")
    print("=" * 80)
    # .to_latex() requires the 'booktabs' LaTeX package
    print(df_scaled.to_latex(float_format="%.4f", escape=True))
    print("\n" * 3)

    print("=" * 80)
    print("--- LaTeX Table 2: Unscaled Coefficients (for 1-unit change) ---")
    print("--- Use this for interpretation on the original feature scale. ---")
    print("=" * 80)
    # .to_latex() requires the 'booktabs' LaTeX package
    print(df_unscaled.to_latex(float_format="%.4f", escape=True))
