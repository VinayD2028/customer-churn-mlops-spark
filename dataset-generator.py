"""
dataset-generator.py
=====================
Synthetic telecom customer churn dataset generator.

This script programmatically creates a realistic 1,000-record customer dataset
that mimics the patterns observed in real-world telecom churn data. The dataset
is designed with deliberate statistical structure to make it suitable for
supervised ML classification tasks.

Design choices:
  - Churn probability is conditionally dependent on tenure, charges, senior
    citizen status, and internet service type — reflecting real business dynamics
  - Stochastic noise (random.random() < 0.6) is applied to prevent the dataset
    from being perfectly separable, which would make ML too easy and unrealistic
  - ~5% of TotalCharges values are set to NaN to simulate real-world data quality
    issues and test the pipeline's missing value handling capabilities

Output:
  customer_churn.csv — saved to the project root directory

Author: Vinay Devabhaktuni
Tech:   pandas, NumPy, Python random
Run:    python3 dataset-generator.py
"""

import pandas as pd
import numpy as np
import random


def generate_customer_churn_data(num_customers=1000):
    """
    Generate a synthetic telecom customer churn dataset.

    The dataset simulates customer behaviour in a telecom company with
    realistic correlations between features and the churn outcome:

    - Short-tenure customers with high monthly charges churn more frequently
      (reflects dissatisfaction in early subscription periods)
    - Senior citizens have a higher churn probability regardless of other factors
      (reflects lower tech adoption and price sensitivity)
    - Fiber optic subscribers churn at higher rates
      (reflects higher price point and competition in premium internet services)
    - Random noise is layered on all churn conditions to prevent the model from
      achieving near-perfect AUC, making it a realistic training challenge

    Args:
        num_customers (int): Number of customer records to generate. Default: 1000

    Returns:
        pd.DataFrame: Synthetic customer dataset with ~5% NaN in TotalCharges
    """

    # --- Customer IDs ---
    # Zero-padded IDs (e.g., CUST00001) for realistic formatting.
    # This column is dropped in the ML pipeline as it carries no predictive signal.
    customerID = [f"CUST{str(i).zfill(5)}" for i in range(1, num_customers + 1)]

    # --- Gender ---
    # Randomly assigned with equal probability.
    # Gender is included as a demographic feature but has low predictive power for churn.
    gender = [random.choice(["Male", "Female"]) for _ in range(num_customers)]

    # --- Senior Citizen flag ---
    # 20% of customers are senior citizens (weights=[0.8, 0.2]).
    # Senior citizens are assigned a higher churn probability in the churn logic below,
    # reflecting real-world patterns of price sensitivity and lower service satisfaction.
    senior_citizen = [random.choices([0, 1], weights=[0.8, 0.2])[0] for _ in range(num_customers)]

    # --- Tenure (months) ---
    # Bimodal distribution: 40% of customers are new (0-12 months),
    # 60% are long-term customers (12-72 months).
    # New customers are at higher churn risk; long-term customers are more loyal.
    tenure = [random.randint(0, 12) if random.random() < 0.4 else random.randint(12, 72) for _ in range(num_customers)]

    # --- Monthly Charges ---
    # New customers (tenure < 12) are charged more on average ($60-$110),
    # reflecting promotional or premium plans that may cause sticker shock.
    # Long-term customers pay $20-$100, reflecting loyalty discounts and plan diversity.
    monthly_charges = [round(random.uniform(60, 110) if tenure[i] < 12 else random.uniform(20, 100), 2) for i in range(num_customers)]

    # --- Total Charges ---
    # Approximated as monthly_charges * tenure plus Gaussian-like noise.
    # Noise is proportional to tenure (longer customers have more billing variance).
    # New customers (tenure=0) get NaN since they haven't been billed yet.
    total_charges = [
        round(monthly_charges[i] * tenure[i] + random.uniform(-10, 10) * tenure[i] if tenure[i] > 0 else np.nan, 2)
        for i in range(num_customers)
    ]

    # --- Phone Service ---
    # Randomly assigned Yes/No with equal probability.
    # Phone service status is included as a binary service feature.
    phone_service = [random.choice(["Yes", "No"]) for _ in range(num_customers)]

    # --- Internet Service ---
    # DSL: 40%, Fiber optic: 40%, No service: 20%.
    # Fiber optic subscribers are assigned higher churn probability below,
    # reflecting premium pricing and competitive alternatives.
    internet_service = [
        random.choices(["DSL", "Fiber optic", "No"], weights=[0.4, 0.4, 0.2])[0] for _ in range(num_customers)
    ]

    # --- Churn Label (target variable) ---
    # Churn is assigned "Yes" when ANY of the following risk conditions are met:
    #   1. New customer (tenure < 12) with high charges (> $70): price shock churn
    #   2. Senior citizen with 40% individual probability: demographic churn risk
    #   3. Fiber optic subscriber with 50% individual probability: service churn risk
    #
    # Even when a risk condition is met, only 60% of those customers actually churn
    # (random.random() < 0.6). This models real-world uncertainty where not every
    # at-risk customer follows through — preventing the dataset from being
    # perfectly deterministic, which would make ML classification too easy.
    churn = [
        "Yes" if (
            (tenure[i] < 12 and monthly_charges[i] > 70) or
            (senior_citizen[i] == 1 and random.random() < 0.4) or
            (internet_service[i] == "Fiber optic" and random.random() < 0.5)
        ) and random.random() < 0.6  # Apply global noise factor to all churn conditions
        else "No"
        for i in range(num_customers)
    ]

    # --- Assemble into DataFrame ---
    # Column order matches the schema expected by customer-churn-analysis.py
    data = {
        "customerID": customerID,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "tenure": tenure,
        "PhoneService": phone_service,
        "InternetService": internet_service,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn
    }
    df = pd.DataFrame(data)

    # --- Inject additional missing values in TotalCharges ---
    # Randomly set 5% of TotalCharges to NaN (on top of the tenure=0 NaNs above).
    # This simulates real-world billing data gaps (e.g., failed payment records)
    # and ensures the preprocessing pipeline's null handling is exercised.
    nan_indices = np.random.choice(df.index, size=int(num_customers * 0.05), replace=False)
    df.loc[nan_indices, 'TotalCharges'] = np.nan

    return df


# --- Main execution ---
# Generate the dataset and save to CSV in the project root.
# The output file (customer_churn.csv) is the input for customer-churn-analysis.py.
df = generate_customer_churn_data(num_customers=1000)
df.to_csv("./customer_churn.csv", index=False)
