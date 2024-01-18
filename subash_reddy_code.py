# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from lmfit import Model


def read_clean_transpose_csv(csv_file_path):
    """
    Reads data from a CSV file, cleans the data, and returns the original,
    cleaned, and transposed data.

    Parameters:
    - csv_file_path (str): Path to the CSV file.

    Returns:
    - original_data (pd.DataFrame): Original data read from the CSV.
    - cleaned_data (pd.DataFrame): Data after cleaning and imputation.
    - transposed_data (pd.DataFrame): Transposed data.
    """

    # Read the data from the CSV file
    original_data = pd.read_csv(csv_file_path)

    # Replace non-numeric values with NaN
    original_data.replace('..', np.nan, inplace=True)

    # Select relevant columns
    columns_of_interest = [
        "CO2 emissions (metric tons per capita)",
        "CO2 emissions from electricity and heat production, total (% of total fuel combustion)",
        "CO2 emissions from residential buildings and commercial and public services (% of total fuel combustion)",
        "Adjusted net national income (annual % growth)"
    ]

    # Create a SimpleImputer instance with strategy='mean'
    imputer = SimpleImputer(strategy='mean')

    # Apply imputer to fill missing values
    cleaned_data = original_data.copy()
    cleaned_data[columns_of_interest] = imputer.fit_transform(cleaned_data[columns_of_interest])

    # Transpose the data
    transposed_data = cleaned_data.transpose()

    return original_data, cleaned_data, transposed_data


def exponential_growth_model(x, a, b):
    """
    Exponential growth model function.

    Parameters:
    - x (array-like): Input values (time points).
    - a (float): Amplitude parameter.
    - b (float): Growth rate parameter.

    Returns:
    - array-like: Exponential growth model values.
    """
    return a * np.exp(b * np.array(x))


def curve_fit_plot():
    """
    Plot the actual data, fitted curve, and confidence interval.

    Parameters:
    - time_data (array-like): Time points.
    - co2_emissions_data (array-like): Actual CO2 emissions data values.
    - result (lmfit.model.ModelResult): Result of the curve fitting.

    Returns:
    None
    """

    plt.figure(figsize=(12, 8))  # Adjust figure size

    # Line plot for the actual CO2 emissions data
    sns.lineplot(x=time_data, y=co2_emissions_data, label='Actual CO2 Emissions Data', color='blue', linewidth=2)

    # Line plot for the exponential growth fit
    sns.lineplot(x=time_data, y=result.best_fit, label='Exponential Growth Fit', color='orange', linewidth=2)

    # Confidence interval plot
    plt.fill_between(time_data, result.best_fit - result.eval_uncertainty(), result.best_fit + result.eval_uncertainty(),
                     color='orange', alpha=0.2, label='Confidence Interval')

    plt.xlabel('Time')
    plt.ylabel('CO2 emissions (metric tons per capita)')
    plt.ylim(0, 15)  # Set y-axis range to [0, 15]
    plt.title('Curve Fit for CO2 Emissions Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# Main Code

csv_file_path = "477dcb8c-e702-4f36-98ef-84512f1db46a_Data.csv"
original_data, cleaned_data, transposed_data = read_clean_transpose_csv(csv_file_path)

# Normalize the data
scaler = StandardScaler()
columns_of_interest = [
    "CO2 emissions (metric tons per capita)",
    "CO2 emissions from electricity and heat production, total (% of total fuel combustion)",
    "CO2 emissions from residential buildings and commercial and public services (% of total fuel combustion)",
    "Adjusted net national income (annual % growth)"
]
df_normalized = scaler.fit_transform(cleaned_data[columns_of_interest])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cleaned_data['Cluster'] = kmeans.fit_predict(df_normalized)

# Extract cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Assuming df_normalized contains the normalized data used for clustering
silhouette_avg = silhouette_score(df_normalized, cleaned_data['Cluster'])
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Visualize the clusters and cluster centers using seaborn for better aesthetics
plt.figure(figsize=(12, 8))  # Adjust figure size
sns.scatterplot(x="Adjusted net national income (annual % growth)",
                y="CO2 emissions (metric tons per capita)",
                hue="Cluster", palette="viridis", data=cleaned_data, s=80)
sns.scatterplot(x=cluster_centers[:, 0], y=cluster_centers[:, 2],
                marker='X', s=200, color='red', label='Cluster Centers')
plt.title('Clustering of Countries with Cluster Centers')
plt.xlabel('Adjusted Net National Income Growth (%)')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.legend()
plt.show()

# Extract relevant data
time_data = cleaned_data['Time']
co2_emissions_data = cleaned_data['CO2 emissions (metric tons per capita)']

# Create an lmfit Model
model = Model(exponential_growth_model)

# Set initial parameter values
params = model.make_params(a=1, b=0.001)

# Fit the model to the data
result = model.fit(co2_emissions_data, x=time_data, params=params)
curve_fit_plot()

# Generate time points for prediction
future_years = [2024, 2027, 2030]

# Predict values for the future years using the fitted model
predicted_values = result.eval(x=np.array(future_years))

# Display the predicted values
for year, value in zip(future_years, predicted_values):
    print(f"Predicted value for {year} is : {value:.2f}")

# Plot the CO2 emissions for all countries
plt.figure(figsize=(12, 8))
sns.lineplot(x=time_data, y=co2_emissions_data, color='orange', linewidth=2)
plt.xlabel('Time')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.ylim(4, 8)  # Set y-axis range 
plt.title('CO2 Emissions Over Time for All Countries')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
