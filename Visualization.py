import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Define directories
results_dir = 'results'
figures_dir = 'figures'

# Ensure figures directory exists
os.makedirs(figures_dir, exist_ok=True)

def plot_performance_metrics(results_path):
    results = pd.read_csv(results_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['True'], label='True Values')
    plt.plot(results['Predicted'], label='Predicted Values', alpha=0.7)
    plt.title('True vs Predicted Values')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(os.path.join(figures_dir, 'true_vs_predicted.png'))
    plt.close()
    print("Saved plot: true_vs_predicted.png")

def plot_residuals(results_path):
    results = pd.read_csv(results_path)
    residuals = results['True'] - results['Predicted']
    
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(figures_dir, 'residuals_distribution.png'))
    plt.close()
    print("Saved plot: residuals_distribution.png")

def plot_pca(data_path, title):
    data = pd.read_csv(data_path)
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(components[:, 0], components[:, 1], alpha=0.5)
    plt.title(f'PCA of {title}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(os.path.join(figures_dir, f'pca_{title.lower().replace(" ", "_")}.png'))
    plt.close()
    print(f"Saved plot: pca_{title.lower().replace(' ', '_')}.png")

def main():
    results_path = os.path.join(results_dir, 'evaluation_results.csv')
    payroll_data_path = os.path.join('data/preprocessed', 'payroll_preprocessed.csv')
    sales_data_path = os.path.join('data/preprocessed', 'sales_preprocessed.csv')
    inventory_data_path = os.path.join('data/preprocessed', 'inventory_preprocessed.csv')
    industry_data_path = os.path.join('data/preprocessed', 'bls_industry_data_preprocessed.csv')

    plot_performance_metrics(results_path)
    plot_residuals(results_path)
    plot_pca(payroll_data_path, 'Payroll Data')
    plot_pca(sales_data_path, 'Sales Data')
    plot_pca(inventory_data_path, 'Inventory Data')
    plot_pca(industry_data_path, 'Industry Data')

if __name__ == "__main__":
    main()
