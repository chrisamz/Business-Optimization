import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from modeling import BusinessOptimizationModel, CompanyDataset

# Define directories
input_data_dir = 'data/preprocessed'
output_model_dir = 'models'
results_dir = 'results'

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

def load_and_preprocess_data():
    payroll_df = pd.read_csv(os.path.join(input_data_dir, 'payroll_preprocessed.csv'))
    sales_df = pd.read_csv(os.path.join(input_data_dir, 'sales_preprocessed.csv'))
    inventory_df = pd.read_csv(os.path.join(input_data_dir, 'inventory_preprocessed.csv'))
    industry_df = pd.read_csv(os.path.join(input_data_dir, 'bls_industry_data_preprocessed.csv'))

    # Assuming 'target' is the column to predict
    target = payroll_df['target'].values
    payroll_df.drop(columns=['target'], inplace=True)

    return payroll_df.values, sales_df.values, inventory_df.values, industry_df.values, target

def normalize_and_vectorize(train_data, test_data):
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data, scaler

def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            payroll = batch['payroll']
            sales = batch['sales']
            inventory = batch['inventory']
            industry = batch['industry']
            target = batch['target']
            
            outputs = model(payroll, sales, inventory, industry)
            y_true.extend(target.numpy())
            y_pred.extend(outputs.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, r2, y_true, y_pred

def main():
    # Load data
    payroll_data, sales_data, inventory_data, industry_data, target = load_and_preprocess_data()
    
    # Split data
    train_size = int(0.8 * len(target))
    test_size = len(target) - train_size
    indices = np.arange(len(target))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_payroll, test_payroll = payroll_data[train_indices], payroll_data[test_indices]
    train_sales, test_sales = sales_data[train_indices], sales_data[test_indices]
    train_inventory, test_inventory = inventory_data[train_indices], inventory_data[test_indices]
    train_industry, test_industry = industry_data[train_indices], industry_data[test_indices]
    train_target, test_target = target[train_indices], target[test_indices]
    
    # Normalize data
    train_payroll, test_payroll, payroll_scaler = normalize_and_vectorize(train_payroll, test_payroll)
    train_sales, test_sales, sales_scaler = normalize_and_vectorize(train_sales, test_sales)
    train_inventory, test_inventory, inventory_scaler = normalize_and_vectorize(train_inventory, test_inventory)
    train_industry, test_industry, industry_scaler = normalize_and_vectorize(train_industry, test_industry)
    
    # Create datasets and dataloaders
    train_dataset = CompanyDataset(train_payroll, train_sales, train_inventory, train_industry, train_target)
    test_dataset = CompanyDataset(test_payroll, test_sales, test_inventory, test_industry, test_target)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load model
    input_size_payroll = train_payroll.shape[1]
    input_size_sales = train_sales.shape[1]
    input_size_inventory = train_inventory.shape[1]
    input_size_industry = train_industry.shape[1]

    model = BusinessOptimizationModel(input_size_payroll, input_size_sales, input_size_inventory, input_size_industry)
    model_path = os.path.join(output_model_dir, 'business_optimization_model.pth')
    model.load_state_dict(torch.load(model_path))
    
    # Evaluate model
    mse, r2, y_true, y_pred = evaluate_model(model, test_loader)
    
    # Print evaluation results
    print(f'Mean Squared Error: {mse}')
    print(f'R2 Score: {r2}')
    
    # Save evaluation results
    results_path = os.path.join(results_dir, 'evaluation_results.csv')
    pd.DataFrame({'True': y_true, 'Predicted': y_pred}).to_csv(results_path, index=False)
    print(f'Results saved to {results_path}')

if __name__ == "__main__":
    main()
