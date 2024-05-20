import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define directories
input_data_dir = 'data/preprocessed'
output_model_dir = 'models'

# Ensure output directory exists
os.makedirs(output_model_dir, exist_ok=True)

class CompanyDataset(Dataset):
    def __init__(self, payroll_data, sales_data, inventory_data, industry_data, target_data):
        self.payroll_data = payroll_data
        self.sales_data = sales_data
        self.inventory_data = inventory_data
        self.industry_data = industry_data
        self.target_data = target_data

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, idx):
        return {
            'payroll': torch.tensor(self.payroll_data[idx], dtype=torch.float32),
            'sales': torch.tensor(self.sales_data[idx], dtype=torch.float32),
            'inventory': torch.tensor(self.inventory_data[idx], dtype=torch.float32),
            'industry': torch.tensor(self.industry_data[idx], dtype=torch.float32),
            'target': torch.tensor(self.target_data[idx], dtype=torch.float32),
        }

class BusinessOptimizationModel(nn.Module):
    def __init__(self, input_size_payroll, input_size_sales, input_size_inventory, input_size_industry):
        super(BusinessOptimizationModel, self).__init__()
        self.payroll_layer = nn.Linear(input_size_payroll, 64)
        self.sales_layer = nn.Linear(input_size_sales, 64)
        self.inventory_layer = nn.Linear(input_size_inventory, 64)
        self.industry_layer = nn.Linear(input_size_industry, 64)
        
        self.combined_layer1 = nn.Linear(64*4, 128)
        self.combined_layer2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()

    def forward(self, payroll, sales, inventory, industry):
        payroll_out = self.relu(self.payroll_layer(payroll))
        sales_out = self.relu(self.sales_layer(sales))
        inventory_out = self.relu(self.inventory_layer(inventory))
        industry_out = self.relu(self.industry_layer(industry))
        
        combined = torch.cat((payroll_out, sales_out, inventory_out, industry_out), dim=1)
        combined_out = self.relu(self.combined_layer1(combined))
        combined_out = self.relu(self.combined_layer2(combined_out))
        output = self.output_layer(combined_out)
        return output

def load_data():
    payroll_df = pd.read_csv(os.path.join(input_data_dir, 'payroll_preprocessed.csv'))
    sales_df = pd.read_csv(os.path.join(input_data_dir, 'sales_preprocessed.csv'))
    inventory_df = pd.read_csv(os.path.join(input_data_dir, 'inventory_preprocessed.csv'))
    industry_df = pd.read_csv(os.path.join(input_data_dir, 'bls_industry_data_preprocessed.csv'))

    # Assuming 'target' is the column to predict
    target = payroll_df['target'].values
    payroll_df.drop(columns=['target'], inplace=True)

    return payroll_df.values, sales_df.values, inventory_df.values, industry_df.values, target

def main():
    # Load data
    payroll_data, sales_data, inventory_data, industry_data, target = load_data()
    
    # Split data
    train_payroll, test_payroll, train_sales, test_sales, train_inventory, test_inventory, train_industry, test_industry, train_target, test_target = train_test_split(
        payroll_data, sales_data, inventory_data, industry_data, target, test_size=0.2, random_state=42
    )

    # Standardize data
    scaler_payroll = StandardScaler()
    scaler_sales = StandardScaler()
    scaler_inventory = StandardScaler()
    scaler_industry = StandardScaler()

    train_payroll = scaler_payroll.fit_transform(train_payroll)
    test_payroll = scaler_payroll.transform(test_payroll)
    train_sales = scaler_sales.fit_transform(train_sales)
    test_sales = scaler_sales.transform(test_sales)
    train_inventory = scaler_inventory.fit_transform(train_inventory)
    test_inventory = scaler_inventory.transform(test_inventory)
    train_industry = scaler_industry.fit_transform(train_industry)
    test_industry = scaler_industry.transform(test_industry)

    # Create datasets and dataloaders
    train_dataset = CompanyDataset(train_payroll, train_sales, train_inventory, train_industry, train_target)
    test_dataset = CompanyDataset(test_payroll, test_sales, test_inventory, test_industry, test_target)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define model
    input_size_payroll = train_payroll.shape[1]
    input_size_sales = train_sales.shape[1]
    input_size_inventory = train_inventory.shape[1]
    input_size_industry = train_industry.shape[1]

    model = BusinessOptimizationModel(input_size_payroll, input_size_sales, input_size_inventory, input_size_industry)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            payroll = batch['payroll']
            sales = batch['sales']
            inventory = batch['inventory']
            industry = batch['industry']
            target = batch['target']

            optimizer.zero_grad()
            outputs = model(payroll, sales, inventory, industry)
            loss = criterion(outputs, target.view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    # Save model
    model_path = os.path.join(output_model_dir, 'business_optimization_model.pth')
    torch.save(model.state_dict(), model_path)
    print("Model saved.")

    # Evaluate model
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            payroll = batch['payroll']
            sales = batch['sales']
            inventory = batch['inventory']
            industry = batch['industry']
            target = batch['target']
            outputs = model(payroll, sales, inventory, industry)
            loss = criterion(outputs, target.view(-1, 1))
            test_loss += loss.item()

    print(f"Test Loss: {test_loss/len(test_loader)}")

if __name__ == "__main__":
    main()
