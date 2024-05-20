import pandas as pd
import os

# Define directories
input_data_dir = 'data/processed'
output_data_dir = 'data/preprocessed'

# Ensure output directory exists
os.makedirs(output_data_dir, exist_ok=True)

def preprocess_payroll_data(df):
    """
    Preprocess the payroll data.
    """
    # Example preprocessing steps:
    # - Handle missing values
    # - Convert data types
    # - Create new features if needed

    df.fillna(0, inplace=True)  # Fill missing values with 0
    df['employee_id'] = df['employee_id'].astype(int)
    df['salary'] = df['salary'].astype(float)

    # Add more preprocessing steps as needed
    return df

def preprocess_sales_data(df):
    """
    Preprocess the sales data.
    """
    # Example preprocessing steps:
    # - Handle missing values
    # - Convert data types
    # - Create new features if needed

    df.fillna(0, inplace=True)  # Fill missing values with 0
    df['sale_id'] = df['sale_id'].astype(int)
    df['sale_amount'] = df['sale_amount'].astype(float)

    # Add more preprocessing steps as needed
    return df

def preprocess_inventory_data(df):
    """
    Preprocess the inventory data.
    """
    # Example preprocessing steps:
    # - Handle missing values
    # - Convert data types
    # - Create new features if needed

    df.fillna(0, inplace=True)  # Fill missing values with 0
    df['item_id'] = df['item_id'].astype(int)
    df['quantity'] = df['quantity'].astype(int)

    # Add more preprocessing steps as needed
    return df

def preprocess_industry_data(df):
    """
    Preprocess the industry data.
    """
    # Example preprocessing steps:
    # - Handle missing values
    # - Convert data types
    # - Filter specific columns or rows

    df.fillna(0, inplace=True)  # Fill missing values with 0
    df['industry'] = df['industry'].astype(str)
    df['employment'] = df['employment'].astype(int)
    df['wage'] = df['wage'].astype(float)

    # Add more preprocessing steps as needed
    return df

def preprocess_data():
    """
    Main function to preprocess all data.
    """
    # Load datasets
    payroll_df = pd.read_csv(os.path.join(input_data_dir, 'payroll_processed.csv'))
    sales_df = pd.read_csv(os.path.join(input_data_dir, 'sales_processed.csv'))
    inventory_df = pd.read_csv(os.path.join(input_data_dir, 'inventory_processed.csv'))
    bls_industry_df = pd.read_csv(os.path.join(input_data_dir, 'bls_industry_data_processed.csv'))

    # Preprocess datasets
    payroll_df = preprocess_payroll_data(payroll_df)
    sales_df = preprocess_sales_data(sales_df)
    inventory_df = preprocess_inventory_data(inventory_df)
    bls_industry_df = preprocess_industry_data(bls_industry_df)

    # Save preprocessed datasets
    payroll_df.to_csv(os.path.join(output_data_dir, 'payroll_preprocessed.csv'), index=False)
    sales_df.to_csv(os.path.join(output_data_dir, 'sales_preprocessed.csv'), index=False)
    inventory_df.to_csv(os.path.join(output_data_dir, 'inventory_preprocessed.csv'), index=False)
    bls_industry_df.to_csv(os.path.join(output_data_dir, 'bls_industry_data_preprocessed.csv'), index=False)

    print("Data preprocessing completed.")

if __name__ == "__main__":
    preprocess_data()
