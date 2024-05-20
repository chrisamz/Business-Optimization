import pandas as pd
import os

# Define directories
company_data_dir = 'data/company'
industry_data_dir = 'data/industry'
output_data_dir = 'data/processed'

# Ensure output directory exists
os.makedirs(output_data_dir, exist_ok=True)

def collect_company_data():
    """
    Collects and merges company data from CSV files.
    """
    payroll_df = pd.read_csv(os.path.join(company_data_dir, 'payroll.csv'))
    sales_df = pd.read_csv(os.path.join(company_data_dir, 'sales.csv'))
    inventory_df = pd.read_csv(os.path.join(company_data_dir, 'inventory.csv'))
    
    # Merge data if necessary
    # For example, merge sales data with payroll data on 'employee_id'
    # merged_df = sales_df.merge(payroll_df, on='employee_id')
    
    # Save the processed company data
    payroll_df.to_csv(os.path.join(output_data_dir, 'payroll_processed.csv'), index=False)
    sales_df.to_csv(os.path.join(output_data_dir, 'sales_processed.csv'), index=False)
    inventory_df.to_csv(os.path.join(output_data_dir, 'inventory_processed.csv'), index=False)

    print("Company data collected and saved.")

def collect_industry_data():
    """
    Collects and processes industry data from CSV files.
    """
    bls_industry_df = pd.read_csv(os.path.join(industry_data_dir, 'bls_industry_data.csv'))
    
    # Process data if necessary
    # For example, filter specific columns or rows based on conditions
    # bls_industry_df = bls_industry_df[['industry', 'employment', 'wage']]

    # Save the processed industry data
    bls_industry_df.to_csv(os.path.join(output_data_dir, 'bls_industry_data_processed.csv'), index=False)

    print("Industry data collected and saved.")

if __name__ == "__main__":
    collect_company_data()
    collect_industry_data()
    print("Data collection completed.")
