import os
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from datetime import datetime

# Define directories
results_dir = 'results'
figures_dir = 'figures'
templates_dir = 'templates'
reports_dir = 'reports'

# Ensure reports directory exists
os.makedirs(reports_dir, exist_ok=True)

def load_results(results_path):
    return pd.read_csv(results_path)

def render_report(template_name, context, output_path):
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template(template_name)
    html_out = template.render(context)
    
    HTML(string=html_out).write_pdf(output_path)
    print(f"Report saved to: {output_path}")

def generate_report():
    # Load results
    results_path = os.path.join(results_dir, 'evaluation_results.csv')
    results = load_results(results_path)

    # Define context for the report
    context = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'true_vs_predicted_path': os.path.join(figures_dir, 'true_vs_predicted.png'),
        'residuals_distribution_path': os.path.join(figures_dir, 'residuals_distribution.png'),
        'pca_payroll_path': os.path.join(figures_dir, 'pca_payroll_data.png'),
        'pca_sales_path': os.path.join(figures_dir, 'pca_sales_data.png'),
        'pca_inventory_path': os.path.join(figures_dir, 'pca_inventory_data.png'),
        'pca_industry_path': os.path.join(figures_dir, 'pca_industry_data.png'),
        'model_performance': {
            'MSE': results['MSE'].iloc[-1],
            'RMSE': results['RMSE'].iloc[-1],
            'MAE': results['MAE'].iloc[-1],
            'R2': results['R2'].iloc[-1]
        },
        'summary': "The model shows strong performance in predicting business process outcomes based on the provided data. The visualizations and residuals analysis indicate good model fit with some room for improvement."
    }

    # Render and save the report
    report_template = 'report_template.html'
    output_path = os.path.join(reports_dir, f'report_{context["date"]}.pdf')
    render_report(report_template, context, output_path)

if __name__ == "__main__":
    generate_report()
