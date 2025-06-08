import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def load_scores(directory):
    scores = []
    for va_type in ['with_VA', 'without_VA']:
        for method in ['baseline', 'length_pruned', 'summary', 'va_pruned']:
            for n in ['n2', 'n4', 'n6', 'n8']:
                path = os.path.join(directory, va_type, method, n)
                file_name = None
                # Adjust file name based on VA type and method
                if method == 'baseline':
                    if va_type == 'with_VA':
                        file_name = f'nohup__{n}.out'
                    else:  # without_VA
                        file_name = f'nohup_--without_va_{n}.out'
                else:
                    for f in os.listdir(path):
                        if n in f and method in f:
                            file_name = f
                            break
                
                if file_name:
                    file_path = os.path.join(path, file_name)
                else:
                    continue
                
                if os.path.exists(file_path):
                    # Read MAE and RMSE from the output file
                    with open(file_path, 'r') as f:
                        content = f.read()
                        # Use regex to extract MAE and RMSE based on the reference format
                        mae_match = re.search(r'MAE: ([\d.]+)', content)
                        rmse_match = re.search(r'RMSE: ([\d.]+)', content)
                        
                        mae = float(mae_match.group(1)) if mae_match else None
                        rmse = float(rmse_match.group(1)) if rmse_match else None
                        
                        scores.append({
                            'VA_Type': va_type,
                            'Method': method,
                            'N': n,
                            'MAE': mae,
                            'RMSE': rmse
                        })
    # Create DataFrame and save to CSV after collecting all scores
    print(f"scores: {scores}")
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv('/Users/kevinawang/Documents/GitHub/VA-classifier/experiments/experiment_scores.csv', index=False)
    return pd.DataFrame(scores)

def plot_metrics(df, metric):
    # Define color palette for VA types
    va_colors = {'with_VA': '#00008b', 'without_VA': '#add8e6'}  # Dark blue for with VA, light blue for without VA
    
    # Create subplots for each method
    methods = ['baseline', 'length_pruned', 'summary', 'va_pruned']
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    # Calculate global y-axis limits for consistent scaling
    y_min = df[['MAE', 'RMSE']].min().min() * 0.9  # 10% padding below
    y_max = df[['MAE', 'RMSE']].max().max() * 1.1  # 10% padding above
    
    for idx, method in enumerate(methods):
        method_data = df[df['Method'] == method]
        
        # Create grouped bar plot
        g = sns.barplot(data=method_data, 
                        x='N', 
                        y=metric,
                        hue='VA_Type',
                        order=['n2', 'n4', 'n6', 'n8'],  # Ensure N values are ordered
                        hue_order=['with_VA', 'without_VA'],  # Ensure VA types are ordered
                        palette=va_colors,  # Use VA colors for distinction
                        ax=axes[idx])
        
        # Set titles and labels
        axes[idx].set_title(f'{method.capitalize()}')
        axes[idx].set_xlabel('Number of Examples')
        axes[idx].set_ylabel(metric)
        
        # Rotate x-axis labels for better readability
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Add value labels on top of bars
        for container in g.containers:
            g.bar_label(container, fmt='%.3f', fontsize=8)
        
        # Add legend
        axes[idx].legend(title='VA Type')
        
        # Set y-axis limits
        axes[idx].set_ylim(y_min, y_max)
    
    plt.suptitle(f'{metric} by Pruning Method and Number of Demos', fontsize=30, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to leave space for the title
    plt.savefig(f'experiments/{metric.lower()}_comparison.png')
    plt.close()

def main():
    directory = '/Users/kevinawang/Documents/GitHub/VA-classifier/experiments'
    
    # Load scores
    df = load_scores(directory)
    
    # Plot MAE and RMSE
    plot_metrics(df, 'MAE')
    plot_metrics(df, 'RMSE')

if __name__ == "__main__":
    main()
