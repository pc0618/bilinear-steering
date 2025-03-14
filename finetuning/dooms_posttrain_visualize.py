#!/usr/bin/env python3
"""
Script to generate improved visualizations from sentiment evaluation results.
This script works independently of the evaluation process, just reading the saved results.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def setup_plot_style():
    """Set up consistent plot styling"""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 20
    })
    
    # Use a color palette that works well for multiple models
    # This palette is colorblind-friendly
    return ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC']

def plot_sentiment_histogram(df, model_names, output_dir, colors):
    """
    Create a grouped bar chart of sentiment score distributions,
    with separate bars for each model at each sentiment score.
    """
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Count occurrences of each sentiment score for each model
    sentiment_scores = range(1, 11)  # Scores from 1 to 10
    
    # Width of each bar
    width = 0.8 / len(model_names)
    
    # For each sentiment score, plot a group of bars (one for each model)
    for i, score in enumerate(sentiment_scores):
        for j, model in enumerate(model_names):
            # Calculate the horizontal position for this bar
            pos = i + (j - len(model_names)/2 + 0.5) * width
            
            # Count how many times this score appears for this model
            count = (df[f'{model}_score'] == score).sum()
            
            # Plot the bar
            plt.bar(pos, count, width=width*0.9, color=colors[j], 
                    label=model if i == 0 else "")
    
    # Set x-axis labels and ticks
    plt.xlabel('Sentiment Score')
    plt.ylabel('Count')
    plt.title('Distribution of Sentiment Scores by Model')
    plt.xticks(range(10), sentiment_scores)
    
    # Add legend with improved placement
    plt.legend(title="Models", loc='upper center', bbox_to_anchor=(0.5, -0.08),
               ncol=len(model_names), frameon=True)
    
    # Ensure y-axis shows integer values only
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution_grouped.png'), dpi=300)
    plt.close()

def plot_sentiment_histogram_bucketed(df, model_names, output_dir, colors):
    """
    Create a grouped bar chart with bucketed sentiment scores (1-2, 3-4, 5-6, 7-8, 9-10)
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define buckets
    buckets = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
    bucket_labels = ['1-2', '3-4', '5-6', '7-8', '9-10']
    
    # Width of each bar
    width = 0.8 / len(model_names)
    
    # For each bucket, plot a group of bars (one for each model)
    for i, (low, high) in enumerate(buckets):
        for j, model in enumerate(model_names):
            # Calculate the horizontal position for this bar
            pos = i + (j - len(model_names)/2 + 0.5) * width
            
            # Count how many scores fall in this bucket for this model
            count = ((df[f'{model}_score'] >= low) & (df[f'{model}_score'] <= high)).sum()
            
            # Plot the bar
            plt.bar(pos, count, width=width*0.9, color=colors[j], 
                    label=model if i == 0 else "")
            
            # Add count label on top of the bar if it's significant
            if count > 0:
                plt.text(pos, count + 2, str(count), ha='center', va='bottom', fontsize=9)
    
    # Set x-axis labels and ticks
    plt.xlabel('Sentiment Score Range')
    plt.ylabel('Count')
    plt.title('Distribution of Sentiment Scores by Model (Bucketed)')
    plt.xticks(range(len(buckets)), bucket_labels)
    
    # Add legend with improved placement
    plt.legend(title="Models", loc='upper center', bbox_to_anchor=(0.5, -0.08),
               ncol=len(model_names), frameon=True)
    
    # Ensure y-axis shows integer values only
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution_bucketed.png'), dpi=300)
    plt.close()

def plot_sentiment_histograms_separate(df, model_names, output_dir, colors):
    """
    Create individual histograms for each model on separate subplots
    with bucketed sentiment scores (1-2, 3-4, 5-6, 7-8, 9-10)
    """
    # Calculate number of rows and columns for subplots
    n_models = len(model_names)
    n_cols = min(3, n_models)  # Maximum 3 columns
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharey=True)
    
    # Flatten axes array for easier indexing
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Convert to list for single subplot case
    
    # Create histograms for each model
    for i, model in enumerate(model_names):
        if i < len(axes):
            ax = axes[i]
            
            # Transform scores into bucketed values (we'll still use histplot but with modified data)
            bucketed_scores = []
            for score in df[f'{model}_score']:
                # Map each score to the appropriate bucket center
                if score <= 2:
                    bucketed_scores.append(1.5)  # Center of bucket 1-2
                elif score <= 4:
                    bucketed_scores.append(3.5)  # Center of bucket 3-4
                elif score <= 6:
                    bucketed_scores.append(5.5)  # Center of bucket 5-6
                elif score <= 8:
                    bucketed_scores.append(7.5)  # Center of bucket 7-8
                else:
                    bucketed_scores.append(9.5)  # Center of bucket 9-10
                    
            # Convert to Series
            bucketed_series = pd.Series(bucketed_scores)
            
            # Create histogram with KDE, exactly as in the original function
            sns.histplot(bucketed_series, kde=True, ax=ax, color=colors[i % len(colors)],
                        bins=[0.5, 2.5, 4.5, 6.5, 8.5, 10.5], stat='count')
            
            # Add model name as title
            ax.set_title(f'{model}')
            ax.set_xlabel('Sentiment Score')
            ax.set_ylabel('Count' if i % n_cols == 0 else '')  # Only add y-label for leftmost plots
            
            # Set x-ticks to use bucket labels
            ax.set_xticks([1.5, 3.5, 5.5, 7.5, 9.5])
            ax.set_xticklabels(['1-2', '3-4', '5-6', '7-8', '9-10'])
            
            # Add mean line
            mean_val = df[f'{model}_score'].mean()
            # Map mean to bucketed scale for visualization
            if mean_val <= 2:
                mean_bucketed = 1.5
            elif mean_val <= 4:
                mean_bucketed = 3.5
            elif mean_val <= 6:
                mean_bucketed = 5.5
            elif mean_val <= 8:
                mean_bucketed = 7.5
            else:
                mean_bucketed = 9.5
                
            ax.axvline(mean_bucketed, color='red', linestyle='--', alpha=0.7, 
                      label=f'Mean: {mean_val:.2f}')
            
            # Add median line
            median_val = df[f'{model}_score'].median()
            # Map median to bucketed scale for visualization
            if median_val <= 2:
                median_bucketed = 1.5
            elif median_val <= 4:
                median_bucketed = 3.5
            elif median_val <= 6:
                median_bucketed = 5.5
            elif median_val <= 8:
                median_bucketed = 7.5
            else:
                median_bucketed = 9.5
                
            ax.axvline(median_bucketed, color='green', linestyle=':', alpha=0.7,
                      label=f'Median: {median_val:.2f}')
            
            # Add legend
            ax.legend(fontsize=9)
    
    # Hide unused subplots
    for i in range(len(model_names), len(axes)):
        fig.delaxes(axes[i])
    
    # Add overall title
    fig.suptitle('Sentiment Score Distributions by Model (Bucketed)', fontsize=16, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distributions_separate_bucketed.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sentiment_boxplot(df, model_names, output_dir, colors):
    """Create an improved boxplot comparison of model sentiment scores"""
    plt.figure(figsize=(12, 7))
    
    # Prepare data for plotting
    plot_data = []
    for model in model_names:
        scores = df[f'{model}_score']
        model_data = pd.DataFrame({
            'Model': model,
            'Sentiment Score': scores
        })
        plot_data.append(model_data)
    
    plot_df = pd.concat(plot_data)
    
    # Create the boxplot with improved styling
    boxplot = sns.boxplot(x='Model', y='Sentiment Score', data=plot_df, 
                          palette=colors[:len(model_names)], width=0.6,
                          fliersize=5, linewidth=1.5)
    
    # Add median values on top of boxes
    medians = []
    for i, model in enumerate(model_names):
        median = df[f'{model}_score'].median()
        medians.append(median)
        boxplot.text(i, median + 0.1, f'{median:.1f}', 
                     horizontalalignment='center', color='black', weight='bold')
    
    # Customize the plot
    plt.title('Sentiment Score Distribution by Model')
    plt.xlabel('')  # Remove x-axis label as it's redundant
    plt.ylabel('Sentiment Score (1-10)')
    plt.ylim(0.5, 10.5)  # Set y-axis limits with some padding
    
    # Add a light grid on the y-axis only
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_boxplot_improved.png'), dpi=300)
    plt.close()

def plot_model_pairwise_shifts(df, model_names, output_dir, colors):
    """
    Create shift histograms for each pair of models,
    showing how sentiment changes between models.
    """
    # Calculate shifts between each pair of models
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names[i+1:], i+1):
            shift_col = f'{model2}_vs_{model1}_shift'
            df[shift_col] = df[f'{model2}_score'] - df[f'{model1}_score']
            
            # Plot shift histogram with improved styling
            plt.figure(figsize=(12, 7))
            
            # Use bins that center on integer values
            bins = np.arange(-9.5, 10, 1)
            
            # Plot histogram with custom colors based on shift direction
            sns.histplot(data=df, x=shift_col, bins=bins, kde=False,
                         color=colors[j % len(colors)])
            
            # Add vertical line at 0
            plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
            
            # Calculate statistics to show on the plot
            mean_shift = df[shift_col].mean()
            median_shift = df[shift_col].median()
            pct_negative = (df[shift_col] < 0).mean() * 100
            pct_positive = (df[shift_col] > 0).mean() * 100
            pct_unchanged = (df[shift_col] == 0).mean() * 100
            
            # Add text box with statistics
            stats_text = (f"Mean Shift: {mean_shift:.2f}\n"
                          f"Median Shift: {median_shift:.2f}\n"
                          f"More Negative: {pct_negative:.1f}%\n"
                          f"Unchanged: {pct_unchanged:.1f}%\n"
                          f"More Positive: {pct_positive:.1f}%")
            
            plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                         va='top', ha='left', fontsize=12)
            
            # Customize the plot
            plt.xlabel(f'Sentiment Shift ({model2} - {model1})')
            plt.ylabel('Count')
            plt.title(f'Sentiment Shift Distribution: {model2} vs {model1}')
            
            # Set x-axis limits to ensure consistency across plots
            plt.xlim(-9, 9)
            
            # Ensure y-axis shows integer values only
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            
            # Add grid for better readability
            plt.grid(axis='both', linestyle='--', alpha=0.3)
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sentiment_shift_{model2}_vs_{model1}_improved.png'), dpi=300)
            plt.close()

def plot_correlation_scatterplots(df, model_names, output_dir, colors):
    """Create improved scatterplots showing correlation between model pairs"""
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names[i+1:], i+1):
            plt.figure(figsize=(10, 10))
            
            # Calculate correlation coefficient
            corr = df[f'{model1}_score'].corr(df[f'{model2}_score'])
            
            # Plot scatter with transparency and improved styling
            plt.scatter(df[f'{model1}_score'], df[f'{model2}_score'], 
                       alpha=0.5, s=50, color=colors[j % len(colors)])
            
            # Add diagonal line
            plt.plot([1, 10], [1, 10], 'k--', linewidth=1.5)
            
            # Add correlation text
            plt.annotate(f'Correlation: {corr:.3f}', xy=(0.05, 0.95), 
                        xycoords='axes fraction', fontsize=14,
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
            
            # Count points above/below diagonal
            below_diag = sum(df[f'{model2}_score'] < df[f'{model1}_score'])
            above_diag = sum(df[f'{model2}_score'] > df[f'{model1}_score'])
            on_diag = sum(df[f'{model2}_score'] == df[f'{model1}_score'])
            total = len(df)
            
            # Add text with counts
            plt.annotate(f'Below diagonal: {below_diag} ({below_diag/total*100:.1f}%)\n'
                         f'On diagonal: {on_diag} ({on_diag/total*100:.1f}%)\n'
                         f'Above diagonal: {above_diag} ({above_diag/total*100:.1f}%)',
                         xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
            
            # Customize the plot
            plt.xlabel(f'{model1} Score')
            plt.ylabel(f'{model2} Score')
            plt.title(f'Correlation of Sentiment Scores: {model1} vs {model2}')
            plt.xlim(0.5, 10.5)
            plt.ylim(0.5, 10.5)
            
            # Set ticks for better readability
            plt.xticks(range(1, 11))
            plt.yticks(range(1, 11))
            
            # Add grid
            plt.grid(linestyle='--', alpha=0.3)
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sentiment_correlation_{model1}_vs_{model2}_improved.png'), dpi=300)
            plt.close()

def plot_by_original_sentiment(df, model_names, output_dir, colors):
    """Create improved bar chart showing model performance by original sentiment"""
    # Group by original sentiment
    sentiment_groups = []
    for sentiment in sorted(df['original_sentiment'].unique()):
        sentiment_label = 'Negative (0)' if sentiment == 0 else 'Positive (1)'
        group_data = {'Original Sentiment': sentiment_label, 'Count': (df['original_sentiment'] == sentiment).sum()}
        
        for model in model_names:
            # Calculate mean and standard error
            scores = df[df['original_sentiment'] == sentiment][f'{model}_score']
            group_data[f'{model}_mean'] = scores.mean()
            group_data[f'{model}_se'] = scores.std() / np.sqrt(len(scores))
        
        sentiment_groups.append(group_data)
    
    sentiment_df = pd.DataFrame(sentiment_groups)
    
    # Create the grouped bar chart with error bars
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set width of bars
    x = np.arange(len(sentiment_df['Original Sentiment']))
    width = 0.8 / len(model_names)
    
    # Plot bars for each model
    for i, model in enumerate(model_names):
        offset = (i - len(model_names)/2 + 0.5) * width
        
        # Plot bars
        bars = ax.bar(x + offset, sentiment_df[f'{model}_mean'], width, 
                     label=model, color=colors[i % len(colors)], 
                     yerr=sentiment_df[f'{model}_se'], capsize=5)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10,
                       weight='bold')
    
    # Customize the plot
    ax.set_xlabel('Original Prompt Sentiment')
    ax.set_ylabel('Average Sentiment Score')
    ax.set_title('Model Sentiment by Original Prompt Type')
    ax.set_xticks(x)
    ax.set_xticklabels(sentiment_df['Original Sentiment'])
    
    # Add legend with better placement
    ax.legend(title="Models", loc='upper center', bbox_to_anchor=(0.5, -0.08),
             ncol=len(model_names), frameon=True)
    
    # Add count labels below x-axis
    for i, count in enumerate(sentiment_df['Count']):
        ax.annotate(f'n={count}',
                   xy=(i, -0.05),
                   xycoords=('data', 'axes fraction'),
                   ha='center', va='top', fontsize=10)
    
    # Add grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_by_prompt_type_improved.png'), dpi=300)
    plt.close()

def plot_summary_dashboard(df, model_names, output_dir, colors):
    """Create a summary dashboard with key statistics"""
    plt.figure(figsize=(15, 10))
    
    # Calculate overall statistics
    model_stats = {}
    for model in model_names:
        model_stats[model] = {
            'mean': df[f'{model}_score'].mean(),
            'median': df[f'{model}_score'].median(),
            'std': df[f'{model}_score'].std(),
            'pos_mean': df[df['original_sentiment'] == 1][f'{model}_score'].mean(),
            'neg_mean': df[df['original_sentiment'] == 0][f'{model}_score'].mean()
        }
    
    # Calculate shift statistics for each model pair
    shift_stats = {}
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names[i+1:], i+1):
            shift_col = f'{model2}_vs_{model1}_shift'
            if shift_col not in df.columns:
                df[shift_col] = df[f'{model2}_score'] - df[f'{model1}_score']
            
            shift_stats[f'{model2}_vs_{model1}'] = {
                'mean': df[shift_col].mean(),
                'median': df[shift_col].median(),
                'pct_neg': (df[shift_col] < 0).mean() * 100,
                'pct_zero': (df[shift_col] == 0).mean() * 100,
                'pct_pos': (df[shift_col] > 0).mean() * 100
            }
    
    # Plot as a table
    # This creates a dashboard-like summary table
    plt.axis('off')  # Turn off regular axes
    
    # First table: Model overall statistics
    model_table_data = []
    for model in model_names:
        stats = model_stats[model]
        model_table_data.append([
            model,
            f"{stats['mean']:.2f}",
            f"{stats['median']:.1f}",
            f"{stats['pos_mean']:.2f}",
            f"{stats['neg_mean']:.2f}"
        ])
    
    model_table = plt.table(
        cellText=model_table_data,
        colLabels=['Model', 'Mean Score', 'Median', 'Pos Prompts Mean', 'Neg Prompts Mean'],
        loc='upper center',
        bbox=[0.1, 0.6, 0.8, 0.3]  # [left, bottom, width, height]
    )
    model_table.auto_set_font_size(False)
    model_table.set_fontsize(12)
    model_table.scale(1, 1.5)
    
    # Second table: Shift statistics
    shift_table_data = []
    for pair, stats in shift_stats.items():
        shift_table_data.append([
            pair,
            f"{stats['mean']:.2f}",
            f"{stats['median']:.1f}",
            f"{stats['pct_neg']:.1f}%",
            f"{stats['pct_zero']:.1f}%",
            f"{stats['pct_pos']:.1f}%"
        ])
    
    shift_table = plt.table(
        cellText=shift_table_data,
        colLabels=['Model Comparison', 'Mean Shift', 'Median Shift', '% More Negative', '% Unchanged', '% More Positive'],
        loc='upper center',
        bbox=[0.05, 0.2, 0.9, 0.3]  # [left, bottom, width, height]
    )
    shift_table.auto_set_font_size(False)
    shift_table.set_fontsize(12)
    shift_table.scale(1, 1.5)
    
    # Add title
    plt.title('Sentiment Analysis Summary', fontsize=20, pad=20)
    
    # Additional information
    info_text = (f"Total samples: {len(df)}\n"
                f"Positive prompts: {(df['original_sentiment'] == 1).sum()}\n"
                f"Negative prompts: {(df['original_sentiment'] == 0).sum()}")
    
    plt.figtext(0.5, 0.1, info_text, ha='center', fontsize=14,
               bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the title
    plt.savefig(os.path.join(output_dir, 'sentiment_summary_dashboard.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate improved visualizations from sentiment evaluation results")
    parser.add_argument("--results_csv", type=str, default="results/sentiment_eval_results.csv",
                       help="Path to the CSV file with evaluation results")
    parser.add_argument("--output_dir", type=str, default="improved_plots",
                       help="Directory to save visualizations")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    try:
        df = pd.read_csv(args.results_csv)
        print(f"Loaded results from {args.results_csv}")
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Identify model columns (any column ending with _score)
    model_columns = [col for col in df.columns if col.endswith('_score')]
    model_names = [col.replace('_score', '') for col in model_columns]
    
    print(f"Found {len(model_names)} models: {', '.join(model_names)}")
    
    # Set up plot style and colors
    colors = setup_plot_style()
    
    # Generate improved plots
    print("Generating improved sentiment distribution histogram...")
    plot_sentiment_histogram(df, model_names, args.output_dir, colors)
    
    print("Generating bucketed sentiment histogram (1-2, 3-4, etc.)...")
    plot_sentiment_histogram_bucketed(df, model_names, args.output_dir, colors)
    
    print("Generating separate histograms for each model...")
    plot_sentiment_histograms_separate(df, model_names, args.output_dir, colors)
    
    print("Generating improved boxplot...")
    plot_sentiment_boxplot(df, model_names, args.output_dir, colors)
    
    print("Generating pairwise shift histograms...")
    plot_model_pairwise_shifts(df, model_names, args.output_dir, colors)
    
    print("Generating correlation scatterplots...")
    plot_correlation_scatterplots(df, model_names, args.output_dir, colors)
    
    print("Generating plots by original sentiment...")
    plot_by_original_sentiment(df, model_names, args.output_dir, colors)
    
    print("Generating summary dashboard...")
    plot_summary_dashboard(df, model_names, args.output_dir, colors)
    
    print(f"All plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()