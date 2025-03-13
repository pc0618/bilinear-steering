import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def load_all_results(probes_dir):
    """Load all results from the probes directory"""
    results_path = os.path.join(probes_dir, "all_results.json")
    
    if not os.path.exists(results_path):
        # Try to reconstruct from individual metrics files
        all_results = {}
        for layer_dir in os.listdir(probes_dir):
            if os.path.isdir(os.path.join(probes_dir, layer_dir)):
                metrics_path = os.path.join(probes_dir, layer_dir, "metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    
                    # Convert directory name back to layer name
                    layer_name = layer_dir.replace("_", ".")
                    all_results[layer_name] = {
                        "metrics": metrics,
                        "probe_path": os.path.join(layer_dir, "probe.pt")
                    }
    else:
        with open(results_path, 'r') as f:
            all_results = json.load(f)
    
    return all_results

def plot_layer_performance(results, metric_name, output_path=None, layer_pattern=None):
    """Plot performance of linear probes across layers"""
    # Extract layer names and metrics
    layer_names = []
    metrics = []
    
    # Filter layers if pattern provided
    import re
    pattern = re.compile(layer_pattern) if layer_pattern else None
    
    for layer_name, layer_results in results.items():
        if pattern and not pattern.search(layer_name):
            continue
            
        if metric_name in layer_results["metrics"]:
            layer_names.append(layer_name)
            metrics.append(layer_results["metrics"][metric_name])
    
    # Sort by layer index if possible
    try:
        # Extract layer indices and sort
        indices = []
        for name in layer_names:
            # Try to find a number in the layer name
            match = re.search(r'layers\.(\d+)', name)
            if match:
                indices.append(int(match.group(1)))
            else:
                indices.append(-1)  # For layers without a clear index
        
        # Sort by index
        sorted_indices = sorted(range(len(indices)), key=lambda i: indices[i])
        layer_names = [layer_names[i] for i in sorted_indices]
        metrics = [metrics[i] for i in sorted_indices]
    except Exception as e:
        print(f"Error sorting layers: {e}")
        print("Using original order")
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Simplify layer names for display
    display_names = []
    for name in layer_names:
        # Try to extract just the layer number
        match = re.search(r'layers\.(\d+)', name)
        if match:
            display_names.append(f"Layer {match.group(1)}")
        else:
            # Simplify by removing common prefixes
            simplified = name.replace("model.layers.", "").replace(".mlp", "")
            display_names.append(simplified)
    
    # Create the plot
    sns.set_style("whitegrid")
    plt.bar(display_names, metrics)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Set labels and title
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} by Layer")
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

# The visualization script (paste.txt) doesn't need many changes,
# but let's update the plot_feature_importance function to handle 
# the 1D weight vector case more explicitly:

def plot_feature_importance(results, layer_name, top_n=10, output_path=None):
    """Plot feature importance for a specific layer"""
    if layer_name not in results:
        print(f"Layer {layer_name} not found in results")
        return
    
    layer_results = results[layer_name]
    
    if "top_features" not in layer_results or "top_weights" not in layer_results:
        print(f"Feature importance data not found for layer {layer_name}")
        return
    
    # Extract data
    features = layer_results["top_features"][:top_n]
    weights = layer_results["top_weights"][:top_n]
    
    # Convert feature indices to strings for display
    feature_labels = [f"Feature {f}" if isinstance(f, (int, np.integer)) else f for f in features]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Sort by weight magnitude
    sorted_indices = sorted(range(len(weights)), key=lambda i: abs(weights[i]), reverse=True)
    sorted_labels = [feature_labels[i] for i in sorted_indices]
    sorted_weights = [weights[i] for i in sorted_indices]
    
    # Create plot
    bars = plt.barh(sorted_labels, sorted_weights)
    
    # Color code based on sign
    for i, bar in enumerate(bars):
        if sorted_weights[i] < 0:
            bar.set_color('r')
    
    plt.title(f"Top Feature Importances for {layer_name}")
    plt.xlabel("Weight Magnitude")
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

# The rest of the visualization script remains the same, as the
# feature importance calculation is already handled in the training script

def create_heatmap(results, metric_name="accuracy", output_path=None, layer_pattern=None):
    """Create a heatmap of layer performances"""
    # Extract layer indices and metrics
    layer_indices = []
    metrics = []
    layer_names = []
    
    # Filter layers if pattern provided
    import re
    pattern = re.compile(layer_pattern) if layer_pattern else None
    
    for layer_name, layer_results in results.items():
        if pattern and not pattern.search(layer_name):
            continue
            
        if metric_name in layer_results["metrics"]:
            # Try to extract layer index
            match = re.search(r'layers\.(\d+)', layer_name)
            if match:
                index = int(match.group(1))
                layer_indices.append(index)
                metrics.append(layer_results["metrics"][metric_name])
                layer_names.append(layer_name)
    
    # Check if we have any data
    if not layer_indices:
        print("No layer indices found in layer names")
        return
    
    # Sort by layer index
    sorted_indices = sorted(range(len(layer_indices)), key=lambda i: layer_indices[i])
    layer_indices = [layer_indices[i] for i in sorted_indices]
    metrics = [metrics[i] for i in sorted_indices]
    layer_names = [layer_names[i] for i in sorted_indices]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    sns.set_style("white")
    
    # Create a matrix of layer indices vs metrics
    max_layer = max(layer_indices)
    metric_matrix = np.zeros(max_layer + 1)
    
    for i, metric in zip(layer_indices, metrics):
        metric_matrix[i] = metric
    
    # Create heatmap
    ax = sns.heatmap(
        metric_matrix.reshape(-1, 1), 
        annot=True, 
        fmt=".5f", 
        cmap="viridis",
        yticklabels=[f"Layer {i}" for i in range(len(metric_matrix))],
        xticklabels=[metric_name]
    )
    
    plt.title(f"{metric_name.capitalize()} by Layer")
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def compare_models(dirs, names=None, metric_name="accuracy", output_path=None):
    """Compare performance across different models"""
    if names is None:
        names = [os.path.basename(d) for d in dirs]
    
    # Load results for each model
    all_model_results = []
    
    for d in dirs:
        results = load_all_results(d)
        all_model_results.append(results)
    
    # Extract common layers and metrics
    common_layers = set.intersection(*[set(r.keys()) for r in all_model_results])
    
    if not common_layers:
        print("No common layers found across models")
        return
    
    # Convert to list and sort
    common_layers = list(common_layers)
    
    # Try to sort by layer index
    try:
        import re
        layer_indices = []
        for layer in common_layers:
            match = re.search(r'layers\.(\d+)', layer)
            if match:
                layer_indices.append(int(match.group(1)))
            else:
                layer_indices.append(-1)
        
        sorted_indices = sorted(range(len(layer_indices)), key=lambda i: layer_indices[i])
        common_layers = [common_layers[i] for i in sorted_indices]
    except Exception as e:
        print(f"Error sorting layers: {e}")
    
    # Extract metrics for each model
    model_metrics = []
    
    for results in all_model_results:
        metrics = []
        for layer in common_layers:
            if layer in results and metric_name in results[layer]["metrics"]:
                metrics.append(results[layer]["metrics"][metric_name])
            else:
                metrics.append(np.nan)
        model_metrics.append(metrics)
    
    # Create figure
    plt.figure(figsize=(14, 7))
    sns.set_style("whitegrid")
    
    # Simplify layer names for display
    display_names = []
    for name in common_layers:
        # Try to extract just the layer number
        match = re.search(r'layers\.(\d+)', name)
        if match:
            display_names.append(f"Layer {match.group(1)}")
        else:
            # Simplify by removing common prefixes
            simplified = name.replace("model.layers.", "").replace(".mlp", "")
            display_names.append(simplified)
    
    # Plot each model
    for metrics, name in zip(model_metrics, names):
        plt.plot(display_names, metrics, marker='o', label=name)
    
    plt.xlabel("Layer")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} Comparison Across Models")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize linear probe results")
    parser.add_argument("--probes_dir", type=str, required=True, help="Directory with probe results")
    parser.add_argument("--output_dir", type=str, default="./probe_visualizations", help="Output directory for plots")
    parser.add_argument("--metric", type=str, default="accuracy", help="Metric to visualize")
    parser.add_argument("--layer_pattern", type=str, default=None, help="Regex pattern to filter layers")
    parser.add_argument("--compare_dirs", type=str, nargs='+', default=None, help="Directories to compare")
    parser.add_argument("--compare_names", type=str, nargs='+', default=None, help="Names for comparison")
    parser.add_argument("--top_features", action="store_true", help="Plot top features for each layer")
    parser.add_argument("--heatmap", action="store_true", help="Create a heatmap visualization")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_all_results(args.probes_dir)
    
    # Plot layer performance
    plot_layer_performance(
        results, 
        args.metric, 
        output_path=os.path.join(args.output_dir, f"layer_performance_{args.metric}.png"),
        layer_pattern=args.layer_pattern
    )
    
    # Create heatmap if requested
    if args.heatmap:
        create_heatmap(
            results,
            metric_name=args.metric,
            output_path=os.path.join(args.output_dir, f"heatmap_{args.metric}.png"),
            layer_pattern=args.layer_pattern
        )
    
    # Plot top features if requested
    if args.top_features:
        features_dir = os.path.join(args.output_dir, "feature_importance")
        os.makedirs(features_dir, exist_ok=True)
        
        for layer_name in results.keys():
            if args.layer_pattern and not re.search(args.layer_pattern, layer_name):
                continue
                
            if "top_features" in results[layer_name] and "top_weights" in results[layer_name]:
                # Create a safe filename
                safe_name = layer_name.replace(".", "_")
                output_path = os.path.join(features_dir, f"{safe_name}_features.png")
                
                plot_feature_importance(
                    results,
                    layer_name,
                    output_path=output_path
                )
    
    # Compare models if requested
    if args.compare_dirs:
        if args.compare_names and len(args.compare_names) != len(args.compare_dirs):
            print("Warning: Number of comparison names does not match number of directories")
            args.compare_names = None
            
        compare_models(
            args.compare_dirs,
            names=args.compare_names,
            metric_name=args.metric,
            output_path=os.path.join(args.output_dir, "model_comparison.png")
        )

if __name__ == "__main__":
    main()