"""
Visualization Utilities

Generate publication-ready plots for experiment results.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Publication-ready style settings
STYLE_CONFIG = {
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
}

# Color palette for consistent styling
COLORS = {
    'baseline': '#2ecc71',
    'optimized': '#3498db',
    'improvement': '#27ae60',
    'degradation': '#e74c3c',
    'neutral': '#95a5a6'
}


def _check_matplotlib():
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def save_figure(fig, filepath: Path, formats: List[str] = None) -> List[Path]:
    """
    Save figure in multiple formats.

    Args:
        fig: matplotlib figure
        filepath: Base filepath (without extension)
        formats: List of formats to save. Default: ['png', 'pdf']

    Returns:
        List of saved file paths
    """
    _check_matplotlib()

    if formats is None:
        formats = ['png', 'pdf']

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        outpath = filepath.with_suffix(f'.{fmt}')
        fig.savefig(outpath, bbox_inches='tight', dpi=150)
        saved_paths.append(outpath)

    return saved_paths


def plot_field_comparison(
    field_accuracies: Dict[str, Dict[str, float]],
    title: str = "Field-wise Accuracy Comparison",
    figsize: Tuple[int, int] = (10, 6)
) -> 'plt.Figure':
    """
    Create grouped bar chart comparing field accuracies across methods.

    Args:
        field_accuracies: Dict of {method_name: {field: accuracy}}
        title: Plot title
        figsize: Figure size tuple

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()

    with plt.rc_context(STYLE_CONFIG):
        fig, ax = plt.subplots(figsize=figsize)

        methods = list(field_accuracies.keys())
        fields = list(field_accuracies[methods[0]].keys())
        n_methods = len(methods)
        n_fields = len(fields)

        bar_width = 0.8 / n_methods
        x = range(n_fields)

        colors = [COLORS.get('baseline', '#2ecc71'), COLORS.get('optimized', '#3498db')]
        if n_methods > 2:
            import matplotlib.cm as cm
            colors = cm.Set2(range(n_methods))

        for i, method in enumerate(methods):
            positions = [xi + i * bar_width for xi in x]
            values = [field_accuracies[method].get(f, 0) * 100 for f in fields]
            ax.bar(positions, values, bar_width, label=method, color=colors[i % len(colors)])

        ax.set_xlabel('Fields')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title)
        ax.set_xticks([xi + bar_width * (n_methods - 1) / 2 for xi in x])
        ax.set_xticklabels(fields, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 105)

        # Add gridlines
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        plt.tight_layout()
        return fig


def plot_accuracy_bars(
    results: Dict[str, float],
    title: str = "Overall Accuracy Comparison",
    figsize: Tuple[int, int] = (8, 5),
    show_improvement: bool = True
) -> 'plt.Figure':
    """
    Create bar chart comparing overall accuracies.

    Args:
        results: Dict of {method_name: accuracy}
        title: Plot title
        figsize: Figure size tuple
        show_improvement: Whether to annotate improvement percentage

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()

    with plt.rc_context(STYLE_CONFIG):
        fig, ax = plt.subplots(figsize=figsize)

        methods = list(results.keys())
        accuracies = [results[m] * 100 for m in methods]

        colors = [COLORS['baseline'] if i == 0 else COLORS['optimized']
                  for i in range(len(methods))]

        bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.annotate(f'{acc:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')

        # Show improvement arrow if applicable
        if show_improvement and len(accuracies) >= 2:
            improvement = accuracies[1] - accuracies[0]
            if improvement > 0:
                ax.annotate(f'+{improvement:.1f}%',
                           xy=(0.5, max(accuracies) + 5),
                           ha='center', fontsize=14, fontweight='bold',
                           color=COLORS['improvement'])

        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title)
        ax.set_ylim(0, max(accuracies) + 15)

        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        plt.tight_layout()
        return fig


def plot_improvement_heatmap(
    comparison: Dict[str, Any],
    title: str = "Per-Sample Improvement Analysis",
    figsize: Tuple[int, int] = (12, 4)
) -> 'plt.Figure':
    """
    Create heatmap showing per-sample improvements/degradations.

    Args:
        comparison: Comparison dict from compare_results()
        title: Plot title
        figsize: Figure size tuple

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()

    with plt.rc_context(STYLE_CONFIG):
        fig, ax = plt.subplots(figsize=figsize)

        n_samples = comparison['summary']['total_samples']
        improvements = {s['sample_id']: s['difference'] for s in comparison['improvements']}
        degradations = {s['sample_id']: -s['difference'] for s in comparison['degradations']}

        # Create color array
        colors_arr = []
        for i in range(1, n_samples + 1):
            if i in improvements:
                colors_arr.append(improvements[i])
            elif i in degradations:
                colors_arr.append(degradations[i])
            else:
                colors_arr.append(0)

        # Plot as horizontal bar segments
        for i, val in enumerate(colors_arr):
            if val > 0:
                color = COLORS['improvement']
            elif val < 0:
                color = COLORS['degradation']
            else:
                color = COLORS['neutral']
            ax.barh(0, 1, left=i, color=color, edgecolor='white', linewidth=0.5)

        ax.set_xlim(0, n_samples)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel('Sample Index')
        ax.set_title(title)

        # Legend
        legend_patches = [
            mpatches.Patch(color=COLORS['improvement'], label=f"Improved ({len(improvements)})"),
            mpatches.Patch(color=COLORS['degradation'], label=f"Degraded ({len(degradations)})"),
            mpatches.Patch(color=COLORS['neutral'], label=f"Unchanged ({comparison['summary']['total_unchanged']})")
        ]
        ax.legend(handles=legend_patches, loc='upper right')

        plt.tight_layout()
        return fig
