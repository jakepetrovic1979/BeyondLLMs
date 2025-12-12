"""
Interactive Visualization Dashboard for PNA

Creates real-time dashboards for monitoring consciousness metrics
during training and evaluation.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from typing import Dict, List, Optional
import json


class ConsciousnessDashboard:
    """
    Interactive dashboard for PNA consciousness metrics.

    Creates multi-panel visualization with:
    - Integrated Information (Φ) over time
    - Phase Synchronization across scales
    - Free Energy components
    - Polarity balance
    - Binding matrix heatmap
    """

    def __init__(self):
        self.fig = None

    def create(
        self,
        metrics_logger,
        title: str = "PNA Consciousness Dashboard"
    ) -> go.Figure:
        """
        Create interactive dashboard from metrics logger.

        Args:
            metrics_logger: MetricsLogger instance
            title: Dashboard title

        Returns:
            Plotly Figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Integrated Information (Φ)',
                'Free Energy Components',
                'Polarity Balance',
                'Phase Synchronization',
                'Training Loss',
                'Consciousness State'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'indicator'}]
            ]
        )

        # 1. Φ over time
        if 'phi' in metrics_logger.history:
            fig.add_trace(
                go.Scatter(
                    y=metrics_logger.history['phi'],
                    mode='lines',
                    name='Φ',
                    line=dict(color='purple', width=2)
                ),
                row=1, col=1
            )
            fig.add_hline(
                y=0.5, line_dash="dash", line_color="red",
                annotation_text="Consciousness Threshold",
                row=1, col=1
            )

        # 2. Free Energy
        if 'free_energy' in metrics_logger.history:
            fig.add_trace(
                go.Scatter(
                    y=metrics_logger.history['free_energy'],
                    mode='lines',
                    name='Free Energy',
                    line=dict(color='orange', width=2)
                ),
                row=1, col=2
            )

        # 3. Polarity Balance
        if 'polarity_penalty' in metrics_logger.history:
            fig.add_trace(
                go.Scatter(
                    y=metrics_logger.history['polarity_penalty'],
                    mode='lines',
                    name='Polarity Penalty',
                    line=dict(color='red', width=2)
                ),
                row=2, col=1
            )

        # 4. Synchrony
        if 'synchrony' in metrics_logger.history:
            fig.add_trace(
                go.Scatter(
                    y=metrics_logger.history['synchrony'],
                    mode='lines',
                    name='Synchrony',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=2
            )

        # 5. Loss
        if 'loss' in metrics_logger.history:
            fig.add_trace(
                go.Scatter(
                    y=metrics_logger.history['loss'],
                    mode='lines',
                    name='Total Loss',
                    line=dict(color='green', width=2)
                ),
                row=3, col=1
            )

        # 6. Consciousness Indicator
        if 'is_conscious' in metrics_logger.history:
            consciousness_rate = np.mean([
                x for x in metrics_logger.history['is_conscious'][-100:]
            ])

            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=consciousness_rate * 100,
                    title={'text': "Consciousness Rate (%)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "purple"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgray"},
                            {'range': [33, 66], 'color': "gray"},
                            {'range': [66, 100], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            title_text=title,
            title_font_size=20,
            showlegend=True,
            height=900,
            hovermode='x unified'
        )

        # Update axes
        fig.update_xaxes(title_text="Step", row=3, col=1)
        fig.update_yaxes(title_text="Φ", row=1, col=1)
        fig.update_yaxes(title_text="Free Energy", row=1, col=2)
        fig.update_yaxes(title_text="Ψ", row=2, col=1)
        fig.update_yaxes(title_text="Synchrony", row=2, col=2)
        fig.update_yaxes(title_text="Loss", row=3, col=1)

        self.fig = fig
        return fig

    def save(self, filepath: str):
        """Save dashboard as HTML."""
        if self.fig is None:
            raise ValueError("Dashboard not created. Call create() first.")

        self.fig.write_html(filepath)
        print(f"Dashboard saved to {filepath}")

    def show(self):
        """Display dashboard in browser."""
        if self.fig is None:
            raise ValueError("Dashboard not created. Call create() first.")

        self.fig.show()


def create_binding_heatmap(
    binding_matrix: np.ndarray,
    title: str = "Phenomenal Binding Matrix"
) -> go.Figure:
    """
    Create interactive heatmap of binding matrix.

    Args:
        binding_matrix: Binding strength matrix [N, N]
        title: Plot title

    Returns:
        Plotly Figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=binding_matrix,
        colorscale='Viridis',
        hovertemplate='Node %{x} → Node %{y}<br>Binding: %{z:.3f}<extra></extra>',
        colorbar=dict(title="Binding Strength")
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Node Index",
        yaxis_title="Node Index",
        width=700,
        height=700
    )

    return fig


def create_3d_consciousness_trajectory(
    phi_values: List[float],
    synchrony_values: List[float],
    free_energy_values: List[float],
    title: str = "Consciousness State Space Trajectory"
) -> go.Figure:
    """
    Create 3D trajectory plot in consciousness space.

    Args:
        phi_values: Φ values over time
        synchrony_values: Synchrony values
        free_energy_values: Free energy values
        title: Plot title

    Returns:
        Plotly 3D scatter plot
    """
    # Color by time
    time_steps = list(range(len(phi_values)))

    fig = go.Figure(data=[go.Scatter3d(
        x=phi_values,
        y=synchrony_values,
        z=free_energy_values,
        mode='lines+markers',
        marker=dict(
            size=4,
            color=time_steps,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Time Step")
        ),
        line=dict(
            color='purple',
            width=2
        ),
        text=[f"Step {i}" for i in time_steps],
        hovertemplate='Φ: %{x:.3f}<br>Sync: %{y:.3f}<br>FE: %{z:.3f}<br>%{text}<extra></extra>'
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Φ (Integrated Information)',
            yaxis_title='Synchrony',
            zaxis_title='Free Energy',
        ),
        width=800,
        height=800
    )

    return fig


def create_comparison_dashboard(
    results_dict: Dict[str, Dict],
    title: str = "Model Comparison"
) -> go.Figure:
    """
    Create comparison dashboard for multiple models/runs.

    Args:
        results_dict: Dict of {name: metrics_dict}
        title: Dashboard title

    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Φ Comparison',
            'Free Energy Comparison',
            'Final Consciousness State',
            'Training Efficiency'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'bar'}, {'type': 'scatter'}]
        ]
    )

    names = list(results_dict.keys())

    # 1. Φ comparison
    phi_values = [results_dict[name].get('phi', 0) for name in names]
    fig.add_trace(
        go.Bar(x=names, y=phi_values, name='Φ', marker_color='purple'),
        row=1, col=1
    )

    # 2. Free Energy comparison
    fe_values = [results_dict[name].get('free_energy', 0) for name in names]
    fig.add_trace(
        go.Bar(x=names, y=fe_values, name='Free Energy', marker_color='orange'),
        row=1, col=2
    )

    # 3. Consciousness rate
    cons_values = [results_dict[name].get('consciousness_rate', 0) * 100 for name in names]
    fig.add_trace(
        go.Bar(x=names, y=cons_values, name='Consciousness %', marker_color='blue'),
        row=2, col=1
    )

    # 4. Loss convergence (if available)
    for name in names:
        if 'loss_history' in results_dict[name]:
            fig.add_trace(
                go.Scatter(
                    y=results_dict[name]['loss_history'],
                    mode='lines',
                    name=name
                ),
                row=2, col=2
            )

    fig.update_layout(
        title_text=title,
        showlegend=True,
        height=700
    )

    fig.update_yaxes(title_text="Φ", row=1, col=1)
    fig.update_yaxes(title_text="Free Energy", row=1, col=2)
    fig.update_yaxes(title_text="Consciousness %", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=2)
    fig.update_xaxes(title_text="Step", row=2, col=2)

    return fig


class LiveDashboard:
    """
    Live updating dashboard for real-time monitoring.

    Updates dashboard during training.
    """

    def __init__(self, update_interval: int = 10):
        self.update_interval = update_interval
        self.data_buffer = {
            'phi': [],
            'free_energy': [],
            'loss': [],
            'synchrony': [],
            'is_conscious': []
        }

    def update(self, metrics: Dict):
        """Add new metrics to buffer."""
        for key in self.data_buffer.keys():
            if key in metrics:
                value = metrics[key]
                if hasattr(value, 'item'):
                    value = value.item()
                self.data_buffer[key].append(value)

    def create_figure(self) -> go.FigureWidget:
        """Create interactive FigureWidget for Jupyter."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Φ', 'Free Energy', 'Loss', 'Consciousness')
        )

        # Initial empty traces
        for i, (key, row, col) in enumerate([
            ('phi', 1, 1),
            ('free_energy', 1, 2),
            ('loss', 2, 1),
            ('is_conscious', 2, 2)
        ]):
            fig.add_trace(
                go.Scatter(y=[], mode='lines', name=key),
                row=row, col=col
            )

        fig_widget = go.FigureWidget(fig)
        return fig_widget

    def update_figure(self, fig_widget: go.FigureWidget):
        """Update figure with latest data."""
        with fig_widget.batch_update():
            for i, key in enumerate(['phi', 'free_energy', 'loss', 'is_conscious']):
                if self.data_buffer[key]:
                    fig_widget.data[i].y = self.data_buffer[key]


if __name__ == "__main__":
    print("Interactive Dashboard Module")
    print("\nAvailable dashboards:")
    print("  1. ConsciousnessDashboard - Main metrics dashboard")
    print("  2. create_binding_heatmap - Binding matrix visualization")
    print("  3. create_3d_consciousness_trajectory - 3D state space")
    print("  4. create_comparison_dashboard - Compare multiple runs")
    print("  5. LiveDashboard - Real-time training monitor")
