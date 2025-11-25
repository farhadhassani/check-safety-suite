"""
generate_dashboard.py
Generates a standalone interactive HTML dashboard for the Check Safety Suite.
"""
import json
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import sys

# Set template
pio.templates.default = "plotly_white"

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def create_gauge(value, title, max_val=1.0, color="blue"):
    return go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_val]},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
        }
    )

def generate_dashboard():
    data_path = Path("benchmarks/results/benchmark_data.json")
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return

    data = load_data(data_path)
    summary = data['summary']
    
    # Create main figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "heatmap"}, {"type": "xy"}]
        ],
        subplot_titles=(
            "F1 Score", "Inference Latency (ms)",
            "Model Comparison (F1 Score)", "ROC Curve",
            "Confusion Matrix", "Latency Distribution"
        ),
        vertical_spacing=0.12
    )

    # 1. Overview Gauges
    fig.add_trace(create_gauge(summary['f1_score'] * 100, "F1 Score (%)", 100, "#667eea"), row=1, col=1)
    fig.add_trace(create_gauge(summary['latency_ms'], "Latency (ms)", 200, "#764ba2"), row=1, col=2)

    # 2. Model Comparison
    comp = data['comparison']
    fig.add_trace(go.Bar(
        x=comp['models'],
        y=comp['f1_scores'],
        marker_color=['#e0e0e0', '#bdc3c7', '#95a5a6', '#667eea'],
        text=[f"{v:.1%}" for v in comp['f1_scores']],
        textposition='auto',
        name="F1 Score"
    ), row=2, col=1)

    # 3. ROC Curve
    roc = data['roc_curve']
    fig.add_trace(go.Scatter(
        x=roc['fpr'],
        y=roc['tpr'],
        mode='lines+markers',
        name='ROC Curve',
        line=dict(color='#f5576c', width=3),
        fill='tozeroy'
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        showlegend=False
    ), row=2, col=2)

    # 4. Confusion Matrix
    cm = data['confusion_matrix']
    z = cm['matrix']
    x = cm['labels']
    y = cm['labels']
    
    # Add annotations
    annotations = []
    for i in range(len(y)):
        for j in range(len(x)):
            annotations.append(dict(
                x=x[j], y=y[i],
                text=str(z[i][j]),
                font=dict(color='white' if z[i][j] > 500 else 'black'),
                showarrow=False
            ))

    fig.add_trace(go.Heatmap(
        z=z, x=x, y=y,
        colorscale='Blues',
        showscale=False,
        texttemplate="%{z}",
        name="Confusion Matrix"
    ), row=3, col=1)

    # 5. Latency Distribution
    lat = data['latency_distribution']
    fig.add_trace(go.Bar(
        x=lat['bins'],
        y=lat['counts'],
        marker_color='#a8edea',
        name="Latency Count"
    ), row=3, col=2)

    # Layout updates
    fig.update_layout(
        title_text="<b>Check Safety Suite - Performance Dashboard</b>",
        title_x=0.5,
        title_font_size=24,
        height=1200,
        showlegend=False,
        template="plotly_white"
    )
    
    # Axis labels
    fig.update_yaxes(title_text="F1 Score", row=2, col=1)
    fig.update_xaxes(title_text="False Positive Rate", row=2, col=2)
    fig.update_yaxes(title_text="True Positive Rate", row=2, col=2)
    fig.update_xaxes(title_text="Latency (ms)", row=3, col=2)
    fig.update_yaxes(title_text="Count", row=3, col=2)

    # Save
    output_path = Path("docs/dashboard.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Dashboard generated at: {output_path.absolute()}")

if __name__ == "__main__":
    generate_dashboard()
