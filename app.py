import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from dotenv import load_dotenv
from O365 import Account
import tempfile
import networkx as nx
import pickle
import json
import streamlit.components.v1 as components
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas
from datetime import datetime, timedelta
import base64
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
from reportlab.graphics import renderPDF
import math
from utils.forecast import BudgetForecaster
from utils.fiscal_forecast import FiscalYearForecaster
from utils.staff_tracking import StaffTracker

# Import styling from styles.py
from utils.styles import get_css_styles, COLORS

# Import AI helper
from utils.ai_helper import BudgetAI

# Load environment variables
load_dotenv()

# Create cache directory if it doesn't exist
CACHE_DIR = os.path.join(tempfile.gettempdir(), 'streamlit_budget_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, 'data_cache.pkl')

def save_to_cache(df, filename):
    """Save DataFrame and filename to cache"""
    cache_data = {
        'df': df,
        'filename': filename
    }
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)

def load_from_cache():
    """Load DataFrame and filename from cache"""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
            return cache_data.get('df'), cache_data.get('filename')
    except Exception:
        pass
    return None, None

def clear_cache():
    """Clear the cache file"""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

# Set page config
st.set_page_config(
    page_title="Budget Visualization Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply the custom CSS from styles.py
st.markdown(get_css_styles(), unsafe_allow_html=True)

def load_excel_data(file):
    """Load and process Excel or CSV data"""
    df = None # Initialize df
    try:
        file_extension = Path(file.name).suffix.lower()
        if file_extension in ['.xlsx', '.xls']:
            engine = 'openpyxl' if file_extension == '.xlsx' else 'xlrd'
            df = pd.read_excel(file, engine=engine)
        elif file_extension == '.csv':
            df = pd.read_csv(file)
        else:
            st.error("Unsupported file format. Please upload an Excel file (.xlsx or .xls) or CSV file (.csv)")
            return None

        # --- Post-Loading Validation and Cleaning ---
        if df is not None:
            # 1. Check required columns
            required_columns = ['Project', 'Expense', 'Budget Line', 'Amount', 'Date']
            if not all(col in df.columns for col in required_columns):
                st.error("File must contain the following columns: Project, Expense, Budget Line, Amount, Date")
                return None

            # 2. Handle 'Vendor Name' (Create if missing, then clean)
            if 'Vendor Name' not in df.columns:
                st.warning("'Vendor Name' column not found. Creating an empty column.")
                df['Vendor Name'] = "" # Create as empty string
            # Ensure it's string type, fill any potential NaNs (from blanks in CSV), strip whitespace
            df['Vendor Name'] = df['Vendor Name'].astype(str).fillna('').str.strip()

            # 3. Handle Optional Date Columns (Create if missing, convert to datetime)
            for date_col in ['Start Date', 'End Date']:
                if date_col not in df.columns:
                    df[date_col] = pd.NaT # Use pandas NaT for missing datetimes
                else:
                    # Convert existing column, making errors into NaT
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    if df[date_col].isnull().all() and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                         # If all values failed conversion and it's not already datetime, try warning user
                         st.info(f"Column '{date_col}' exists but contains values that could not be parsed as dates.")


            # 4. Convert main 'Date' column robustly
            try:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                if df['Date'].isnull().any():
                    # Check if the original column had non-null values before coercing
                    # This requires checking before conversion or handling this differently.
                    # For now, just warn if any dates are missing after conversion.
                     st.warning("Some values in the main 'Date' column could not be converted to dates and are now treated as missing.")
            except Exception as e:
                st.error(f"Error converting main 'Date' column: {e}")
                return None # Main date is critical

        return df
        # --- End Post-Loading Validation ---

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def create_project_expense_chart(df, selected_view="All Selected Projects"):
    """Create interactive horizontal bar chart for project expenses or expense breakdown for a specific project"""
    if selected_view == "All Selected Projects":
        # Show total expenses by project
        plot_df = df.groupby('Project')['Amount'].sum().reset_index()
        title = 'Total Expenses by Project'
        y_column = 'Project'
        # Alternate colors for bars
        colors = [COLORS['primary'] if i % 2 == 0 else COLORS['secondary'] 
                 for i in range(len(plot_df))]
    else:
        # Show expense breakdown for selected project
        plot_df = df[df['Project'] == selected_view].groupby('Budget Line')['Amount'].sum().reset_index()
        title = f'Expense Breakdown - {selected_view}'
        y_column = 'Budget Line'
        # Use gradient colors for breakdown
        colors = [COLORS['secondary']] * len(plot_df)

    fig = px.bar(
        plot_df,
        x='Amount',
        y=y_column,
        orientation='h',
        title=title,
        height=380  # Increased height
    )
    
    # Update bar colors
    fig.update_traces(marker_color=colors)
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_color=COLORS['text'],
        title_font_size=14,  # Smaller title font
        margin=dict(l=10, r=10, t=30, b=10),  # Reduced margins
        yaxis_title="",
        xaxis_title="Amount ($)",
        showlegend=False,
        font=dict(size=10)  # Smaller font
    )
    
    # Update hover template based on view
    if selected_view == "All Selected Projects":
        hover_template = "<b>%{y}</b><br>Total: $%{x:,.2f}"
    else:
        hover_template = "<b>%{y}</b><br>Amount: $%{x:,.2f}"
    
    fig.update_traces(
        hovertemplate=hover_template,
    )
    return fig

def create_budget_impact_chart(df, selected_projects, title_suffix=""):
    """Create line chart showing budget impact"""
    if not selected_projects:
        return None
    
    if isinstance(selected_projects, str):
        selected_projects = [selected_projects]
        
    filtered_df = df[df['Project'].isin(selected_projects)]
    
    # Create title based on selection
    if len(selected_projects) == 1:
        title = f'Budget Impact Over Time - {selected_projects[0]}'
        line_color = COLORS['secondary']  # Use purple for single project
        fill_color = 'rgba(123, 104, 238, 0.2)'  # Purple with opacity
    else:
        title = 'Cumulative Budget Impact Over Time - Multiple Projects'
        line_color = COLORS['primary']  # Use blue for multiple projects
        fill_color = 'rgba(74, 144, 226, 0.2)'  # Blue with opacity
    
    fig = px.line(
        filtered_df.groupby('Date')['Amount'].sum().cumsum().reset_index(),
        x='Date',
        y='Amount',
        title=title,
        height=380  # Increased height
    )
    
    # Update line color
    fig.update_traces(line_color=line_color)
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_color=COLORS['text'],
        title_font_size=14,  # Smaller title font
        margin=dict(l=10, r=10, t=30, b=10),  # Reduced margins
        yaxis_title="Cumulative Amount ($)",
        xaxis_title="Date",
        font=dict(size=10)  # Smaller font
    )
    
    # Add gradient fill
    fig.add_traces(
        go.Scatter(
            x=filtered_df.groupby('Date')['Amount'].sum().cumsum().reset_index()['Date'],
            y=filtered_df.groupby('Date')['Amount'].sum().cumsum().reset_index()['Amount'],
            fill='tozeroy',
            fillcolor=fill_color,
            line=dict(color=line_color),
            showlegend=False
        )
    )
    return fig

def create_network_graph(df, expanded_nodes=None):
    """Create an interactive network graph showing relationships between projects and budget lines"""
    if expanded_nodes is None:
        expanded_nodes = set()
        
    # Create nodes for projects and budget lines
    project_amounts = df.groupby('Project')['Amount'].sum()
    budget_amounts = df.groupby('Budget Line')['Amount'].sum()
    
    # Create nodes
    nodes = []
    node_colors = []
    node_sizes = []
    node_texts = []
    node_types = []  # To keep track of node types (project vs budget)
    
    # Add project nodes first (always visible)
    for proj, amount in project_amounts.items():
        nodes.append(proj)
        node_colors.append(COLORS['primary'])
        base_size = 30  # Reduced from 40
        size_scale = 20  # Reduced from 30
        node_sizes.append(base_size + (amount / project_amounts.max() * size_scale))
        node_texts.append(
            f"Project: {proj}<br>"
            f"Total: ${amount:,.2f}<br>"
            f"Click to {'collapse' if proj in expanded_nodes else 'expand'}"
        )
        node_types.append('project')
    
    # Add budget line nodes only for expanded projects
    edges_to_show = []
    for proj in expanded_nodes:
        project_budget_lines = df[df['Project'] == proj]['Budget Line'].unique()
        for budget in project_budget_lines:
            if budget not in nodes:  # Add budget node if not already added
                amount = budget_amounts[budget]
                nodes.append(budget)
                node_colors.append(COLORS['secondary'])
                node_sizes.append(base_size + (amount / budget_amounts.max() * size_scale))
                node_texts.append(f"Budget Line: {budget}<br>Total: ${amount:,.2f}")
                node_types.append('budget')
            # Add edge to the list of edges to show
            edges_to_show.append((proj, budget))
    
    # Create network layout
    G = nx.Graph()
    
    # Add nodes with their types
    for i, node in enumerate(nodes):
        G.add_node(node, node_type=node_types[i])
    
    # Add edges with weights only for expanded nodes
    edges_df = df.groupby(['Project', 'Budget Line'])['Amount'].sum().reset_index()
    for _, row in edges_df.iterrows():
        if (row['Project'], row['Budget Line']) in edges_to_show:
            G.add_edge(row['Project'], row['Budget Line'], weight=row['Amount'])
    
    # Use circular layout for even distribution
    pos = nx.circular_layout(G, scale=1.8)  # Reduced scale from 2.0
    
    # Extract node positions
    node_x = []
    node_y = []
    for node in nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    # Create edges (lines between nodes)
    edge_traces = []
    if edges_to_show:  # Only create edges if there are expanded nodes
        max_amount = edges_df['Amount'].max()
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            amount = G.edges[edge]['weight']
            opacity = min(1, amount / max_amount)
            
            # Create curved edges
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            curve_x = mid_x + (y1 - y0) * 0.1
            curve_y = mid_y - (x1 - x0) * 0.1
            
            edge_trace = go.Scatter(
                x=[x0, curve_x, x1],
                y=[y0, curve_y, y1],
                mode='lines',
                line=dict(
                    width=1 + (opacity * 2),  # Reduced from 3
                    color=f'rgba(74, 144, 226, {opacity})',
                    shape='spline'
                ),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
    
    # Create the figure
    fig = go.Figure()
    
    # Add all edge traces
    for trace in edge_traces:
        fig.add_trace(trace)
    
    # Add nodes
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color='#FFFFFF'),  # Reduced from 2
            opacity=0.9
        ),
        text=nodes,
        hovertext=node_texts,
        hoverinfo='text',
        textposition='middle center',
        textfont=dict(
            size=11,  # Reduced from 14
            color=COLORS['text']
        )
    )
    fig.add_trace(node_trace)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Project-Budget Network Analysis<br><sup>Click on projects to expand/collapse budget lines</sup>',
            x=0.5,
            y=0.98,
            xanchor='center',
            yanchor='top',
            font=dict(size=14)
        ),
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_color=COLORS['text'],
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-2.5, 2.5],
            scaleanchor='y',
            scaleratio=1,
            constrain='domain',
            fixedrange=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-2.5, 2.5],
            constrain='domain',
            fixedrange=False
        ),
        height=580,  # Increased height
        width=800,
        dragmode='pan',
        modebar=dict(
            bgcolor='rgba(0,0,0,0)',
            color=COLORS['text'],
            activecolor=COLORS['primary'],
            orientation='v',
            add=['zoom', 'pan', 'zoomIn', 'zoomOut', 'resetScale2d', 'toImage'],
            remove=['autoScale2d']
        )
    )
    
    return fig

def create_drilldown_view(df, selected_item):
    """Create a detailed view for the selected project or budget line"""
    if selected_item in df['Project'].unique():
        # Project drilldown
        filtered_df = df[df['Project'] == selected_item]
        title = f"Breakdown for Project: {selected_item}"
        group_by = 'Budget Line'
    else:
        # Budget line drilldown
        filtered_df = df[df['Budget Line'] == selected_item]
        title = f"Breakdown for Budget Line: {selected_item}"
        group_by = 'Project'
    
    fig = px.pie(
        filtered_df,
        values='Amount',
        names=group_by,
        title=title,
        color_discrete_sequence=px.colors.sequential.Blues_r,
        height=300  # Increased height
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text'],
        title_font_color=COLORS['text'],
        title_font_size=14,  # Smaller title font
        margin=dict(l=10, r=10, t=30, b=10),  # Reduced margins
        font=dict(size=10)  # Smaller font
    )
    
    return fig

def generate_pdf_report(df, charts=None):
    """Generate an infographic-style PDF report with visualizations and data summaries
    
    Args:
        df: The DataFrame with budget data
        charts: Dictionary with pre-generated charts
        
    Returns:
        BytesIO: A buffer with the PDF report
    """
    buffer = io.BytesIO()
    
    # Create the PDF document with landscape orientation (swap A4 width and height)
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=(A4[1], A4[0]),  # Landscape orientation
        topMargin=0.3*inch,
        leftMargin=0.3*inch,
        rightMargin=0.3*inch,
        bottomMargin=0.3*inch
    )
    
    # Get sample stylesheet and create custom styles with unique names
    styles = getSampleStyleSheet()
    
    # More compact styles for landscape layout
    title_style = ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=0.1*inch,
        textColor=colors.HexColor('#4A90E2')
    )
    
    subtitle_style = ParagraphStyle(
        name='CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=10,
        alignment=TA_CENTER,
        spaceAfter=0.1*inch,
        textColor=colors.HexColor('#7B68EE')
    )
    
    section_heading_style = ParagraphStyle(
        name='CustomSectionHeading',
        parent=styles['Heading2'],
        fontSize=10,
        alignment=TA_LEFT,
        spaceBefore=0.1*inch,
        spaceAfter=0.05*inch,
        textColor=colors.HexColor('#4A90E2')
    )
    
    normal_style = ParagraphStyle(
        name='CustomNormal',
        parent=styles['Normal'],
        fontSize=8,
        spaceBefore=0.05*inch,
        spaceAfter=0.05*inch
    )
    
    # Add the custom styles to the stylesheet
    styles.add(title_style)
    styles.add(subtitle_style)
    styles.add(section_heading_style)
    styles.add(normal_style)
    
    # Build the document content
    content = []
    
    # Create a table for header with title and date
    header_data = [
        [Paragraph("Budget Visualization Report", title_style)],
        [Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", subtitle_style)]
    ]
    header_table = Table(header_data, colWidths=[8*inch])
    header_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('VALIGN', (0, 0), (0, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (0, -1), 2),
        ('TOPPADDING', (0, 0), (0, -1), 2),
    ]))
    content.append(header_table)
    content.append(Spacer(1, 0.1*inch))
    
    # Calculate metrics
    total_expenses = df['Amount'].sum()
    project_count = len(df['Project'].unique())
    avg_per_project = total_expenses / project_count
    
    df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M')
    monthly_expenses = df.groupby('Month')['Amount'].sum()
    
    if len(monthly_expenses) > 1:
        current_month = monthly_expenses.index[-1]
        prev_month = monthly_expenses.index[-2]
        current_month_amount = monthly_expenses[current_month]
        prev_month_amount = monthly_expenses[prev_month]
        mom_change = ((current_month_amount - prev_month_amount) / prev_month_amount) * 100
        mom_change_formatted = f"{mom_change:+.1f}%"
    else:
        mom_change_formatted = "N/A"
    
    # Create a two-column layout for all content
    col_width = 4.1*inch
    
    # Create the main table structure that will hold all our content in a grid
    grid_data = [[]]
    
    # Left column content
    left_column = []
    
    # Metrics section
    left_column.append(Paragraph("Budget Summary", section_heading_style))
    
    metrics_data = [
        ["Total Expenses", f"${total_expenses:,.2f}"],
        ["Projects", str(project_count)],
        ["Avg/Project", f"${avg_per_project:,.2f}"],
        ["Month-over-Month", mom_change_formatted]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[1.5*inch, 2.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#262B32')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
        ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#1E2126')),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#4A90E2')),
    ]))
    left_column.append(metrics_table)
    left_column.append(Spacer(1, 0.1*inch))
    
    # Project expenses section
    left_column.append(Paragraph("Project Expenses", section_heading_style))
    
    # Get top 5 projects by expense
    expenses_by_project = df.groupby('Project')['Amount'].sum().reset_index().sort_values('Amount', ascending=False).head(7)
    
    project_data = [["Project", "Total ($)"]]
    for _, row in expenses_by_project.iterrows():
        project_data.append([row['Project'], f"${row['Amount']:,.2f}"])
    
    project_table = Table(project_data, colWidths=[3*inch, 1*inch])
    project_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90E2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#1E2126'), colors.HexColor('#262B32')]),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
    ]))
    left_column.append(project_table)
    
    # Monthly trend section (in left column)
    left_column.append(Spacer(1, 0.1*inch))
    left_column.append(Paragraph("Monthly Expense Trend", section_heading_style))
    
    monthly_df = df.groupby(pd.to_datetime(df['Date']).dt.strftime('%Y-%m'))['Amount'].sum().reset_index()
    monthly_df.columns = ['Month', 'Amount']
    monthly_df = monthly_df.sort_values('Month').tail(6)  # Show only last 6 months
    
    monthly_data = [["Month", "Amount ($)"]]
    for _, row in monthly_df.iterrows():
        monthly_data.append([row['Month'], f"${row['Amount']:,.2f}"])
    
    monthly_table = Table(monthly_data, colWidths=[2*inch, 2*inch])
    monthly_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90E2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#1E2126'), colors.HexColor('#262B32')]),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
    ]))
    left_column.append(monthly_table)
    
    # Right column content
    right_column = []
    
    # Budget breakdown
    right_column.append(Paragraph("Budget Breakdown", section_heading_style))
    
    budget_data = [["Budget Line", "Amount ($)"]]
    budget_breakdown = df.groupby('Budget Line')['Amount'].sum().reset_index().sort_values('Amount', ascending=False).head(7)
    for _, row in budget_breakdown.iterrows():
        budget_data.append([row['Budget Line'], f"${row['Amount']:,.2f}"])
    
    budget_table = Table(budget_data, colWidths=[3*inch, 1*inch])
    budget_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7B68EE')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#1E2126'), colors.HexColor('#262B32')]),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
    ]))
    right_column.append(budget_table)
    right_column.append(Spacer(1, 0.1*inch))
    
    # Project-budget connections
    right_column.append(Paragraph("Project-Budget Connections", section_heading_style))
    
    connections_data = [["Project", "Budget Line", "Amount ($)"]]
    connections = df.groupby(['Project', 'Budget Line'])['Amount'].sum().reset_index().sort_values('Amount', ascending=False)
    
    # Limit to top 12 connections for space
    for _, row in connections.head(12).iterrows():
        connections_data.append([
            row['Project'][:15] + ('...' if len(row['Project']) > 15 else ''),
            row['Budget Line'][:15] + ('...' if len(row['Budget Line']) > 15 else ''),
            f"${row['Amount']:,.2f}"
        ])
    
    connections_table = Table(connections_data, colWidths=[1.7*inch, 1.7*inch, 0.7*inch])
    connections_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7B68EE')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#1E2126'), colors.HexColor('#262B32')]),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
    ]))
    right_column.append(connections_table)
    
    # Combine left and right columns
    grid_data = [[left_column, right_column]]
    
    # Create a table with two columns for the layout
    main_table = Table(grid_data, colWidths=[col_width, col_width])
    main_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
    ]))
    content.append(main_table)
    
    # Add footer
    def add_footer(canvas, doc):
        canvas.saveState()
        footer_text = f"Budget Visualization Dashboard • Generated on {datetime.now().strftime('%B %d, %Y')}"
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#7B68EE'))
        canvas.drawString(0.3*inch, 0.3*inch, footer_text)
        # Add a decorative line
        canvas.setStrokeColor(colors.HexColor('#4A90E2'))
        canvas.line(0.3*inch, 0.5*inch, A4[1] - 0.3*inch, 0.5*inch)
        canvas.restoreState()
    
    # Build the PDF
    doc.build(content, onFirstPage=add_footer, onLaterPages=add_footer)
    
    # Get the value from the buffer
    buffer.seek(0)
    return buffer

# Update export button functionality
def export_data_button(df):
    export_type = st.radio(
        "Export format:",
        ["PDF Report", "CSV Data"],
        horizontal=True,
        key="export_format"
    )
    
    if export_type == "CSV Data":
        # For CSV, use native Streamlit download button
        if st.button("Generate CSV", key="export_button", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="budget_analysis.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_csv"
            )
    else:  # PDF Report
        if st.button("Generate PDF Report", key="export_button", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                # Generate PDF
                pdf_buffer = generate_pdf_report(df)
                # Create download button with the PDF data
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"budget_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf"
                )
def create_network_tree_data(df):
    """Convert DataFrame to hierarchical structure for network tree,
       adding type, id, and value properties for D3.js visualization."""
    project_data = []
    total_budget = df['Amount'].sum() # Calculate grand total

    for project_name, project_group in df.groupby('Project'):
        budget_lines = []
        project_total_amount = float(project_group['Amount'].sum()) # Project total

        for budget_line, budget_group in project_group.groupby('Budget Line'):
            amount = float(budget_group['Amount'].sum())
            node_id = f"{project_name}_{budget_line}".replace(" ", "_").replace("/", "_")
            budget_lines.append({
                'name': budget_line,
                'type': 'expense',
                'value': amount,     # <<< Value for expense node
                'id': node_id
            })

        project_id = f"{project_name}".replace(" ", "_").replace("/", "_")
        project_data.append({
            'name': project_name,
            'type': 'project',
            'value': project_total_amount, # <<< Value for project node
            'id': project_id,
            'children': budget_lines
        })

    return {
        'name': 'Budget Overview',
        'type': 'root',
        'id': 'root',
        'value': float(total_budget), # <<< Add total value to root node data
        'children': project_data
    }

def main():
    st.title("Budget Visualization Dashboard")
    
    # Load cached data if available
    cached_df, cached_filename = load_from_cache()
    
    # Initialize AI helper
    if 'ai_helper' not in st.session_state:
        st.session_state.ai_helper = BudgetAI()
    
    # If data is loaded, update the AI helper with the data
    if cached_df is not None:
        st.session_state.ai_helper.set_data(cached_df)
    
    # Sidebar for file upload and project selection
    with st.sidebar:
        st.header("Data Input")
        upload_method = st.radio(
            "Choose upload method:",
            ["Local File", "OneDrive", "SharePoint"],
            horizontal=True  # Make radio buttons horizontal to save space
        )
        
        if upload_method == "Local File":
            uploaded_file = st.file_uploader("Upload Excel or CSV file", type=['xlsx', 'xls', 'csv'])
            
            # Show current file and clear button if data is cached
            if cached_df is not None:
                st.info(f"Currently using: {cached_filename}")
                if st.button("Clear cached data"):
                    clear_cache()
                    st.rerun()
            
            # Load data if new file is uploaded
            if uploaded_file:
                if cached_filename != uploaded_file.name:
                    df = load_excel_data(uploaded_file)
                    if df is not None:
                        save_to_cache(df, uploaded_file.name)
                        # Store DataFrame in session state for access by all tabs
                        st.session_state.df = df
                        st.rerun()
                df = cached_df
            else:
                df = cached_df  # Use cached data if no new file is uploaded
            
            # If we have cached data but haven't set the session state, do it now
            if df is not None and 'df' not in st.session_state:
                st.session_state.df = df
        
        elif upload_method in ["OneDrive", "SharePoint"]:
            st.info("Microsoft integration coming soon!")
            uploaded_file = None
            df = cached_df  # Use cached data if available
        

        
        # Project selection (hidden from user but used for internal logic)
        if df is not None:
            projects = list(df['Project'].unique())  # Convert numpy array to list
            selected_projects = projects  # Use all projects by default
            
            # Add export functionality to sidebar
            st.markdown("<hr style='margin: 0.5rem 0'>", unsafe_allow_html=True)
            st.markdown("### Export Data")
            export_data_button(df)
        else:
            df = None
            selected_projects = []

    # Main content area
    if df is not None:
        # Calculate month-over-month metrics
        df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M')
        monthly_expenses = df.groupby('Month')['Amount'].sum()
        
        if len(monthly_expenses) > 1:
            current_month = monthly_expenses.index[-1]
            prev_month = monthly_expenses.index[-2]
            current_month_amount = monthly_expenses[current_month]
            prev_month_amount = monthly_expenses[prev_month]
            mom_change = ((current_month_amount - prev_month_amount) / prev_month_amount) * 100
            
            # Calculate average monthly expense
            avg_monthly_expense = monthly_expenses.mean()
            
            # Calculate month with highest expenses
            peak_month = monthly_expenses.idxmax()
            peak_amount = monthly_expenses[peak_month]
        else:
            mom_change = 0
            avg_monthly_expense = monthly_expenses.iloc[0] if len(monthly_expenses) > 0 else 0
            peak_month = monthly_expenses.index[0] if len(monthly_expenses) > 0 else None
            peak_amount = monthly_expenses.iloc[0] if len(monthly_expenses) > 0 else 0
        
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        
        # Create metrics in two rows with consistent spacing
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        
        with row1_col1:
            total_expenses = df['Amount'].sum()
            st.metric(
                "Total Expenses",
                f"${total_expenses:,.2f}",
                help="Total sum of all expenses across all projects"
            )
        
        with row1_col2:
            st.metric(
                "Month-over-Month Change",
                f"{mom_change:+.1f}%",
                delta=f"{mom_change:+.1f}%",
                delta_color="inverse",
                help="Percentage change in expenses compared to previous month"
            )
        
        with row1_col3:
            st.metric(
                "Average Monthly Expense",
                f"${avg_monthly_expense:,.2f}",
                help="Average expense per month"
            )
        
        # Add minimal spacing between rows
        st.markdown('<div style="height: 0.25rem;"></div>', unsafe_allow_html=True)
        
        # Second row of metrics
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        
        with row2_col1:
            st.metric(
                "Total Projects",
                len(projects),
                help="Number of active projects"
            )
        
        with row2_col2:
            st.metric(
                "Peak Month",
                f"{peak_month}",
                delta=f"${peak_amount:,.2f}",
                help="Month with highest expenses and the corresponding amount"
            )
        
        with row2_col3:
            avg_project_expense = total_expenses / len(projects)
            st.metric(
                "Avg Expense per Project",
                f"${avg_project_expense:,.2f}",
                help="Average expense per project"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Project Overview",
            "Detailed Analysis",
            "Budget Assistant",
            "Network Tree",
            "Vendors",
            "Forecast",
            "Staff"
        ])
        
        with tab1:
            # Create a single dropdown with all options
            view_options = ["All Selected Projects"] + list(df['Project'].unique())
            selected_view = st.selectbox(
                "Select view:",
                view_options,
                key="view_selector",
                help="View all selected projects or analyze an individual project"
            )
            
            # Create two columns for the charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Create and display the project expense chart with the selected view
                expense_chart = create_project_expense_chart(df, selected_view)
                st.plotly_chart(expense_chart, use_container_width=True, config={'displayModeBar': False})
            
            with chart_col2:
                if selected_projects:
                    # Show either individual project or all selected projects
                    if selected_view == "All Selected Projects":
                        projects_to_show = selected_projects
                    else:
                        projects_to_show = [selected_view]
                    
                    st.plotly_chart(create_budget_impact_chart(df, projects_to_show), use_container_width=True)
                else:
                    st.info("Please select projects to analyze in the sidebar")

            # Add drilldown functionality moved from tab2_col2
            st.subheader("Detailed Breakdown")
            
            # Set the default selected item based on the main view selection
            default_item = selected_view if selected_view != "All Selected Projects" else None
            
            # If default_item is None (All Selected Projects), show dropdown for selection
            if default_item is None:
                all_items = list(df['Project'].unique()) + list(df['Budget Line'].unique())
                selected_item = st.selectbox(
                    "Select for detailed breakdown:",
                    all_items
                )
            else:
                # Use the selected view directly
                selected_item = default_item
                st.info(f"Showing detailed breakdown for: {selected_item}")
            
            if selected_item:
                drilldown_fig = create_drilldown_view(df, selected_item)
                st.plotly_chart(drilldown_fig, use_container_width=True)
                
                # Show compact detailed table
                st.subheader("Detailed Transactions")
                if selected_item in df['Project'].unique():
                    detailed_df = df[df['Project'] == selected_item]
                else:
                    detailed_df = df[df['Budget Line'] == selected_item]
                st.dataframe(
                    detailed_df.sort_values('Date', ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    height=200  # Increased height for table
                )

        with tab2:
            st.subheader("Detailed Expense Breakdown")
            
            # Add filter and search section
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                # Date range filter
                min_date = pd.to_datetime(df['Date']).min().date()
                max_date = pd.to_datetime(df['Date']).max().date()
                date_range = st.date_input(
                    "Date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
                
                # Amount range filter
                min_amount = float(df['Amount'].min())
                max_amount = float(df['Amount'].max())
                amount_range = st.slider(
                    "Amount range ($)",
                    min_value=min_amount,
                    max_value=max_amount,
                    value=(min_amount, max_amount),
                    format="$%.2f",
                )
            
            with filter_col2:
                # Project filter (multiselect)
                all_projects = sorted(df['Project'].unique())
                selected_project_filter = st.multiselect(
                    "Filter by Project",
                    options=all_projects,
                    default=[],
                    placeholder="All Projects"
                )
                
                # Budget Line filter (multiselect)
                all_budget_lines = sorted(df['Budget Line'].unique())
                selected_budget_filter = st.multiselect(
                    "Filter by Budget Line",
                    options=all_budget_lines,
                    default=[],
                    placeholder="All Budget Lines"
                )
            
            # Search box
            search_text = st.text_input("Search in Expense Description", placeholder="Type to search...")
            
            # Apply filters
            filtered_df = df.copy()
            
            # Date filter
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (pd.to_datetime(filtered_df['Date']).dt.date >= start_date) & 
                    (pd.to_datetime(filtered_df['Date']).dt.date <= end_date)
                ]
            
            # Amount filter
            filtered_df = filtered_df[
                (filtered_df['Amount'] >= amount_range[0]) & 
                (filtered_df['Amount'] <= amount_range[1])
            ]
            
            # Project filter
            if selected_project_filter:
                filtered_df = filtered_df[filtered_df['Project'].isin(selected_project_filter)]
            
            # Budget Line filter
            if selected_budget_filter:
                filtered_df = filtered_df[filtered_df['Budget Line'].isin(selected_budget_filter)]
            
            # Text search
            if search_text:
                filtered_df = filtered_df[filtered_df['Expense'].str.contains(search_text, case=False)]
            
            # Add filter summary
            active_filters = []
            if len(date_range) == 2 and (date_range[0] > min_date or date_range[1] < max_date):
                active_filters.append(f"Date: {date_range[0]} to {date_range[1]}")
            if amount_range[0] > min_amount or amount_range[1] < max_amount:
                active_filters.append(f"Amount: ${amount_range[0]:,.2f} to ${amount_range[1]:,.2f}")
            if selected_project_filter:
                active_filters.append(f"Projects: {', '.join(selected_project_filter)}")
            if selected_budget_filter:
                active_filters.append(f"Budget Lines: {', '.join(selected_budget_filter)}")
            if search_text:
                active_filters.append(f"Search: '{search_text}'")
            
            if active_filters:
                st.markdown(f"**Active filters:** {' | '.join(active_filters)}")
            
            # Show filter results count
            st.markdown(f"**Showing {len(filtered_df)} of {len(df)} records**")
            
            # Display filtered data
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True,
                height=None  # Remove fixed height to use container height
            )
            
        with tab3:
            st.subheader("Budget Assistant")
            
            # Create two columns - a narrow one for examples and a wider one for chat
            ai_col1, ai_col2 = st.columns([1, 3])
            
            with ai_col1:
                st.markdown("### Example Questions")
                
                # Create categories for questions
                example_categories = {
                    "Project Analysis": [
                        "What is the total expense for each project?",
                        "Which project has the highest spending?",
                        "Calculate the average expense per project"
                    ],
                    "Budget Categories": [
                        "Show a breakdown of expenses by budget line",
                        "Generate a pie chart of expense categories",
                        "Show all transactions for Marketing"
                    ],
                    "Time Analysis": [
                        "What was the monthly spend trend over the past 6 months?",
                        "How much did we spend in February 2024?",
                        "What's the month-over-month change in spending?"
                    ],
                    "Insights": [
                        "Identify any months where expenses exceeded $10,000",
                        "What is the average monthly spend?",
                        "Highlight any negative or zero expense entries"
                    ]
                }
                
                # Initialize session state for storing the selected question
                if "selected_example" not in st.session_state:
                    st.session_state.selected_example = None
                
                # Create expandable sections for each category with clickable buttons
                for category, questions in example_categories.items():
                    with st.expander(category, expanded=False):
                        for question in questions:
                            if st.button(question, key=f"btn_{question}", use_container_width=True):
                                st.session_state.selected_example = question
                                # Add to chat history if not already in the chat
                                if "ai_chat_history" in st.session_state:
                                    # Check if this exact question is not the last question asked
                                    if (not st.session_state.ai_chat_history or 
                                        st.session_state.ai_chat_history[-1]["role"] != "user" or
                                        st.session_state.ai_chat_history[-1]["content"] != question):
                                        st.session_state.ai_chat_history.append({"role": "user", "content": question})
                                        # Will be processed in the main chat area
            
            with ai_col2:
                # Create a container for the chat messages with auto-scroll
                chat_container = st.container()
                
                # Custom CSS for better chat appearance
                st.markdown("""
                <style>
                .stChatMessage {
                    padding: 10px 15px;
                    border-radius: 15px;
                    margin-bottom: 10px;
                    max-width: 90%;
                }
                .stChatMessage[data-testid="stChatMessageUser"] {
                    background-color: rgba(74, 144, 226, 0.2);
                    border-top-right-radius: 5px;
                    margin-left: auto;
                }
                .stChatMessage[data-testid="stChatMessageAssistant"] {
                    background-color: rgba(120, 120, 120, 0.1);
                    border-top-left-radius: 5px;
                    margin-right: auto;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Add a placeholder for auto-scrolling at the bottom
                scroll_to = st.empty()
                
                # Create the chat input at the bottom of the chat interface
                user_query = st.chat_input("Ask a question about the budget data...")
                
                # Initialize chat history if it doesn't exist
                if "ai_chat_history" not in st.session_state:
                    st.session_state.ai_chat_history = []
                
                # When either the user inputs a query or selects an example
                query_to_process = None
                
                if user_query:
                    query_to_process = user_query
                    # Add user message to history
                    st.session_state.ai_chat_history.append({"role": "user", "content": user_query})
                
                # Check if we have a selected example from the buttons
                elif st.session_state.selected_example and len(st.session_state.ai_chat_history) > 0:
                    # If the last message is from user and matches the selected example, process it
                    last_msg = st.session_state.ai_chat_history[-1]
                    if last_msg["role"] == "user" and last_msg["content"] == st.session_state.selected_example:
                        query_to_process = st.session_state.selected_example
                        # Reset so we don't process it again
                        st.session_state.selected_example = None
                
                # Process query if we have one
                if query_to_process:
                    # Get AI response
                    with st.spinner("Analyzing budget data..."):
                        response = st.session_state.ai_helper.process_query(query_to_process)
                    
                    # Handle different types of responses
                    if isinstance(response, dict) and "type" in response:
                        # For chart responses
                        if response["type"] == "chart":
                            content = response.get("text", "Here's what I found:")
                            st.session_state.ai_chat_history.append({"role": "assistant", "content": content, "chart": response["figure"]})
                        
                        # For instruction responses (like export)
                        elif response["type"] == "instruction" and response["action"] == "export_pdf":
                            st.session_state.ai_chat_history.append({"role": "assistant", "content": "I'll generate a PDF report for you. You can download it below."})
                            # Set a flag to trigger PDF export
                            st.session_state.trigger_export = True
                    else:
                        # For regular text responses
                        st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
                
                # Display chat history in the container
                with chat_container:
                    # If the chat is empty, show a welcome message
                    if not st.session_state.ai_chat_history:
                        st.markdown("""
                        ### 👋 Welcome to Budget Assistant!
                        
                        I can help you analyze your budget data through natural language questions. Try asking me about:
                        - Total expenses by project
                        - Monthly spending trends
                        - Project comparisons
                        - Budget category breakdowns
                        
                        Select an example from the left panel or type your question below!
                        """)
                    else:
                        # Display each message in the chat history
                        for message in st.session_state.ai_chat_history:
                            if message["role"] == "user":
                                with st.chat_message("user"):
                                    st.write(message["content"])
                            else:
                                with st.chat_message("assistant"):
                                    st.write(message["content"])
                                    
                                    # If the message has a chart, display it
                                    if "chart" in message:
                                        st.plotly_chart(message["chart"], use_container_width=True)
                
                # Auto-scroll to the bottom of the chat
                if st.session_state.ai_chat_history:
                    scroll_to.markdown('<div id="end-of-chat"></div>', unsafe_allow_html=True)
                    st.markdown(
                        """
                        <script>
                            function scrollToBottom() {
                                var endOfChat = document.getElementById('end-of-chat');
                                if (endOfChat) {
                                    endOfChat.scrollIntoView();
                                }
                            }
                            scrollToBottom();
                            // Also try to scroll after a short delay to ensure all content is loaded
                            setTimeout(scrollToBottom, 500);
                        </script>
                        """,
                        unsafe_allow_html=True
                    )
                    
                # Handle PDF export instruction if triggered
                if df is not None and "trigger_export" in st.session_state and st.session_state.trigger_export:
                    with st.spinner("Generating PDF report..."):
                        pdf_buffer = generate_pdf_report(df)
                        st.download_button(
                            label="⬇️ Download AI Generated PDF Report",
                            data=pdf_buffer,
                            file_name=f"budget_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="ai_download_pdf"
                        )
                    st.session_state.trigger_export = False
                    
                # Add a clear chat button
                if st.session_state.ai_chat_history:
                    if st.button("Clear Chat", key="clear_chat", use_container_width=True):
                        st.session_state.ai_chat_history = []
                        st.rerun()
                    
        with tab4:
            st.subheader("Network Tree Visualization")
            
            # Add a button to load the visualization
            if 'show_network_tree' not in st.session_state:
                st.session_state.show_network_tree = False
            
            if not st.session_state.show_network_tree:
                st.info("Click the button below to load the network tree visualization.")
                if st.button("Load Network Tree", use_container_width=True):
                    st.session_state.show_network_tree = True
                    st.rerun()
            else:
                # Create the network tree data
                tree_data = create_network_tree_data(df)
                
                # Convert the data to JSON string
                tree_data_json = json.dumps(tree_data)
                
                # Read the HTML template
                with open('network_tree.html', 'r') as f:
                    html_template = f.read()
                
                # Replace the placeholder with actual data
                html_content = html_template.replace('const data = {};', f'const data = {tree_data_json};')
                
                # Create a custom component for the network tree
                components.html(html_content, height=650)
                
                # Add a button to reload the visualization
                if st.button("Reload Network Tree", use_container_width=True):
                    st.session_state.show_network_tree = False
                    st.rerun()
                    
        with tab5:
            st.subheader("Vendor Analysis")
            
            # Get vendor data
            if 'Vendor Name' in df.columns:
                vendors_df = df[['Vendor Name', 'Project', 'Budget Line', 'Amount', 'Date']]
                
                # Calculate vendor metrics
                vendor_metrics = vendors_df.groupby('Vendor Name').agg({
                    'Amount': ['sum', 'mean', 'count'],
                    'Project': 'nunique',
                    'Budget Line': 'nunique',
                    'Date': ['min', 'max']
                }).reset_index()
                
                # Flatten multi-index columns
                vendor_metrics.columns = ['Vendor Name', 'Total Amount', 'Average Amount', 
                                         'Number of Transactions', 'Unique Projects', 
                                         'Unique Budget Lines', 'First Engagement', 'Last Engagement']
                
                # Filter out empty vendor names
                vendor_metrics = vendor_metrics[vendor_metrics['Vendor Name'] != ""]
                
                # Add search and filter functionality
                st.subheader("Vendor Search & Filters")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Search by vendor name
                    vendor_search = st.text_input("Search Vendors", placeholder="Enter vendor name...")
                    
                    # Amount range filter
                    if not vendor_metrics.empty:
                        min_vendor_amount = float(vendor_metrics['Total Amount'].min())
                        max_vendor_amount = float(vendor_metrics['Total Amount'].max())
                        vendor_amount_range = st.slider(
                            "Total Amount Range ($)",
                            min_value=min_vendor_amount,
                            max_value=max_vendor_amount,
                            value=(min_vendor_amount, max_vendor_amount),
                            format="$%.2f",
                        )
                    
                with col2:
                    # Project filter for vendors
                    vendor_projects = st.multiselect(
                        "Filter by Projects",
                        options=sorted(df['Project'].unique()),
                        default=[],
                        placeholder="All Projects"
                    )
                    
                    # Budget line filter for vendors
                    vendor_budget_lines = st.multiselect(
                        "Filter by Budget Lines",
                        options=sorted(df['Budget Line'].unique()),
                        default=[],
                        placeholder="All Budget Lines"
                    )
                
                # Apply filters to base dataframe first
                filtered_vendors_df = vendors_df.copy()
                
                if vendor_search:
                    filtered_vendors_df = filtered_vendors_df[
                        filtered_vendors_df['Vendor Name'].str.contains(vendor_search, case=False, na=False)
                    ]
                
                if vendor_projects:
                    filtered_vendors_df = filtered_vendors_df[
                        filtered_vendors_df['Project'].isin(vendor_projects)
                    ]
                
                if vendor_budget_lines:
                    filtered_vendors_df = filtered_vendors_df[
                        filtered_vendors_df['Budget Line'].isin(vendor_budget_lines)
                    ]
                
                # Get unique vendor names from filtered data
                filtered_vendor_names = filtered_vendors_df['Vendor Name'].unique()
                
                # Filter vendor metrics
                filtered_vendor_metrics = vendor_metrics[
                    vendor_metrics['Vendor Name'].isin(filtered_vendor_names)
                ]
                
                if not vendor_metrics.empty:
                    filtered_vendor_metrics = filtered_vendor_metrics[
                        (filtered_vendor_metrics['Total Amount'] >= vendor_amount_range[0]) &
                        (filtered_vendor_metrics['Total Amount'] <= vendor_amount_range[1])
                    ]
                
                # Display filter summary
                active_filters = []
                if vendor_search:
                    active_filters.append(f"Search: '{vendor_search}'")
                if vendor_projects:
                    active_filters.append(f"Projects: {', '.join(vendor_projects)}")
                if vendor_budget_lines:
                    active_filters.append(f"Budget Lines: {', '.join(vendor_budget_lines)}")
                if not vendor_metrics.empty and (vendor_amount_range[0] > min_vendor_amount or vendor_amount_range[1] < max_vendor_amount):
                    active_filters.append(f"Amount: ${vendor_amount_range[0]:,.2f} to ${vendor_amount_range[1]:,.2f}")
                
                if active_filters:
                    st.markdown(f"**Active filters:** {' | '.join(active_filters)}")
                
                # Display vendor metrics
                st.subheader(f"Vendor List ({len(filtered_vendor_metrics)} vendors)")
                
                if filtered_vendor_metrics.empty:
                    st.info("No vendors match the selected filters.")
                else:
                    # Sort by total amount by default
                    sorted_vendor_metrics = filtered_vendor_metrics.sort_values('Total Amount', ascending=False)
                    
                    # Format money columns
                    display_df = sorted_vendor_metrics.copy()
                    display_df['Total Amount'] = display_df['Total Amount'].apply(lambda x: f"${x:,.2f}")
                    display_df['Average Amount'] = display_df['Average Amount'].apply(lambda x: f"${x:,.2f}")
                    
                    # Display the table
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "First Engagement": st.column_config.DateColumn("First Engagement", format="MMM D, YYYY"),
                            "Last Engagement": st.column_config.DateColumn("Last Engagement", format="MMM D, YYYY"),
                        }
                    )
                
                # Add vendor transaction details section
                st.subheader("Vendor Transaction Details")
                
                # Create dropdown for selecting a specific vendor
                if len(filtered_vendor_names) > 0 and any(v != "" for v in filtered_vendor_names):
                    selected_vendor = st.selectbox(
                        "Select a vendor to view detailed transactions:",
                        options=[v for v in filtered_vendor_names if v != ""],
                        index=0 if len([v for v in filtered_vendor_names if v != ""]) > 0 else None
                    )
                    
                    if selected_vendor:
                        # Get transactions for selected vendor
                        vendor_transactions = vendors_df[vendors_df['Vendor Name'] == selected_vendor]
                        
                        # Display summary metrics for the selected vendor
                        vendor_summary = vendor_metrics[vendor_metrics['Vendor Name'] == selected_vendor].iloc[0]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Amount", f"${vendor_summary['Total Amount']:,.2f}")
                        with col2:
                            st.metric("# Transactions", int(vendor_summary['Number of Transactions']))
                        with col3:
                            st.metric("Unique Projects", int(vendor_summary['Unique Projects']))
                        with col4:
                            st.metric("Avg. Transaction", f"${vendor_summary['Average Amount']:,.2f}")
                        
                        # Show transaction history
                        st.subheader(f"Transaction History for {selected_vendor}")
                        st.dataframe(
                            vendor_transactions.sort_values('Date', ascending=False),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
                                "Amount": st.column_config.NumberColumn("Amount", format="$%.2f")
                            }
                        )
                        
                        # Add visualization of vendor transactions over time
                        if len(vendor_transactions) > 1:
                            st.subheader("Transaction History Chart")
                            
                            time_chart_type = st.radio(
                                "Chart type:",
                                ["Cumulative", "Individual Transactions"],
                                horizontal=True
                            )
                            
                            # Sort by date for time series
                            time_df = vendor_transactions.sort_values('Date')
                            
                            if time_chart_type == "Cumulative":
                                # Create cumulative chart
                                cumulative_df = time_df.copy()
                                cumulative_df['Cumulative Amount'] = cumulative_df['Amount'].cumsum()
                                
                                fig = px.line(
                                    cumulative_df,
                                    x='Date',
                                    y='Cumulative Amount',
                                    title=f"Cumulative Spending with {selected_vendor}",
                                    height=400
                                )
                                
                                # Improve styling
                                fig.update_layout(
                                    xaxis_title="Date",
                                    yaxis_title="Cumulative Amount ($)",
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='#333333'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Create bar chart for individual transactions
                                fig = px.bar(
                                    time_df,
                                    x='Date',
                                    y='Amount',
                                    color='Project',
                                    title=f"Transaction History with {selected_vendor}",
                                    height=400
                                )
                                
                                # Improve styling
                                fig.update_layout(
                                    xaxis_title="Date",
                                    yaxis_title="Amount ($)",
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='#333333'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No vendors found with the current filters.")
            else:
                st.warning("The 'Vendor Name' column is missing in the data. Please ensure your data includes vendor information.")

        with tab6:
            st.header("Fiscal Year Budget Forecasting")
            
            if 'df' in st.session_state and st.session_state.df is not None:
                # Initialize forecaster
                fiscal_forecaster = FiscalYearForecaster(st.session_state.df)
                
                # Create columns for filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Project filter
                    project = st.selectbox(
                        "Select Project",
                        ["All"] + sorted(st.session_state.df['Project'].unique().tolist()),
                        key="fiscal_project"
                    )
                    
                with col2:
                    # Budget line filter
                    budget_line = st.selectbox(
                        "Select Budget Line",
                        ["All"] + sorted(st.session_state.df['Budget Line'].unique().tolist()),
                        key="fiscal_budget_line"
                    )
                    
                with col3:
                    # Period type selection
                    period_type = st.selectbox(
                        "Forecast by",
                        ["quarter", "annual"],
                        format_func=lambda x: "Quarterly" if x == "quarter" else "Annually",
                        key="fiscal_period_type"
                    )
                
                # Convert "All" to None for filtering
                project = None if project == "All" else project
                budget_line = None if budget_line == "All" else budget_line
                
                # Forecast parameters
                st.subheader("Fiscal Year Parameters")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if period_type == "quarter":
                        periods_ahead = st.slider(
                            "Quarters to Forecast",
                            min_value=1,
                            max_value=12,
                            value=4,
                            help="Number of fiscal quarters to forecast into the future",
                            key="fiscal_periods_ahead"
                        )
                    else:
                        periods_ahead = st.slider(
                            "Years to Forecast",
                            min_value=1,
                            max_value=5,
                            value=3,
                            help="Number of fiscal years to forecast into the future",
                            key="fiscal_periods_ahead"
                        )
                
                with col2:
                    fiscal_year_start_month = st.selectbox(
                        "Fiscal Year Start Month",
                        options=list(range(1, 13)),
                        format_func=lambda x: datetime(2023, x, 1).strftime("%B"),
                        index=6,  # Default to July (7) for Australian standard
                        help="Month when the fiscal year begins (Australian standard is July)",
                        key="fiscal_year_start"
                    )
                
                with col3:
                    confidence_level = st.slider(
                        "Confidence Level (%)",
                        min_value=80,
                        max_value=99,
                        value=95,
                        help="Confidence level for the forecast interval",
                        key="fiscal_confidence"
                    )
                
                # Budget Cut Risk Analysis section
                st.subheader("Budget Cut Risk Analysis")
                
                # Enable budget cut analysis toggle
                enable_budget_cut = st.checkbox("Enable Budget Cut Analysis", value=False, key="enable_budget_cut")
                
                budget_cut_percentage = 0
                budget_cut_amount = 0
                threshold_percentage = 0
                
                if enable_budget_cut:
                    # Create columns for budget cut parameters
                    cut_col1, cut_col2 = st.columns(2)
                    
                    with cut_col1:
                        # Option to specify cut by percentage or absolute amount
                        cut_type = st.radio(
                            "Budget Cut Type",
                            options=["Percentage", "Fixed Amount"],
                            horizontal=True,
                            key="cut_type"
                        )
                        
                        if cut_type == "Percentage":
                            budget_cut_percentage = st.slider(
                                "Budget Cut Percentage (%)",
                                min_value=0,
                                max_value=50,
                                value=10,
                                help="Percentage by which the budget will be reduced",
                                key="budget_cut_percentage"
                            )
                        else:
                            # For fixed amount, we need to calculate a reasonable max value
                            # based on the forecast data (will be set after generating initial forecast)
                            budget_cut_amount = st.number_input(
                                "Budget Cut Amount ($)",
                                min_value=0,
                                value=10000,
                                step=1000,
                                format="%d",
                                help="Fixed amount by which the budget will be reduced",
                                key="budget_cut_amount"
                            )
                    
                    with cut_col2:
                        # Threshold for determining delivery risk
                        threshold_percentage = st.slider(
                            "Delivery Risk Threshold (%)",
                            min_value=0,
                            max_value=30,
                            value=15,
                            help="Percentage reduction that would negatively impact delivery",
                            key="threshold_percentage"
                        )
                        
                        # Display note about threshold
                        st.info("The delivery risk threshold indicates the maximum budget reduction that can be absorbed without negatively impacting project delivery.")
                
                # Generate and display forecast
                if st.button("Generate Fiscal Forecast", key="generate_fiscal"):
                    with st.spinner("Generating fiscal forecast..."):
                        # Create base forecast plot
                        fig = fiscal_forecaster.create_fiscal_forecast_plot(
                            periods_ahead=periods_ahead,
                            period_type=period_type,
                            project=project,
                            budget_line=budget_line,
                            fiscal_year_start_month=fiscal_year_start_month
                        )
                        
                        # Calculate and get base metrics
                        metrics = fiscal_forecaster.calculate_fiscal_forecast_metrics(
                            periods_ahead=periods_ahead,
                            period_type=period_type,
                            project=project,
                            budget_line=budget_line,
                            fiscal_year_start_month=fiscal_year_start_month
                        )
                        
                        # If budget cut analysis is enabled, generate modified forecast
                        if enable_budget_cut:
                            # Get forecast data
                            historical_data, future_data, _ = fiscal_forecaster.generate_fiscal_forecast(
                                periods_ahead=periods_ahead,
                                period_type=period_type,
                                project=project,
                                budget_line=budget_line,
                                fiscal_year_start_month=fiscal_year_start_month
                            )
                            
                            # Calculate cut amount based on total forecast
                            total_forecast = metrics['total_forecast']
                            
                            if cut_type == "Percentage":
                                cut_amount = total_forecast * (budget_cut_percentage / 100)
                            else:
                                cut_amount = min(budget_cut_amount, total_forecast * 0.9)  # Cap at 90% of forecast
                            
                            # Calculate adjusted forecast values
                            cut_factor = cut_amount / total_forecast if total_forecast > 0 else 0
                            future_data_cut = future_data.copy()
                            future_data_cut['Amount'] = future_data['Amount'] * (1 - cut_factor)
                            
                            # Calculate new total after cuts
                            total_after_cut = metrics['total_forecast'] - cut_amount
                            
                            # Determine if cuts exceed threshold (would impact delivery)
                            cut_percentage = (cut_amount / total_forecast * 100) if total_forecast > 0 else 0
                            delivery_at_risk = cut_percentage > threshold_percentage
                            
                            # Add reduced budget trace to plot
                            fig.add_trace(
                                go.Scatter(
                                    x=future_data_cut['FiscalPeriod'],
                                    y=future_data_cut['Amount'],
                                    name='Reduced Budget Forecast',
                                    line=dict(
                                        color='red' if delivery_at_risk else 'orange',
                                        dash='dash'
                                    )
                                )
                            )
                            
                            # Add annotation about budget cut
                            fig.add_annotation(
                                x=0.5,
                                y=1.05,
                                xref="paper",
                                yref="paper",
                                text=f"Budget Cut Analysis: {budget_cut_percentage if cut_type == 'Percentage' else budget_cut_amount:,.0f}{' %' if cut_type == 'Percentage' else ' $'} reduction",
                                showarrow=False,
                                font=dict(
                                    color='red' if delivery_at_risk else 'black',
                                    size=12
                                ),
                                bgcolor='rgba(255, 220, 220, 0.5)' if delivery_at_risk else 'rgba(255, 255, 220, 0.5)'
                            )
                        
                        # Display the forecast chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display metrics in a grid
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Total Historical",
                                f"${metrics['total_historical']:,.2f}",
                                help="Total historical spending"
                            )
                            st.metric(
                                "Total Forecast",
                                f"${metrics['total_forecast']:,.2f}",
                                help="Total forecasted spending"
                            )
                        
                        with col2:
                            st.metric(
                                f"Avg {period_type.capitalize()} Historical",
                                f"${metrics['avg_period_historical']:,.2f}",
                                help=f"Average {period_type} historical spending"
                            )
                            st.metric(
                                f"Avg {period_type.capitalize()} Forecast",
                                f"${metrics['avg_period_forecast']:,.2f}",
                                help=f"Average {period_type} forecasted spending"
                            )
                        
                        with col3:
                            if metrics['growth_rate'] is not None:
                                st.metric(
                                    "Period-over-Period Growth",
                                    f"{metrics['growth_rate']:.1f}%",
                                    help="Growth rate between last historical period and first forecast period"
                                )
                            if metrics['cagr'] is not None:
                                st.metric(
                                    "Projected Annual Growth",
                                    f"{metrics['cagr']:.1f}%",
                                    help="Compound annual growth rate"
                                )
                        
                        # Display budget cut impact metrics if enabled
                        if enable_budget_cut:
                            st.subheader("Budget Cut Impact Analysis")
                            
                            impact_col1, impact_col2, impact_col3 = st.columns(3)
                            
                            with impact_col1:
                                st.metric(
                                    "Original Budget Forecast",
                                    f"${metrics['total_forecast']:,.2f}",
                                    help="Total forecasted spending before cuts"
                                )
                                st.metric(
                                    "Reduced Budget Forecast",
                                    f"${total_after_cut:,.2f}",
                                    delta=f"-${cut_amount:,.2f}",
                                    delta_color="inverse",
                                    help="Total forecasted spending after cuts"
                                )
                            
                            with impact_col2:
                                st.metric(
                                    "Budget Cut Percentage",
                                    f"{cut_percentage:.1f}%",
                                    help="Percentage of budget being cut"
                                )
                                st.metric(
                                    "Delivery Risk Threshold",
                                    f"{threshold_percentage:.1f}%",
                                    help="Maximum cut that won't impact delivery"
                                )
                            
                            with impact_col3:
                                # Calculate savings per period
                                avg_period_savings = cut_amount / periods_ahead
                                
                                st.metric(
                                    f"Average {period_type.capitalize()} Savings",
                                    f"${avg_period_savings:,.2f}",
                                    help=f"Average savings per {period_type}"
                                )
                                
                                # Display risk indicator
                                risk_label = "HIGH" if delivery_at_risk else "LOW"
                                risk_color = "red" if delivery_at_risk else "green"
                                
                                st.markdown(
                                    f"""
                                    <div style="
                                        background-color: {'rgba(255,0,0,0.1)' if delivery_at_risk else 'rgba(0,128,0,0.1)'};
                                        padding: 10px; 
                                        border-radius: 5px;
                                        border-left: 5px solid {risk_color};
                                        text-align: center;
                                    ">
                                        <span style="font-weight: bold; color: {risk_color};">
                                            Delivery Risk: {risk_label}
                                        </span>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                            
                            # Display risk explanation
                            if delivery_at_risk:
                                st.warning(
                                    f"""
                                    **Delivery Risk Alert:** The proposed budget cut of {cut_percentage:.1f}% exceeds the 
                                    delivery risk threshold of {threshold_percentage:.1f}%. This level of reduction may 
                                    negatively impact project delivery timelines or quality.
                                    
                                    Consider:
                                    - Revising the budget cut percentage downward
                                    - Extending project timelines to accommodate reduced resources
                                    - Prioritizing critical deliverables and postponing non-essential components
                                    """
                                )
                            else:
                                st.success(
                                    f"""
                                    **Low Delivery Risk:** The proposed budget cut of {cut_percentage:.1f}% is below the 
                                    delivery risk threshold of {threshold_percentage:.1f}%. Project delivery should 
                                    remain on track with careful management of resources.
                                    """
                                )
                            
                            # Add Potential Cut Candidates section
                            st.subheader("Potential Cut Candidates")
                            
                            # Get data for all projects and budget lines if not filtering
                            if project is None and budget_line is None:
                                project_data = df.groupby('Project')['Amount'].sum().reset_index()
                                budget_line_data = df.groupby('Budget Line')['Amount'].sum().reset_index()
                                
                                # Calculate recent activity (days since last transaction)
                                project_data['Recent Activity'] = project_data['Project'].apply(
                                    lambda p: (datetime.now() - pd.to_datetime(df[df['Project'] == p]['Date']).max()).days
                                )
                                budget_line_data['Recent Activity'] = budget_line_data['Budget Line'].apply(
                                    lambda b: (datetime.now() - pd.to_datetime(df[df['Budget Line'] == b]['Date']).max()).days
                                )
                                
                                # Calculate cut impact (higher = easier to cut)
                                project_data['Cut Impact'] = (
                                    project_data['Recent Activity'] / project_data['Recent Activity'].max() * 0.5 +
                                    (1 - (project_data['Amount'] / project_data['Amount'].max())) * 0.5
                                )
                                budget_line_data['Cut Impact'] = (
                                    budget_line_data['Recent Activity'] / budget_line_data['Recent Activity'].max() * 0.5 +
                                    (1 - (budget_line_data['Amount'] / budget_line_data['Amount'].max())) * 0.5
                                )
                                
                                # Sort by cut impact (descending = easier to cut first)
                                project_data = project_data.sort_values('Cut Impact', ascending=False)
                                budget_line_data = budget_line_data.sort_values('Cut Impact', ascending=False)
                                
                                # Calculate cumulative amount and how many items needed to reach target cut
                                project_data['Cumulative Amount'] = project_data['Amount'].cumsum()
                                budget_line_data['Cumulative Amount'] = budget_line_data['Amount'].cumsum()
                                
                                # Display two columns for project vs budget line cuts
                                cut_col1, cut_col2 = st.columns(2)
                                
                                with cut_col1:
                                    st.markdown("##### Projects Ranked by Cut Priority")
                                    
                                    # Show top candidates
                                    projects_display = project_data.head(5).copy()
                                    projects_display['% of Target'] = (projects_display['Amount'] / cut_amount * 100).round(1)
                                    projects_display['Amount'] = projects_display['Amount'].map('${:,.2f}'.format)
                                    projects_display['Recent Activity'] = projects_display['Recent Activity'].astype(int).astype(str) + " days"
                                    
                                    # Display as table
                                    st.dataframe(
                                        projects_display[['Project', 'Amount', '% of Target', 'Recent Activity']],
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                    
                                    # Calculate how many projects needed to reach target
                                    projects_needed = (project_data['Cumulative Amount'] < cut_amount).sum() + 1
                                    projects_needed = min(projects_needed, len(project_data))
                                    
                                    if projects_needed <= len(project_data):
                                        st.info(f"Cutting the top {projects_needed} projects would meet the target reduction of ${cut_amount:,.2f}")
                                    else:
                                        st.warning(f"Even cutting all {len(project_data)} projects would only save ${project_data['Amount'].sum():,.2f}")
                                
                                with cut_col2:
                                    st.markdown("##### Budget Lines Ranked by Cut Priority")
                                    
                                    # Show top candidates
                                    budget_display = budget_line_data.head(5).copy()
                                    budget_display['% of Target'] = (budget_display['Amount'] / cut_amount * 100).round(1)
                                    budget_display['Amount'] = budget_display['Amount'].map('${:,.2f}'.format)
                                    budget_display['Recent Activity'] = budget_display['Recent Activity'].astype(int).astype(str) + " days"
                                    
                                    # Display as table
                                    st.dataframe(
                                        budget_display[['Budget Line', 'Amount', '% of Target', 'Recent Activity']],
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                    
                                    # Calculate how many budget lines needed to reach target
                                    budgets_needed = (budget_line_data['Cumulative Amount'] < cut_amount).sum() + 1
                                    budgets_needed = min(budgets_needed, len(budget_line_data))
                                    
                                    if budgets_needed <= len(budget_line_data):
                                        st.info(f"Cutting the top {budgets_needed} budget lines would meet the target reduction of ${cut_amount:,.2f}")
                                    else:
                                        st.warning(f"Even cutting all {len(budget_line_data)} budget lines would only save ${budget_line_data['Amount'].sum():,.2f}")
                            
                            # Add options for mitigating high-risk cuts
                            if delivery_at_risk:
                                st.subheader("Risk Mitigation Options")
                                
                                mitigation_options = [
                                    "Extend project timeline",
                                    "Reduce project scope",
                                    "Postpone non-critical components",
                                    "Reallocate resources from other projects",
                                    "Seek alternative funding sources"
                                ]
                                
                                # Display options as checkboxes
                                selected_options = []
                                for option in mitigation_options:
                                    if st.checkbox(option, key=f"mitigation_{option}"):
                                        selected_options.append(option)
                                
                                # Initialize impact_text regardless of selection
                                impact_text = ""
                                
                                # If any options selected, populate the impact text
                                if selected_options:
                                    for option in selected_options:
                                        if option == "Extend project timeline":
                                            impact_text += "- Timeline extension may reduce resource demands per period\n"
                                        elif option == "Reduce project scope":
                                            impact_text += "- Scope reduction will require stakeholder agreement on priorities\n"
                                        elif option == "Postpone non-critical components":
                                            impact_text += "- Postponing components may help focus remaining budget on critical deliverables\n"
                                        elif option == "Reallocate resources from other projects":
                                            impact_text += "- Resource reallocation requires coordination across project portfolios\n"
                                        elif option == "Seek alternative funding sources":
                                            impact_text += "- Alternative funding may offset budget shortfalls but needs exploration\n"
                            
                                # Display the message only if there are selected options
                                if selected_options:
                                    st.info(f"**Selected Mitigation Approaches:**\n{impact_text}")
                        
                        # Add forecast assumptions and notes
                        st.subheader("Fiscal Forecast Notes")
                        fiscal_period_label = "quarterly" if period_type == "quarter" else "annual"
                        fiscal_period_unit = "quarters" if period_type == "quarter" else "years"
                        
                        st.markdown(f"""
                        **Forecast Overview:**
                        - This forecast shows projected budget trends for future fiscal {fiscal_period_unit}
                        - Fiscal year begins in {datetime(2023, fiscal_year_start_month, 1).strftime("%B")}
                        - Historical data is aggregated by fiscal {fiscal_period_label} periods
                        - The confidence interval shown represents the {confidence_level}% confidence level
                        
                        **Methodology:**
                        - Linear regression model trained on historical fiscal period data
                        - Trend analysis considers both sequential patterns and period-specific variations
                        - Forecast accuracy improves with more historical data points
                        
                        **Planning Recommendations:**
                        - Use this forecast for budget planning and resource allocation
                        - Consider adjusting future budgets based on projected growth trends
                        - Review historical spending patterns to identify potential areas for optimization
                        - Update forecasts regularly as new data becomes available
                        """)
            else:
                st.warning("Please upload data to generate fiscal year forecasts.")

        with tab7:
            st.header("Staff Expenses Analysis")
            
            if 'df' in st.session_state and st.session_state.df is not None:
                # Initialize staff tracker
                staff_tracker = StaffTracker(st.session_state.df)
                
                # Get staff data and metrics
                staff_df = staff_tracker.get_staff_data()
                staff_summary = staff_tracker.get_staff_summary()
                
                if not staff_df.empty:
                    # Display metrics
                    st.subheader("Staff Expense Overview")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Total Staff Expenses",
                            f"${staff_summary['total_expenses']:,.2f}",
                            f"{staff_summary['mom_change']:+.1f}%",
                            help="Total expenses for professional services, contractors, and consultants"
                        )
                        st.metric(
                            "Average Transaction",
                            f"${staff_summary['avg_transaction']:,.2f}",
                            help="Average amount per staff-related transaction"
                        )
                    
                    with col2:
                        st.metric(
                            "Month-over-Month Change",
                            f"{staff_summary['mom_change']:+.1f}%",
                            delta=f"{staff_summary['mom_change']:+.1f}%",
                            delta_color="inverse",
                            help="Percentage change in staff expenses compared to previous month"
                        )
                        st.metric(
                            "Peak Month",
                            f"{staff_summary['peak_month']}",
                            delta=f"${staff_summary['peak_amount']:,.2f}",
                            help="Month with highest staff expenses"
                        )
                    
                    with col3:
                        st.metric(
                            "Number of Transactions",
                            f"{staff_summary['transaction_count']:,}",
                            help="Total number of staff-related transactions"
                        )
                        st.metric(
                            "Average Monthly",
                            f"${staff_summary['avg_monthly']:,.2f}",
                            help="Average monthly staff expenses"
                        )
                    
                    # Add detailed transaction view
                    st.subheader("Detailed Staff Transactions")
                    
                    # Add filters
                    filter_col1, filter_col2 = st.columns(2)
                    
                    with filter_col1:
                        # Category filter
                        categories = ["All"] + sorted(staff_df['Staff Category'].unique().tolist())
                        selected_category = st.selectbox(
                            "Filter by Category",
                            categories,
                            key="staff_category_filter"
                        )
                        
                        # Project filter
                        projects = ["All"] + sorted(staff_df['Project'].unique().tolist())
                        selected_project = st.selectbox(
                            "Filter by Project",
                            projects,
                            key="staff_project_filter"
                        )
                    
                    with filter_col2:
                        # Date range filter
                        min_date = pd.to_datetime(staff_df['Date']).min().date()
                        max_date = pd.to_datetime(staff_df['Date']).max().date()
                        date_range = st.date_input(
                            "Date Range",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            key="staff_date_range"
                        )
                        
                        # Amount range filter
                        min_amount = float(staff_df['Amount'].min())
                        max_amount = float(staff_df['Amount'].max())
                        amount_range = st.slider(
                            "Amount Range ($)",
                            min_value=min_amount,
                            max_value=max_amount,
                            value=(min_amount, max_amount),
                            format="$%.2f",
                            key="staff_amount_range"
                        )
                    
                    # Apply filters
                    filtered_staff_df = staff_df.copy()
                    
                    if selected_category != "All":
                        filtered_staff_df = filtered_staff_df[filtered_staff_df['Staff Category'] == selected_category]
                    
                    if selected_project != "All":
                        filtered_staff_df = filtered_staff_df[filtered_staff_df['Project'] == selected_project]
                    
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        filtered_staff_df = filtered_staff_df[
                            (pd.to_datetime(filtered_staff_df['Date']).dt.date >= start_date) & 
                            (pd.to_datetime(filtered_staff_df['Date']).dt.date <= end_date)
                        ]
                    
                    filtered_staff_df = filtered_staff_df[
                        (filtered_staff_df['Amount'] >= amount_range[0]) & 
                        (filtered_staff_df['Amount'] <= amount_range[1])
                    ]
                    
                    # Display filtered data
                    st.dataframe(
                        filtered_staff_df.sort_values('Date', ascending=False),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
                            "Amount": st.column_config.NumberColumn("Amount", format="$%.2f")
                        }
                    )
                    
                    # Display charts
                    st.subheader("Staff Expense Analysis")
                    
                    # Create two columns for charts
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        # Show project breakdown
                        project_chart = staff_tracker.create_project_breakdown_chart()
                        st.plotly_chart(project_chart, use_container_width=True)
                    
                    with chart_col2:
                        # Show vendor breakdown if available
                        vendor_chart = staff_tracker.create_vendor_breakdown_chart()
                        if vendor_chart:
                            st.plotly_chart(vendor_chart, use_container_width=True)
                    
                    # Add summary statistics
                    st.subheader("Summary Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Category Breakdown**")
                        for category, amount in staff_summary['category_breakdown'].items():
                            st.markdown(f"- {category}: ${amount:,.2f}")
                    
                    with col2:
                        st.markdown("**Top Projects**")
                        top_projects = sorted(staff_summary['project_breakdown'].items(), key=lambda x: x[1], reverse=True)[:5]
                        for project, amount in top_projects:
                            st.markdown(f"- {project}: ${amount:,.2f}")
                    
                    with col3:
                        if staff_summary['vendor_breakdown']:
                            st.markdown("**Top Vendors**")
                            top_vendors = sorted(staff_summary['vendor_breakdown'].items(), key=lambda x: x[1], reverse=True)[:5]
                            for vendor, amount in top_vendors:
                                st.markdown(f"- {vendor}: ${amount:,.2f}")
                else:
                    st.info("No staff-related expenses found in the data. This could be because:")
                    st.markdown("""
                    - The budget lines don't match the expected categories (Professional Services, Contractors, Consultants)
                    - The data hasn't been categorized with staff-related expenses
                    - The data is empty or doesn't contain the required columns
                    
                    Please ensure your data includes budget lines related to staff expenses, such as:
                    - Professional Services
                    - Consulting Services
                    - Contractors
                    - External Consultants
                    """)
            else:
                st.warning("Please upload data to analyze staff expenses.")

    else:
        st.info("Please upload an Excel file to begin analysis")

def plotly_chart_with_click_event(fig, key=None, config=None):
    """Wrapper for st.plotly_chart that enables click event handling"""
    # Add click event handling to the figure
    fig.update_layout(clickmode='event+select')
    
    # Create the chart
    clicked_data = st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        config=config
    )
    
    return clicked_data

if __name__ == "__main__":
    main() 