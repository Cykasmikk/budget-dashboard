"""
Styles module for the Budget Visualization Dashboard
Contains styling elements and color definitions for the application
"""

# Color scheme for plots
COLORS = {
    'primary': '#4A90E2',    # Bright blue
    'secondary': '#7B68EE',  # Purple
    'accent': '#9F7AEA',     # Lighter purple
    'gradient_start': '#4A90E2',  # Blue
    'gradient_end': '#7B68EE',    # Purple
    'background': '#1E2126',      # Dark background
    'text': '#FFFFFF'             # White text
}

def get_css_styles():
    """Return the CSS styles for the application"""
    return """
    <style>
    /* Main background */
    .stApp {
        background-color: #1E2126;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #262B32;
    }
    
    /* Main content padding - significantly reduced */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* Header/Title adjustments - compact */
    .stTitle {
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    h1 {
        margin-top: 0 !important;
        padding-top: 0 !important;
        margin-bottom: 0.5rem !important;
        font-size: 1.5rem !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #4A90E2, #7B68EE);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 5px 10px;
        transition: all 0.3s ease;
        font-size: 0.8rem;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #7B68EE, #4A90E2);
        transform: translateY(-2px);
    }
    
    /* Cards/Containers */
    .stCard {
        background-color: #262B32;
        border-radius: 10px;
        padding: 0.5rem;
        border: 1px solid rgba(123, 104, 238, 0.1);
    }
    
    /* Metrics - more compact */
    .stMetric {
        background: linear-gradient(45deg, rgba(74, 144, 226, 0.1), rgba(123, 104, 238, 0.2));
        border-radius: 6px;
        padding: 0.25rem !important;
        height: 60px !important;
        min-height: 60px !important;
        max-height: 60px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        overflow: hidden !important;
    }
    
    /* Make metric labels consistent size */
    .stMetric label {
        font-size: 0.65rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.1rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        line-height: 1 !important;
        flex-shrink: 0 !important;
    }
    
    /* Make metric values consistent size */
    .stMetric [data-testid="stMetricValue"] {
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        margin: 0 !important;
        padding: 0 !important;
        flex-shrink: 0 !important;
    }
    
    /* Make metric deltas consistent size */
    .stMetric [data-testid="stMetricDelta"] {
        font-size: 0.65rem !important;
        line-height: 1 !important;
        margin: 0 !important;
        padding: 0 !important;
        flex-shrink: 0 !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: linear-gradient(45deg, rgba(74, 144, 226, 0.1), rgba(123, 104, 238, 0.1));
        min-height: 1.5rem !important;
        line-height: 1.5rem !important;
    }
    
    .stSelectbox [data-testid="stText"] {
        font-size: 0.8rem !important;
    }
    
    /* Text colors */
    .stMarkdown {
        color: #FFFFFF;
        margin-bottom: 0.25rem !important;
        margin-top: 0.25rem !important;
    }
    
    /* Container for metrics */
    .metrics-container {
        margin: 0.25rem 0;
        gap: 0.25rem;
    }
    
    /* Ensure columns have equal width */
    [data-testid="column"] {
        width: calc(33.33% - 0.25rem) !important;
        padding: 0.1rem !important;
    }
    
    /* Add ellipsis for long text in metrics */
    .stMetric > div {
        width: 100% !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Remove extra spacing in metrics */
    .stMetric > div > div {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Adjust spacing between rows */
    .metrics-container > div {
        margin-bottom: 0.25rem !important;
    }

    /* Tab container adjustments */
    .stTabs {
        margin-top: 0.25rem !important;
    }
    
    /* Tab labels */
    button[role="tab"] {
        padding: 0.25rem 0.5rem !important;
        min-height: 1.75rem !important;
        font-size: 0.8rem !important;
    }
    
    /* Tab content area */
    [data-baseweb="tab-panel"] {
        padding: 0.5rem 0 !important;
        height: calc(100vh - 240px) !important;
        box-sizing: border-box !important;
    }
    
    /* Tab-specific content containers */
    [data-baseweb="tab-panel"] > div {
        height: 100% !important;
    }
    
    /* Make the dataframe use full height in tab3 */
    [data-baseweb="tab-panel"] [data-testid="stDataFrame"] {
        height: 100% !important;
        max-height: none !important;
    }

    /* Adjust spacing for plotly charts */
    .js-plotly-plot {
        margin-top: 0.25rem !important;
        margin-bottom: 0.25rem !important;
    }

    /* Streamlit default header adjustments */
    [data-testid="stHeader"] {
        height: auto !important;
        min-height: 0 !important;
        padding-top: 0.25rem !important;
    }

    /* Adjust main app padding */
    .main .block-container {
        padding-top: 0.5rem !important;
        max-width: 100% !important;
    }
    
    /* Dataframe/table adjustments */
    .stDataFrame {
        height: auto !important;
        max-height: 200px !important;
        overflow-y: auto !important;
        font-size: 0.75rem !important;
    }
    
    /* Minimize space between sections */
    .stSectionContainer {
        margin-top: 0.25rem !important;
        margin-bottom: 0.25rem !important;
    }
    
    /* Subheader styling */
    h2, h3 {
        font-size: 1rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Label text */
    .css-16idsys p {
        font-size: 0.8rem !important;
        margin-bottom: 0.25rem !important;
    }
    
    /* Remove all tooltip icons to save space */
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Hide footer */
    footer {
        display: none !important;
    }

    /* Fix grid/flexbox gaps throughout the app */
    /* Target Streamlit's grid container */
    [data-testid="stHorizontalBlock"] {
        gap: 0 !important;
        column-gap: 0.2rem !important;
    }
    
    /* Target any flexbox containers */
    div[data-testid="StyledLinkIconContainer"] {
        gap: 0 !important;
    }
    
    /* Target any grid containers */
    .row-widget {
        gap: 0 !important;
    }
    
    /* Fix gap between filter inputs */
    .stDateInput, .stMultiSelect, .stTextInput, .stSlider {
        margin-bottom: 0.2rem !important;
    }
    
    /* Reduce default margins on all elements */
    div:has(> [data-testid="stVerticalBlock"]) {
        gap: 0 !important;
        row-gap: 0.2rem !important;
    }
    
    /* Apply tighter packing to the whole UI */
    [data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    
    /* Remove margin from input container elements */
    [data-baseweb="select"] {
        margin-top: 0 !important;
        margin-bottom: 0.2rem !important;
    }
    
    /* Chart container spacing */
    .element-container {
        margin-top: 0 !important;
        margin-bottom: 0.1rem !important;
    }
    
    /* Tighter flex layout for filters */
    [data-testid="stHorizontalBlock"] > div {
        padding-left: 0 !important;
        padding-right: 0.2rem !important;
    }
    </style>
    """ 