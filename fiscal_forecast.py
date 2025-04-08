import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FiscalYearForecaster:
    def __init__(self, df):
        """Initialize the fiscal year forecaster with historical data"""
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Month'] = self.df['Date'].dt.to_period('M')
        self.df['Quarter'] = self.df['Date'].dt.to_period('Q')
        self.df['Year'] = self.df['Date'].dt.year
        self.scaler = StandardScaler()
        
        # Define staff-related budget lines
        self.staff_categories = [
            'Professional Services',
            'Consulting Services',
            'External Consultants',
            'Professional Fees',
            'Consultant Fees',
            'Contractors',
            'Contract Labor',
            'External Contractors',
            'Contract Services',
            'Consultants',
            'Management Consultants',
            'Technical Consultants',
            'Business Consultants',
            'Salary',
            'Wages',
            'Staff',
            'Personnel',
            'Employee'
        ]
    
    def is_staff_related(self, budget_line):
        """Check if a budget line is staff-related"""
        if budget_line is None:
            return False
        return any(category.lower() in budget_line.lower() for category in self.staff_categories)
    
    def prepare_quarterly_data(self, project=None, budget_line=None, fiscal_year_start_month=10):
        """Prepare quarterly data for forecasting with fiscal year support"""
        # Filter data based on project and budget line if specified
        df_filtered = self.df.copy()
        if project:
            df_filtered = df_filtered[df_filtered['Project'] == project]
        if budget_line:
            df_filtered = df_filtered[df_filtered['Budget Line'] == budget_line]
        
        # Calculate fiscal year and quarter
        df_filtered['FiscalYear'] = df_filtered['Date'].apply(
            lambda x: x.year if x.month >= fiscal_year_start_month else x.year - 1
        )
        df_filtered['FiscalQuarter'] = df_filtered['Date'].apply(
            lambda x: (x.month - fiscal_year_start_month) % 12 // 3 + 1
        )
        
        # Create fiscal period label (e.g., "FY2023-Q1")
        df_filtered['FiscalPeriod'] = df_filtered.apply(
            lambda x: f"FY{x['FiscalYear']}-Q{x['FiscalQuarter']}", 
            axis=1
        )
        
        # Group by fiscal period and calculate total amount
        quarterly_data = df_filtered.groupby(['FiscalYear', 'FiscalQuarter', 'FiscalPeriod'])['Amount'].sum().reset_index()
        quarterly_data = quarterly_data.sort_values(['FiscalYear', 'FiscalQuarter'])
        
        # Create features for regression
        quarterly_data['Period_Number'] = range(len(quarterly_data))
        
        return quarterly_data
    
    def prepare_annual_data(self, project=None, budget_line=None, fiscal_year_start_month=10):
        """Prepare annual data for forecasting with fiscal year support"""
        # Filter data based on project and budget line if specified
        df_filtered = self.df.copy()
        if project:
            df_filtered = df_filtered[df_filtered['Project'] == project]
        if budget_line:
            df_filtered = df_filtered[df_filtered['Budget Line'] == budget_line]
        
        # Calculate fiscal year
        df_filtered['FiscalYear'] = df_filtered['Date'].apply(
            lambda x: x.year if x.month >= fiscal_year_start_month else x.year - 1
        )
        
        # Create fiscal year label (e.g., "FY2023")
        df_filtered['FiscalPeriod'] = df_filtered.apply(
            lambda x: f"FY{x['FiscalYear']}", 
            axis=1
        )
        
        # Group by fiscal year and calculate total amount
        annual_data = df_filtered.groupby(['FiscalYear', 'FiscalPeriod'])['Amount'].sum().reset_index()
        annual_data = annual_data.sort_values('FiscalYear')
        
        # Create features for regression
        annual_data['Period_Number'] = range(len(annual_data))
        
        return annual_data
    
    def train_model(self, data, window_size=4):
        """Calculate Simple Moving Average using the provided data"""
        # Calculate SMA for the last window_size periods
        sma = data['Amount'].rolling(window=window_size, min_periods=1).mean()
        # Get the last SMA value as the base for forecasting
        last_sma = sma.iloc[-1]
        
        return last_sma, data['Amount'].std()  # Return last SMA and standard deviation for confidence intervals
    
    def generate_fiscal_forecast(self, periods_ahead=4, period_type='quarter', project=None, budget_line=None, fiscal_year_start_month=10):
        """Generate forecast for specified number of fiscal periods ahead using Simple Moving Average"""
        # Prepare data based on period type
        if period_type == 'quarter':
            historical_data = self.prepare_quarterly_data(project, budget_line, fiscal_year_start_month)
            period_label = 'Quarter'
        else:  # annual
            historical_data = self.prepare_annual_data(project, budget_line, fiscal_year_start_month)
            period_label = 'Year'
        
        # Calculate SMA and standard deviation
        last_sma, std_dev = self.train_model(historical_data)
        
        # Generate future periods
        last_period_number = historical_data['Period_Number'].iloc[-1]
        future_period_numbers = range(last_period_number + 1, last_period_number + periods_ahead + 1)
        
        # Get last fiscal year and quarter/period
        last_fiscal_year = historical_data['FiscalYear'].iloc[-1]
        
        # Create future periods data
        future_data = pd.DataFrame({'Period_Number': future_period_numbers})
        
        if period_type == 'quarter':
            last_fiscal_quarter = historical_data['FiscalQuarter'].iloc[-1]
            
            # Generate future fiscal years and quarters
            future_fiscal_years = []
            future_fiscal_quarters = []
            future_fiscal_periods = []
            
            current_year = last_fiscal_year
            current_quarter = last_fiscal_quarter
            
            for _ in range(periods_ahead):
                current_quarter += 1
                if current_quarter > 4:
                    current_quarter = 1
                    current_year += 1
                
                future_fiscal_years.append(current_year)
                future_fiscal_quarters.append(current_quarter)
                future_fiscal_periods.append(f"FY{current_year}-Q{current_quarter}")
            
            future_data['FiscalYear'] = future_fiscal_years
            future_data['FiscalQuarter'] = future_fiscal_quarters
            future_data['FiscalPeriod'] = future_fiscal_periods
            
        else:  # annual
            # Generate future fiscal years
            future_fiscal_years = [last_fiscal_year + i + 1 for i in range(periods_ahead)]
            future_fiscal_periods = [f"FY{year}" for year in future_fiscal_years]
            
            future_data['FiscalYear'] = future_fiscal_years
            future_data['FiscalPeriod'] = future_fiscal_periods
        
        # Generate initial predictions using SMA
        predictions = [last_sma] * len(future_period_numbers)
        
        # Apply realistic constraints to predictions
        # Check if budget line is staff-related
        is_staff = self.is_staff_related(budget_line)
        
        # Get recent average (last 2-3 periods if available)
        recent_periods = min(3, len(historical_data))
        recent_avg = historical_data['Amount'].tail(recent_periods).mean() if recent_periods > 0 else 1000
        
        # Minimum expense threshold - ensures forecasts never go to zero/negative
        min_expense = max(recent_avg * 0.2, 100)  # At least 20% of recent average or $100, whichever is larger
        
        # Calculate recent trend
        if len(historical_data) >= 2:
            recent_trend = (historical_data['Amount'].iloc[-1] - historical_data['Amount'].iloc[-2]) / historical_data['Amount'].iloc[-2]
        else:
            recent_trend = 0
        
        # Apply different constraints based on expense type and recent trend
        for i in range(len(predictions)):
            # Calculate quarter-over-quarter growth rate
            if i == 0:
                qoq_growth = (predictions[i] - historical_data['Amount'].iloc[-1]) / historical_data['Amount'].iloc[-1]
            else:
                qoq_growth = (predictions[i] - predictions[i-1]) / predictions[i-1]
            
            # If recent trend is negative (cost cutting), limit growth
            if recent_trend < 0:
                # Maximum annual growth rate of 2.5% (approximately 0.6% per quarter)
                max_qoq_growth = 0.006
                if qoq_growth > max_qoq_growth:
                    predictions[i] = predictions[i-1] * (1 + max_qoq_growth)
            
            if is_staff:
                # For staff-related expenses, maintain more stability
                # Staff costs typically don't decrease unless there's an intentional reduction
                if predictions[i] < recent_avg:
                    # Apply diminishing returns to savings for staff expenses
                    # Maximum decrease is 30% from recent average for staff-related expenses
                    decrease_factor = 0.7
                    predictions[i] = max(predictions[i], recent_avg * decrease_factor)
            else:
                # For non-staff expenses, still enforce minimum but allow more variability
                if predictions[i] < recent_avg * 0.5:
                    # Apply diminishing returns to savings
                    relative_decrease = (recent_avg - predictions[i]) / recent_avg
                    # The larger the decrease, the less effective it becomes
                    adjusted_decrease = relative_decrease * (1 - 0.5 * relative_decrease)
                    predictions[i] = recent_avg * (1 - adjusted_decrease)
            
            # Final check to ensure no prediction goes below minimum threshold
            predictions[i] = max(min_expense, predictions[i])
        
        # Update the predictions in the DataFrame
        future_data['Amount'] = predictions
        
        # Calculate confidence intervals (using standard deviation)
        confidence_interval = 1.96 * std_dev  # 95% confidence interval
        
        return historical_data, future_data, confidence_interval
    
    def create_fiscal_forecast_plot(self, periods_ahead=4, period_type='quarter', project=None, budget_line=None, fiscal_year_start_month=10):
        """Create a plot showing historical data and fiscal forecast"""
        historical_data, future_data, confidence_interval = self.generate_fiscal_forecast(
            periods_ahead, period_type, project, budget_line, fiscal_year_start_month
        )
        
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(
            go.Bar(
                x=historical_data['FiscalPeriod'],
                y=historical_data['Amount'],
                name='Historical',
                marker_color='#4A90E2'
            )
        )
        
        # Add forecast
        fig.add_trace(
            go.Bar(
                x=future_data['FiscalPeriod'],
                y=future_data['Amount'],
                name='Forecast',
                marker_color='#7B68EE',
                marker_pattern_shape='/'
            )
        )
        
        # Add a line connecting all periods
        all_periods = pd.concat([
            historical_data[['FiscalPeriod', 'Amount']],
            future_data[['FiscalPeriod', 'Amount']]
        ])
        
        fig.add_trace(
            go.Scatter(
                x=all_periods['FiscalPeriod'],
                y=all_periods['Amount'],
                mode='lines',
                line=dict(color='rgba(0,0,0,0.5)', width=2),
                name='Trend',
                showlegend=True
            )
        )
        
        # Ensure confidence intervals don't go below minimum threshold
        recent_periods = min(3, len(historical_data))
        recent_avg = historical_data['Amount'].tail(recent_periods).mean() if recent_periods > 0 else 1000
        min_expense = max(recent_avg * 0.2, 100)
        
        # Add confidence interval for forecast periods
        for i, row in future_data.iterrows():
            lower_bound = max(row['Amount'] - confidence_interval, min_expense)
            upper_bound = row['Amount'] + confidence_interval
            
            fig.add_trace(
                go.Scatter(
                    x=[row['FiscalPeriod'], row['FiscalPeriod']],
                    y=[lower_bound, upper_bound],
                    mode='lines',
                    line=dict(color='rgba(123, 104, 238, 0.6)', width=4),
                    showlegend=False
                )
            )
        
        # Update layout
        title = f"Fiscal {period_type.capitalize()} Forecast"
        if project:
            title += f" - {project}"
        if budget_line:
            title += f" ({budget_line})"
            
        fig.update_layout(
            title=title,
            xaxis_title=f"Fiscal {period_type.capitalize()}",
            yaxis_title="Amount ($)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#333333',
            showlegend=True,
            hovermode="x unified",
            barmode='group'
        )
        
        # Format y-axis as currency
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        
        return fig
    
    def calculate_fiscal_forecast_metrics(self, periods_ahead=4, period_type='quarter', project=None, budget_line=None, fiscal_year_start_month=10):
        """Calculate various fiscal forecast metrics"""
        historical_data, future_data, confidence_interval = self.generate_fiscal_forecast(
            periods_ahead, period_type, project, budget_line, fiscal_year_start_month
        )
        
        # Calculate metrics
        total_historical = historical_data['Amount'].sum()
        total_forecast = future_data['Amount'].sum()
        avg_period_historical = historical_data['Amount'].mean()
        avg_period_forecast = future_data['Amount'].mean()
        
        # Calculate growth rates
        if len(historical_data) > 0:
            last_period_historical = historical_data['Amount'].iloc[-1]
            first_period_forecast = future_data['Amount'].iloc[0]
            growth_rate = ((first_period_forecast - last_period_historical) / last_period_historical) * 100 if last_period_historical != 0 else 0
            
            # Calculate compound annual growth rate (CAGR)
            if len(historical_data) >= 4 and period_type == 'quarter':
                last_year_historical = historical_data['Amount'].tail(4).sum()
                first_year_forecast = future_data['Amount'].head(4).sum() if len(future_data) >= 4 else future_data['Amount'].sum()
                cagr = ((first_year_forecast / last_year_historical) ** (1 / 1) - 1) * 100 if last_year_historical != 0 else 0
            elif len(historical_data) >= 1 and period_type == 'annual':
                last_year_historical = historical_data['Amount'].iloc[-1]
                first_year_forecast = future_data['Amount'].iloc[0]
                cagr = ((first_year_forecast / last_year_historical) ** (1 / 1) - 1) * 100 if last_year_historical != 0 else 0
            else:
                cagr = None
        else:
            growth_rate = None
            cagr = None
        
        return {
            'total_historical': total_historical,
            'total_forecast': total_forecast,
            'avg_period_historical': avg_period_historical,
            'avg_period_forecast': avg_period_forecast,
            'growth_rate': growth_rate,
            'cagr': cagr,
            'confidence_interval': confidence_interval
        } 