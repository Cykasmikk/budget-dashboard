import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BudgetForecaster:
    def __init__(self, df):
        """Initialize the forecaster with historical data"""
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Month'] = self.df['Date'].dt.to_period('M')
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
    
    def prepare_data(self, project=None, budget_line=None):
        """Prepare data for forecasting"""
        # Filter data based on project and budget line if specified
        df_filtered = self.df.copy()
        if project:
            df_filtered = df_filtered[df_filtered['Project'] == project]
        if budget_line:
            df_filtered = df_filtered[df_filtered['Budget Line'] == budget_line]
            
        # Group by month and calculate total amount
        monthly_data = df_filtered.groupby('Month')['Amount'].sum().reset_index()
        monthly_data['Month'] = monthly_data['Month'].astype(str)
        
        # Create features for regression
        monthly_data['Month_Number'] = range(len(monthly_data))
        monthly_data['Year'] = monthly_data['Month'].str[:4].astype(int)
        monthly_data['Month_Of_Year'] = monthly_data['Month'].str[5:].astype(int)
        
        return monthly_data
    
    def train_model(self, project=None, budget_line=None):
        """Train the forecasting model"""
        monthly_data = self.prepare_data(project, budget_line)
        
        # Prepare features
        X = monthly_data[['Month_Number', 'Year', 'Month_Of_Year']]
        y = monthly_data['Amount']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        return model, X_scaled, y
    
    def generate_forecast(self, months_ahead=12, project=None, budget_line=None):
        """Generate forecast for specified number of months ahead"""
        model, X_scaled, y = self.train_model(project, budget_line)
        monthly_data = self.prepare_data(project, budget_line)
        
        # Generate future dates
        last_date = pd.to_datetime(monthly_data['Month'].iloc[-1])
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months_ahead, freq='M')
        
        # Prepare future features
        future_data = pd.DataFrame({
            'Month': future_dates.to_period('M').astype(str),
            'Month_Number': range(len(monthly_data), len(monthly_data) + months_ahead),
            'Year': future_dates.year,
            'Month_Of_Year': future_dates.month
        })
        
        # Scale future features
        X_future = future_data[['Month_Number', 'Year', 'Month_Of_Year']]
        X_future_scaled = self.scaler.transform(X_future)
        
        # Generate initial predictions
        predictions = model.predict(X_future_scaled)
        
        # Apply realistic constraints to predictions
        # Check if budget line is staff-related
        is_staff = self.is_staff_related(budget_line)
        
        # Get recent average (last 3 months if available)
        recent_avg = monthly_data['Amount'].tail(3).mean() if len(monthly_data) >= 3 else monthly_data['Amount'].mean()
        
        # Minimum expense threshold - ensures forecasts never go to zero/negative
        min_expense = max(recent_avg * 0.2, 100)  # At least 20% of recent average or $100, whichever is larger
        
        # Apply different constraints based on expense type
        for i in range(len(predictions)):
            if is_staff:
                # For staff-related expenses, maintain more stability
                # If prediction is going down, limit the rate of decrease
                # Staff costs typically don't decrease unless there's an intentional reduction
                if predictions[i] < recent_avg:
                    # Apply diminishing returns to savings
                    decrease_factor = 0.7  # Maximum decrease is 30% from recent average
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
        
        # Calculate confidence intervals
        std_dev = np.std(y - model.predict(X_scaled))
        confidence_interval = 1.96 * std_dev  # 95% confidence interval
        
        return future_data, predictions, confidence_interval
    
    def create_forecast_plot(self, months_ahead=12, project=None, budget_line=None):
        """Create a plot showing historical data and forecast"""
        monthly_data = self.prepare_data(project, budget_line)
        future_data, predictions, confidence_interval = self.generate_forecast(months_ahead, project, budget_line)
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add historical data
        fig.add_trace(
            go.Scatter(
                x=monthly_data['Month'],
                y=monthly_data['Amount'],
                name='Historical',
                line=dict(color='#4A90E2')
            ),
            secondary_y=False
        )
        
        # Add forecast
        fig.add_trace(
            go.Scatter(
                x=future_data['Month'],
                y=predictions,
                name='Forecast',
                line=dict(color='#7B68EE', dash='dash')
            ),
            secondary_y=False
        )
        
        # Add confidence interval - ensure lower bound doesn't go below minimum
        recent_avg = monthly_data['Amount'].tail(3).mean() if len(monthly_data) >= 3 else monthly_data['Amount'].mean()
        min_expense = max(recent_avg * 0.2, 100)  # Same minimum threshold as in generate_forecast
        
        lower_bounds = np.maximum(predictions - confidence_interval, min_expense)
        
        fig.add_trace(
            go.Scatter(
                x=future_data['Month'].tolist() + future_data['Month'].tolist()[::-1],
                y=(predictions + confidence_interval).tolist() + lower_bounds.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(123, 104, 238, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval'
            ),
            secondary_y=False
        )
        
        # Update layout
        title = "Budget Forecast"
        if project:
            title += f" - {project}"
        if budget_line:
            title += f" ({budget_line})"
            
        fig.update_layout(
            title=title,
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#333333',
            showlegend=True
        )
        
        return fig
    
    def calculate_forecast_metrics(self, months_ahead=12, project=None, budget_line=None):
        """Calculate various forecast metrics"""
        monthly_data = self.prepare_data(project, budget_line)
        future_data, predictions, confidence_interval = self.generate_forecast(months_ahead, project, budget_line)
        
        # Calculate metrics
        total_historical = monthly_data['Amount'].sum()
        total_forecast = predictions.sum()
        avg_monthly_historical = monthly_data['Amount'].mean()
        avg_monthly_forecast = predictions.mean()
        
        # Calculate year-over-year growth
        if len(monthly_data) >= 12:
            last_year = monthly_data['Amount'].tail(12).sum()
            forecast_year = predictions[:12].sum()
            yoy_growth = ((forecast_year - last_year) / last_year) * 100
        else:
            yoy_growth = None
        
        return {
            'total_historical': total_historical,
            'total_forecast': total_forecast,
            'avg_monthly_historical': avg_monthly_historical,
            'avg_monthly_forecast': avg_monthly_forecast,
            'yoy_growth': yoy_growth,
            'confidence_interval': confidence_interval
        } 