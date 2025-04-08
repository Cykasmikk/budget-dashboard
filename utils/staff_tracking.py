import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

class StaffTracker:
    def __init__(self, df):
        """Initialize the staff tracker with the main DataFrame"""
        self.df = df
        
        # Define staff-related budget lines
        self.staff_categories = {
            'Professional Services': [
                'Professional Services',
                'Consulting Services',
                'External Consultants',
                'Professional Fees',
                'Consultant Fees'
            ],
            'Contractors': [
                'Contractors',
                'Contract Labor',
                'External Contractors',
                'Contract Services'
            ],
            'Consultants': [
                'Consultants',
                'Management Consultants',
                'Technical Consultants',
                'Business Consultants'
            ]
        }
        
        # Create a mapping of budget lines to categories
        self.category_mapping = {}
        for category, keywords in self.staff_categories.items():
            for keyword in keywords:
                self.category_mapping[keyword] = category
    
    def get_staff_data(self):
        """Extract and categorize staff-related expenses"""
        # Create a copy of the DataFrame
        staff_df = self.df.copy()
        
        # Add category column based on budget line
        staff_df['Staff Category'] = staff_df['Budget Line'].map(
            lambda x: next((cat for cat, keywords in self.staff_categories.items() 
                          if any(keyword.lower() in x.lower() for keyword in keywords)), 
                         'Other')
        )
        
        # Filter for staff-related expenses
        staff_df = staff_df[staff_df['Staff Category'] != 'Other']
        
        return staff_df
    
    def calculate_staff_metrics(self):
        """Calculate key metrics for staff expenses"""
        staff_df = self.get_staff_data()
        
        metrics = {
            'total_staff_expenses': staff_df['Amount'].sum(),
            'category_breakdown': staff_df.groupby('Staff Category')['Amount'].sum().to_dict(),
            'project_breakdown': staff_df.groupby('Project')['Amount'].sum().to_dict(),
            'monthly_trend': staff_df.groupby(pd.to_datetime(staff_df['Date']).dt.to_period('M'))['Amount'].sum(),
            'vendor_breakdown': staff_df.groupby('Vendor Name')['Amount'].sum().to_dict() if 'Vendor Name' in staff_df.columns else {},
            'avg_transaction': staff_df['Amount'].mean(),
            'max_transaction': staff_df['Amount'].max(),
            'transaction_count': len(staff_df)
        }
        
        return metrics
    
    def create_staff_trend_chart(self):
        """Create a line chart showing staff expenses over time"""
        staff_df = self.get_staff_data()
        
        # Group by date and category
        monthly_data = staff_df.groupby([
            pd.to_datetime(staff_df['Date']).dt.to_period('M'),
            'Staff Category'
        ])['Amount'].sum().reset_index()
        
        # Convert period to datetime for plotting
        monthly_data['Date'] = monthly_data['Date'].astype(str)
        
        fig = px.line(
            monthly_data,
            x='Date',
            y='Amount',
            color='Staff Category',
            title='Staff Expenses Over Time',
            height=400
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Amount ($)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#333333'
        )
        
        return fig
    
    def create_category_pie_chart(self):
        """Create a pie chart showing staff expense categories"""
        staff_df = self.get_staff_data()
        
        category_data = staff_df.groupby('Staff Category')['Amount'].sum().reset_index()
        
        fig = px.pie(
            category_data,
            values='Amount',
            names='Staff Category',
            title='Staff Expense Categories',
            height=400
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#333333'
        )
        
        return fig
    
    def create_project_breakdown_chart(self):
        """Create a bar chart showing staff expenses by project"""
        staff_df = self.get_staff_data()
        
        project_data = staff_df.groupby('Project')['Amount'].sum().reset_index()
        project_data = project_data.sort_values('Amount', ascending=True)  # Sort for horizontal bar chart
        
        fig = px.bar(
            project_data,
            x='Amount',
            y='Project',
            orientation='h',
            title='Staff Expenses by Project',
            height=400
        )
        
        fig.update_layout(
            xaxis_title="Amount ($)",
            yaxis_title="Project",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#333333'
        )
        
        return fig
    
    def create_vendor_breakdown_chart(self):
        """Create a bar chart showing staff expenses by vendor"""
        staff_df = self.get_staff_data()
        
        if 'Vendor Name' not in staff_df.columns:
            return None
        
        vendor_data = staff_df.groupby('Vendor Name')['Amount'].sum().reset_index()
        vendor_data = vendor_data.sort_values('Amount', ascending=True)  # Sort for horizontal bar chart
        
        fig = px.bar(
            vendor_data,
            x='Amount',
            y='Vendor Name',
            orientation='h',
            title='Staff Expenses by Vendor',
            height=400
        )
        
        fig.update_layout(
            xaxis_title="Amount ($)",
            yaxis_title="Vendor",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#333333'
        )
        
        return fig
    
    def get_staff_summary(self):
        """Generate a summary of staff expenses"""
        staff_df = self.get_staff_data()
        metrics = self.calculate_staff_metrics()
        
        # Calculate month-over-month change
        monthly_expenses = staff_df.groupby(pd.to_datetime(staff_df['Date']).dt.to_period('M'))['Amount'].sum()
        if len(monthly_expenses) > 1:
            current_month = monthly_expenses.index[-1]
            prev_month = monthly_expenses.index[-2]
            mom_change = ((monthly_expenses[current_month] - monthly_expenses[prev_month]) / monthly_expenses[prev_month]) * 100
        else:
            mom_change = 0
        
        # Calculate average monthly expense
        avg_monthly = monthly_expenses.mean()
        
        # Calculate peak month
        peak_month = monthly_expenses.idxmax()
        peak_amount = monthly_expenses[peak_month]
        
        return {
            'total_expenses': metrics['total_staff_expenses'],
            'category_breakdown': metrics['category_breakdown'],
            'project_breakdown': metrics['project_breakdown'],
            'vendor_breakdown': metrics['vendor_breakdown'],
            'avg_transaction': metrics['avg_transaction'],
            'max_transaction': metrics['max_transaction'],
            'transaction_count': metrics['transaction_count'],
            'mom_change': mom_change,
            'avg_monthly': avg_monthly,
            'peak_month': peak_month,
            'peak_amount': peak_amount
        } 