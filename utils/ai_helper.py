"""
AI Helper Module for Budget Visualization Dashboard
Provides natural language query processing and response generation for budget data
"""

import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from utils.styles import COLORS
import streamlit as st

class BudgetAI:
    """AI assistant for answering budget-related questions"""
    
    def __init__(self, df=None):
        """Initialize the AI helper with dataframe"""
        self.df = df
        self.keywords = {
            'total': ['total', 'sum', 'overall', 'all'],
            'average': ['average', 'avg', 'mean', 'typical'],
            'project': ['project', 'projects'],
            'budget_line': ['budget line', 'category', 'categories', 'budget category', 'expense category'],
            'monthly': ['monthly', 'month', 'months', 'per month'],
            'trend': ['trend', 'over time', 'pattern', 'history'],
            'highest': ['highest', 'most', 'largest', 'biggest', 'top', 'maximum', 'max'],
            'lowest': ['lowest', 'least', 'smallest', 'minimum', 'min'],
            'compare': ['compare', 'comparison', 'versus', 'vs'],
            'chart': ['chart', 'graph', 'plot', 'visualization', 'visualize', 'show me'],
            'pie': ['pie', 'pie chart', 'distribution'],
            'bar': ['bar', 'bar chart', 'column chart'],
            'line': ['line', 'line chart', 'time series', 'trend'],
            'filter': ['filter', 'only', 'just', 'specifically', 'for'],
            'export': ['export', 'download', 'save', 'generate', 'report', 'pdf', 'csv'],
            'date': ['date', 'year', 'month', 'period', 'time', 'when'],
            'threshold': ['exceed', 'above', 'below', 'over', 'under', 'more than', 'less than', 'greater than'],
            'negative': ['negative', 'zero', 'missing', 'empty', 'invalid'],
            'variance': ['variance', 'difference', 'change', 'delta', 'variation'],
            'department': ['department', 'team', 'group', 'division'],
            'breakdown': ['breakdown', 'break down', 'distribution', 'composition', 'split', 'divide', 'divided']
        }
        self.chart_types = {
            'pie': self._create_pie_chart,
            'bar': self._create_bar_chart,
            'line': self._create_line_chart,
        }
        
    def set_data(self, df):
        """Update the dataframe"""
        self.df = df
        
    def process_query(self, query):
        """Process a natural language query and return a response"""
        if self.df is None or self.df.empty:
            return "I don't have any budget data to analyze. Please upload a file first."
        
        query = query.lower().strip()
        
        # Preprocess the dataframe to ensure date is in datetime format
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Month'] = self.df['Date'].dt.to_period('M')
        
        # Match query to type of analysis
        if self._contains_keywords(query, self.keywords['total'] + self.keywords['project']):
            if self._contains_keywords(query, self.keywords['chart'] + self.keywords['bar']):
                return self._total_expenses_by_project_chart()
            return self._total_expenses_by_project()
            
        elif self._contains_keywords(query, self.keywords['highest'] + self.keywords['project']):
            return self._highest_spending_project()
            
        elif self._contains_keywords(query, self.keywords['budget_line']) and self._contains_keywords(query, self.keywords['breakdown']):
            if self._contains_keywords(query, self.keywords['pie']):
                return self._budget_line_breakdown_pie()
            else:
                return self._budget_line_breakdown()
                
        elif any(month in query for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']):
            return self._spending_by_specific_month(query)
            
        elif self._contains_keywords(query, self.keywords['monthly'] + self.keywords['trend']):
            return self._monthly_spend_trend()
            
        elif self._contains_keywords(query, self.keywords['average'] + self.keywords['project']):
            return self._average_expense_per_project()
            
        elif self._contains_keywords(query, self.keywords['monthly'] + self.keywords['average']):
            return self._average_monthly_spend()
            
        elif self._contains_keywords(query, self.keywords['negative']):
            return self._find_negative_entries()
            
        elif self._contains_keywords(query, self.keywords['threshold']):
            # Try to extract the threshold amount
            amount_match = re.search(r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*k?', query)
            threshold = 10000  # Default
            if amount_match:
                amount_str = amount_match.group(1).replace(',', '')
                threshold = float(amount_str)
                if 'k' in amount_match.group(0).lower():
                    threshold *= 1000
            return self._find_expenses_above_threshold(threshold)
            
        elif self._contains_keywords(query, self.keywords['variance']):
            return self._monthly_variance()
            
        elif self._contains_keywords(query, self.keywords['filter']):
            # Try to extract what to filter for
            project_match = re.search(r'for\s+"([^"]+)"|for\s+([^\s.,]+)', query)
            if project_match:
                filter_term = project_match.group(1) if project_match.group(1) else project_match.group(2)
                return self._filter_by_term(filter_term)
            else:
                return "I'm not sure which project or category you want to filter for. Can you specify a project or budget line name?"
        
        elif self._contains_keywords(query, self.keywords['pie']):
            return self._create_pie_chart(self.df.groupby('Budget Line')['Amount'].sum().reset_index(), 
                                         'Budget Line', 'Amount', 'Expense Distribution by Budget Line')
        
        elif self._contains_keywords(query, self.keywords['bar'] + ['project']):
            return self._create_bar_chart(self.df.groupby('Project')['Amount'].sum().reset_index(),
                                         'Project', 'Amount', 'Total Expenses by Project')
                                         
        elif self._contains_keywords(query, self.keywords['export']) and ('pdf' in query or 'report' in query):
            return {"type": "instruction", "action": "export_pdf"}
            
        else:
            return "I'm not sure how to answer that question about the budget data. Try asking about total expenses, project breakdown, monthly trends, or specific filtering."
    
    def _contains_keywords(self, text, keywords):
        """Check if any of the keywords are in the text"""
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                return True
        return False
        
    def _total_expenses_by_project(self):
        """Calculate total expenses by project"""
        project_totals = self.df.groupby('Project')['Amount'].sum().sort_values(ascending=False)
        response = "### Total Expense by Project\n\n"
        
        for project, amount in project_totals.items():
            response += f"- **{project}**: ${amount:,.2f}\n"
            
        total = project_totals.sum()
        response += f"\n**Total**: ${total:,.2f}"
        
        return response
        
    def _total_expenses_by_project_chart(self):
        """Create a bar chart of total expenses by project"""
        project_totals = self.df.groupby('Project')['Amount'].sum().reset_index().sort_values('Amount', ascending=False)
        
        fig = px.bar(
            project_totals,
            x='Project',
            y='Amount',
            title='Total Expenses by Project',
            color_discrete_sequence=[COLORS['primary'], COLORS['secondary']]
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text'],
            title_font_color=COLORS['text'],
            xaxis_title="Project",
            yaxis_title="Amount ($)",
            title_font_size=16
        )
        
        # Format y-axis labels as currency
        fig.update_yaxes(tickprefix="$")
        
        return {"type": "chart", "figure": fig}
        
    def _highest_spending_project(self):
        """Find the project with the highest spending"""
        project_totals = self.df.groupby('Project')['Amount'].sum()
        highest_project = project_totals.idxmax()
        highest_amount = project_totals.max()
        
        response = f"The project with the highest spending is **{highest_project}** with a total of **${highest_amount:,.2f}**.\n\n"
        
        # Add comparison to average
        avg_amount = project_totals.mean()
        pct_above_avg = (highest_amount - avg_amount) / avg_amount * 100
        
        response += f"This is ${highest_amount - avg_amount:,.2f} or {pct_above_avg:.1f}% above the average project spending of ${avg_amount:,.2f}."
        
        return response
        
    def _budget_line_breakdown(self):
        """Show breakdown of expenses by budget line"""
        budget_totals = self.df.groupby('Budget Line')['Amount'].sum().sort_values(ascending=False)
        total = budget_totals.sum()
        
        response = "### Expense Breakdown by Budget Line\n\n"
        
        for budget, amount in budget_totals.items():
            percentage = (amount / total) * 100
            response += f"- **{budget}**: ${amount:,.2f} ({percentage:.1f}%)\n"
            
        response += f"\n**Total**: ${total:,.2f}"
        
        return response
        
    def _budget_line_breakdown_pie(self):
        """Create a pie chart of budget line breakdown"""
        budget_totals = self.df.groupby('Budget Line')['Amount'].sum().reset_index()
        
        fig = px.pie(
            budget_totals,
            values='Amount',
            names='Budget Line',
            title='Expense Distribution by Budget Line',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text'],
            title_font_color=COLORS['text'],
            title_font_size=16
        )
        
        # Add hover data with dollar values
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.2f}<br>Percentage: %{percent}'
        )
        
        return {"type": "chart", "figure": fig}
        
    def _monthly_spend_trend(self):
        """Analyze the monthly spend trend"""
        monthly_totals = self.df.groupby(self.df['Date'].dt.to_period('M').astype(str))['Amount'].sum().reset_index()
        monthly_totals = monthly_totals.sort_values('Date')
        
        # Get last 6 months if available
        if len(monthly_totals) > 6:
            monthly_totals = monthly_totals.tail(6)
            
        response = "### Monthly Spend Trend\n\n"
        
        for _, row in monthly_totals.iterrows():
            response += f"- **{row['Date']}**: ${row['Amount']:,.2f}\n"
            
        # Calculate trend
        if len(monthly_totals) > 1:
            first_month = monthly_totals.iloc[0]['Amount']
            last_month = monthly_totals.iloc[-1]['Amount']
            change = ((last_month - first_month) / first_month) * 100
            
            trend_direction = "increased" if change > 0 else "decreased"
            response += f"\nSpending has {trend_direction} by {abs(change):.1f}% from {monthly_totals.iloc[0]['Date']} to {monthly_totals.iloc[-1]['Date']}."
            
            # Create line chart
            fig = px.line(
                monthly_totals,
                x='Date',
                y='Amount',
                title='Monthly Expense Trend',
                markers=True
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                title_font_color=COLORS['text'],
                xaxis_title="Month",
                yaxis_title="Amount ($)",
                title_font_size=16
            )
            
            # Format y-axis labels as currency
            fig.update_yaxes(tickprefix="$")
            
            # Add gradient fill
            fig.add_traces(
                go.Scatter(
                    x=monthly_totals['Date'],
                    y=monthly_totals['Amount'],
                    fill='tozeroy',
                    fillcolor='rgba(74, 144, 226, 0.2)',
                    line=dict(color=COLORS['primary']),
                    showlegend=False
                )
            )
            
            return {"type": "chart", "figure": fig, "text": response}
        
        return response
        
    def _average_expense_per_project(self):
        """Calculate the average expense per project"""
        project_totals = self.df.groupby('Project')['Amount'].sum()
        avg_expense = project_totals.mean()
        
        response = f"The average expense per project is **${avg_expense:,.2f}**.\n\n"
        
        # Add min and max for context
        min_project = project_totals.idxmin()
        min_amount = project_totals.min()
        max_project = project_totals.idxmax()
        max_amount = project_totals.max()
        
        response += f"The project with the lowest expense is **{min_project}** (${min_amount:,.2f}).\n"
        response += f"The project with the highest expense is **{max_project}** (${max_amount:,.2f})."
        
        return response
        
    def _spending_by_specific_month(self, query):
        """Find spending in a specific month mentioned in the query"""
        # Extract month and possibly year from query
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        month_match = None
        for month_name, month_num in months.items():
            if month_name in query:
                month_match = (month_name, month_num)
                break
                
        if not month_match:
            return "I couldn't identify which month you're asking about."
            
        month_name, month_num = month_match
        
        # Check for year in query
        year_match = re.search(r'\b(20\d{2})\b', query)
        year = int(year_match.group(1)) if year_match else None
        
        if year:
            # Filter data for specific month and year
            mask = (self.df['Date'].dt.month == month_num) & (self.df['Date'].dt.year == year)
            month_year_str = f"{month_name.capitalize()} {year}"
        else:
            # If no year specified, find the most recent occurrence of that month
            all_years = self.df[self.df['Date'].dt.month == month_num]['Date'].dt.year.unique()
            
            if len(all_years) == 0:
                return f"I don't have any data for {month_name.capitalize()}."
                
            year = max(all_years)
            mask = (self.df['Date'].dt.month == month_num) & (self.df['Date'].dt.year == year)
            month_year_str = f"{month_name.capitalize()} {year}"
            
        month_data = self.df[mask]
        
        if len(month_data) == 0:
            return f"I don't have any expense data for {month_year_str}."
            
        total_amount = month_data['Amount'].sum()
        
        response = f"In **{month_year_str}**, the total expense was **${total_amount:,.2f}**.\n\n"
        
        # Add breakdown by project
        project_totals = month_data.groupby('Project')['Amount'].sum().sort_values(ascending=False)
        
        response += "### Breakdown by Project\n\n"
        for project, amount in project_totals.items():
            percentage = (amount / total_amount) * 100
            response += f"- **{project}**: ${amount:,.2f} ({percentage:.1f}%)\n"
            
        return response
        
    def _average_monthly_spend(self):
        """Calculate the average monthly spend"""
        monthly_totals = self.df.groupby(self.df['Date'].dt.to_period('M'))['Amount'].sum()
        avg_monthly = monthly_totals.mean()
        
        response = f"The average monthly spending is **${avg_monthly:,.2f}**.\n\n"
        
        # Find months with highest and lowest spending
        max_month = monthly_totals.idxmax()
        max_amount = monthly_totals.max()
        min_month = monthly_totals.idxmin()
        min_amount = monthly_totals.min()
        
        response += f"The month with the highest spending was **{max_month}** (${max_amount:,.2f}).\n"
        response += f"The month with the lowest spending was **{min_month}** (${min_amount:,.2f})."
        
        return response
        
    def _find_negative_entries(self):
        """Identify any negative or zero expense entries"""
        negative_entries = self.df[self.df['Amount'] <= 0]
        
        if len(negative_entries) == 0:
            return "There are no negative or zero expense entries in the data."
            
        response = f"I found **{len(negative_entries)}** entries with negative or zero amounts:\n\n"
        
        # Group by Project for better readability
        grouped = negative_entries.groupby('Project')
        
        for project, group in grouped:
            response += f"### Project: {project}\n\n"
            
            for _, row in group.iterrows():
                response += f"- **{row['Date'].strftime('%Y-%m-%d')}**: {row['Budget Line']} - ${row['Amount']:,.2f} - {row['Expense']}\n"
                
        return response
        
    def _find_expenses_above_threshold(self, threshold):
        """Find months where expenses exceeded a threshold"""
        monthly_totals = self.df.groupby(self.df['Date'].dt.to_period('M'))['Amount'].sum()
        months_above = monthly_totals[monthly_totals > threshold]
        
        if len(months_above) == 0:
            return f"There are no months where expenses exceeded ${threshold:,.2f}."
            
        response = f"I found **{len(months_above)}** months where expenses exceeded ${threshold:,.2f}:\n\n"
        
        for month, amount in months_above.items():
            response += f"- **{month}**: ${amount:,.2f}\n"
            
        # Create bar chart
        months_df = pd.DataFrame({'Month': months_above.index.astype(str), 'Amount': months_above.values})
        
        fig = px.bar(
            months_df,
            x='Month',
            y='Amount',
            title=f'Months Exceeding ${threshold:,.2f} in Expenses',
            color_discrete_sequence=[COLORS['primary']]
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text'],
            title_font_color=COLORS['text'],
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            title_font_size=16
        )
        
        # Add threshold line
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=threshold,
            x1=len(months_df)-0.5,
            y1=threshold,
            line=dict(
                color="red",
                width=2,
                dash="dash",
            )
        )
        
        # Add threshold annotation
        fig.add_annotation(
            x=len(months_df)/2,
            y=threshold*1.05,
            text=f"Threshold: ${threshold:,.2f}",
            showarrow=False,
            font=dict(color="red")
        )
        
        # Format y-axis labels as currency
        fig.update_yaxes(tickprefix="$")
        
        return {"type": "chart", "figure": fig, "text": response}
        
    def _monthly_variance(self):
        """Calculate month-to-month variance in spending"""
        monthly_totals = self.df.groupby(self.df['Date'].dt.to_period('M'))['Amount'].sum().sort_index()
        
        if len(monthly_totals) < 2:
            return "I need at least two months of data to calculate variance."
            
        # Calculate month-over-month changes
        monthly_changes = monthly_totals.pct_change() * 100
        
        response = "### Month-to-Month Spending Variance\n\n"
        
        for month, change in monthly_changes.items():
            if pd.isna(change):
                continue
                
            direction = "increase" if change > 0 else "decrease"
            response += f"- **{month}**: {abs(change):.1f}% {direction} from previous month\n"
            
        # Identify month with highest variance
        max_variance_month = monthly_changes.abs().idxmax()
        max_variance = monthly_changes.loc[max_variance_month]
        
        if not pd.isna(max_variance):
            direction = "increase" if max_variance > 0 else "decrease"
            response += f"\nThe month with the highest variance was **{max_variance_month}** with a {abs(max_variance):.1f}% {direction} from the previous month."
            
        return response
        
    def _filter_by_term(self, term):
        """Filter expenses by a specific term (project, budget line, or expense description)"""
        # Try to find matches in projects
        project_matches = [p for p in self.df['Project'].unique() if term.lower() in p.lower()]
        
        # Try to find matches in budget lines
        budget_matches = [b for b in self.df['Budget Line'].unique() if term.lower() in b.lower()]
        
        # Try to find matches in expense descriptions
        desc_matches = self.df[self.df['Expense'].str.contains(term, case=False)]
        
        response = ""
        
        if project_matches:
            # Filter for the first matching project
            project = project_matches[0]
            project_data = self.df[self.df['Project'] == project]
            total = project_data['Amount'].sum()
            
            response += f"### Expenses for Project: {project}\n\n"
            response += f"Total amount: **${total:,.2f}**\n\n"
            
            # Breakdown by budget line
            budget_totals = project_data.groupby('Budget Line')['Amount'].sum().sort_values(ascending=False)
            
            response += "#### Breakdown by Budget Line\n\n"
            for budget, amount in budget_totals.items():
                percentage = (amount / total) * 100
                response += f"- **{budget}**: ${amount:,.2f} ({percentage:.1f}%)\n"
                
            # Create pie chart
            fig = px.pie(
                project_data.groupby('Budget Line')['Amount'].sum().reset_index(),
                values='Amount',
                names='Budget Line',
                title=f'Expense Distribution for {project}',
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                title_font_color=COLORS['text'],
                title_font_size=16
            )
            
            return {"type": "chart", "figure": fig, "text": response}
            
        elif budget_matches:
            # Filter for the first matching budget line
            budget = budget_matches[0]
            budget_data = self.df[self.df['Budget Line'] == budget]
            total = budget_data['Amount'].sum()
            
            response += f"### Expenses for Budget Line: {budget}\n\n"
            response += f"Total amount: **${total:,.2f}**\n\n"
            
            # Breakdown by project
            project_totals = budget_data.groupby('Project')['Amount'].sum().sort_values(ascending=False)
            
            response += "#### Breakdown by Project\n\n"
            for project, amount in project_totals.items():
                percentage = (amount / total) * 100
                response += f"- **{project}**: ${amount:,.2f} ({percentage:.1f}%)\n"
                
            # Create bar chart
            fig = px.bar(
                budget_data.groupby('Project')['Amount'].sum().reset_index().sort_values('Amount', ascending=False),
                x='Project',
                y='Amount',
                title=f'Project Breakdown for {budget}',
                color_discrete_sequence=[COLORS['primary']]
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=COLORS['text'],
                title_font_color=COLORS['text'],
                xaxis_title="Project",
                yaxis_title="Amount ($)",
                title_font_size=16
            )
            
            # Format y-axis labels as currency
            fig.update_yaxes(tickprefix="$")
            
            return {"type": "chart", "figure": fig, "text": response}
            
        elif not desc_matches.empty:
            # Show all expenses matching the description
            total = desc_matches['Amount'].sum()
            
            response += f"### Expenses matching '{term}'\n\n"
            response += f"Total amount: **${total:,.2f}**\n\n"
            
            # Group by project and budget line
            grouped = desc_matches.groupby(['Project', 'Budget Line'])['Amount'].sum().reset_index()
            
            response += "#### Detailed Breakdown\n\n"
            for _, row in grouped.sort_values('Amount', ascending=False).iterrows():
                response += f"- **{row['Project']}** - {row['Budget Line']}: ${row['Amount']:,.2f}\n"
                
            return response
            
        else:
            return f"I couldn't find any expenses related to '{term}'. Try a different search term."
    
    def _create_pie_chart(self, df, names_col, values_col, title):
        """Create a pie chart visualization"""
        fig = px.pie(
            df,
            values=values_col,
            names=names_col,
            title=title,
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text'],
            title_font_color=COLORS['text'],
            title_font_size=16
        )
        
        # Add hover data with dollar values
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.2f}<br>Percentage: %{percent}'
        )
        
        return {"type": "chart", "figure": fig}
        
    def _create_bar_chart(self, df, x_col, y_col, title):
        """Create a bar chart visualization"""
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            title=title,
            color_discrete_sequence=[COLORS['primary']]
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text'],
            title_font_color=COLORS['text'],
            title_font_size=16
        )
        
        # Format y-axis labels as currency
        fig.update_yaxes(tickprefix="$")
        
        return {"type": "chart", "figure": fig}
        
    def _create_line_chart(self, df, x_col, y_col, title):
        """Create a line chart visualization"""
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            title=title,
            markers=True,
            color_discrete_sequence=[COLORS['primary']]
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=COLORS['text'],
            title_font_color=COLORS['text'],
            title_font_size=16
        )
        
        # Format y-axis labels as currency
        fig.update_yaxes(tickprefix="$")
        
        # Add gradient fill
        fig.add_traces(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                fill='tozeroy',
                fillcolor='rgba(74, 144, 226, 0.2)',
                line=dict(color=COLORS['primary']),
                showlegend=False
            )
        )
        
        return {"type": "chart", "figure": fig} 