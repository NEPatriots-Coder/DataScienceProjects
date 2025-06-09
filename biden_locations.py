import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class PresidentialDataAnalyzer:
    """
    Advanced analyzer for presidential activity data providing strategic insights
    """
    
    def __init__(self, file_path: str = None, data: pd.DataFrame = None):
        if file_path:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                self.df = pd.read_excel(file_path)
            else:
                self.df = pd.read_csv(file_path)
        elif data is not None:
            self.df = data.copy()
        else:
            raise ValueError("Either file_path or data must be provided")
        
        self.clean_and_enhance_data()
        
    def clean_and_enhance_data(self):
        """Clean data and create enhanced features for analysis"""
        # Basic cleaning
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Enhanced feature engineering
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6])
        
        # Clean boolean columns
        boolean_cols = ['in_washington_dc', 'location2_verified']
        for col in boolean_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].map({
                    'TRUE': True, 'FALSE': False, 
                    True: True, False: False
                }).fillna(False)
        
        # Analyze action complexity
        if 'action' in self.df.columns:
            self.df['action_word_count'] = self.df['action'].astype(str).apply(lambda x: len(x.split()))
            self.df['action_semicolon_count'] = self.df['action'].astype(str).str.count(';')
            self.df['action_complexity_score'] = self.df['action_word_count'] + (self.df['action_semicolon_count'] * 3)
        
        # Geographic analysis
        self.df['has_multiple_locations'] = (~self.df['location2'].isna()) & (self.df['location1'] != self.df['location2'])
        
        # Clean location data
        for col in ['location1', 'location2']:
            if col in self.df.columns:
                self.df[f'{col}_state'] = self.df[col].astype(str).str.extract(r',\s*([A-Z]{2}|[A-Za-z\s]+)$')[0]
                self.df[f'{col}_is_international'] = ~self.df[col].astype(str).str.contains(r',\s*[A-Z]{2}$', na=False)
    
    def analyze_activity_patterns(self) -> Dict:
        """Deep analysis of activity patterns and trends"""
        insights = {}
        
        # Temporal pattern analysis
        daily_activity = self.df.groupby('day_of_week').size()
        weekend_vs_weekday = {
            'weekend_activities': self.df[self.df['is_weekend']].shape[0],
            'weekday_activities': self.df[~self.df['is_weekend']].shape[0],
            'weekend_percentage': (self.df[self.df['is_weekend']].shape[0] / len(self.df)) * 100
        }
        
        # Activity intensity analysis
        if 'action_complexity_score' in self.df.columns:
            complexity_stats = {
                'avg_complexity': self.df['action_complexity_score'].mean(),
                'high_complexity_threshold': self.df['action_complexity_score'].quantile(0.75),
                'high_complexity_days': len(self.df[self.df['action_complexity_score'] > self.df['action_complexity_score'].quantile(0.75)])
            }
            insights['complexity_analysis'] = complexity_stats
        
        # Geographic mobility analysis
        if 'in_washington_dc' in self.df.columns:
            mobility_patterns = {
                'dc_percentage': (self.df['in_washington_dc'].sum() / len(self.df)) * 100,
                'travel_frequency': len(self.df[~self.df['in_washington_dc']]),
                'consecutive_dc_days': self._find_consecutive_patterns(self.df['in_washington_dc']),
                'consecutive_travel_days': self._find_consecutive_patterns(~self.df['in_washington_dc'])
            }
            insights['mobility_patterns'] = mobility_patterns
        
        insights['temporal_patterns'] = {
            'daily_distribution': daily_activity.to_dict(),
            'weekend_analysis': weekend_vs_weekday,
            'most_active_day': daily_activity.idxmax(),
            'least_active_day': daily_activity.idxmin()
        }
        
        return insights
    
    def _find_consecutive_patterns(self, boolean_series) -> Dict:
        """Find patterns of consecutive True values"""
        if boolean_series.empty:
            return {'max_consecutive': 0, 'avg_consecutive': 0}
            
        # Find consecutive streaks
        streaks = []
        current_streak = 0
        
        for value in boolean_series:
            if value:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        return {
            'max_consecutive': max(streaks) if streaks else 0,
            'avg_consecutive': np.mean(streaks) if streaks else 0,
            'total_streaks': len(streaks)
        }
    
    def analyze_location_strategy(self) -> Dict:
        """Strategic analysis of location choices and patterns"""
        location_insights = {}
        
        # State-level analysis
        if 'location1_state' in self.df.columns:
            state_visits = self.df['location1_state'].value_counts()
            
            # Strategic states analysis (swing states, key political states)
            strategic_states = ['Pennsylvania', 'Michigan', 'Wisconsin', 'Arizona', 'Georgia', 
                              'North Carolina', 'Florida', 'Nevada', 'PA', 'MI', 'WI', 'AZ', 'GA', 'NC', 'FL', 'NV']
            
            strategic_visits = 0
            for state in strategic_states:
                strategic_visits += state_visits.get(state, 0)
            
            location_insights['state_strategy'] = {
                'total_states_visited': state_visits.nunique(),
                'strategic_state_visits': strategic_visits,
                'strategic_percentage': (strategic_visits / len(self.df)) * 100,
                'top_states': state_visits.head(10).to_dict(),
                'concentration_index': self._calculate_concentration_index(state_visits)
            }
        
        # International vs domestic analysis
        if 'location1_is_international' in self.df.columns:
            international_analysis = {
                'international_trips': self.df['location1_is_international'].sum(),
                'domestic_trips': (~self.df['location1_is_international']).sum(),
                'international_percentage': (self.df['location1_is_international'].sum() / len(self.df)) * 100
            }
            location_insights['international_strategy'] = international_analysis
        
        return location_insights
    
    def _calculate_concentration_index(self, series) -> float:
        """Calculate Herfindahl-Hirschman Index for geographic concentration"""
        if series.empty:
            return 0
        proportions = series / series.sum()
        return (proportions ** 2).sum()
    
    def detect_anomalies_and_patterns(self) -> Dict:
        """Detect unusual patterns and anomalies in presidential activities"""
        anomalies = {}
        
        # Unusual activity days
        if 'action_complexity_score' in self.df.columns:
            complexity_mean = self.df['action_complexity_score'].mean()
            complexity_std = self.df['action_complexity_score'].std()
            
            # Days with unusually high activity
            high_activity_threshold = complexity_mean + (2 * complexity_std)
            high_activity_days = self.df[self.df['action_complexity_score'] > high_activity_threshold]
            
            anomalies['high_activity_days'] = {
                'count': len(high_activity_days),
                'dates': high_activity_days['date'].dt.strftime('%Y-%m-%d').tolist()[:10],  # Top 10
                'avg_complexity_score': high_activity_days['action_complexity_score'].mean()
            }
        
        # Unusual travel patterns
        if 'in_washington_dc' in self.df.columns:
            # Detect unusual clustering of travel
            travel_days = self.df[~self.df['in_washington_dc']]['date']
            if len(travel_days) > 1:
                travel_gaps = travel_days.diff().dt.days.dropna()
                unusual_travel_clusters = travel_gaps[travel_gaps == 1].count()  # Consecutive travel days
                
                anomalies['travel_patterns'] = {
                    'consecutive_travel_instances': unusual_travel_clusters,
                    'avg_days_between_travel': travel_gaps.mean()
                }
        
        return anomalies
    
    def generate_strategic_insights(self) -> str:
        """Generate actionable strategic insights"""
        activity_patterns = self.analyze_activity_patterns()
        location_strategy = self.analyze_location_strategy()
        anomalies = self.detect_anomalies_and_patterns()
        
        report = f"""
STRATEGIC PRESIDENTIAL ACTIVITY ANALYSIS
{'='*60}

🎯 KEY STRATEGIC INSIGHTS:

📊 ACTIVITY INTENSITY PATTERNS:
"""
        
        # Activity patterns insights
        if 'complexity_analysis' in activity_patterns:
            complexity = activity_patterns['complexity_analysis']
            report += f"""
• Average activity complexity: {complexity['avg_complexity']:.1f}
• High-intensity days: {complexity['high_complexity_days']} ({(complexity['high_complexity_days']/len(self.df)*100):.1f}%)
• Strategic implication: {'High activity periods may indicate crisis management or major policy pushes' if complexity['high_complexity_days'] > len(self.df)*0.2 else 'Relatively consistent activity levels suggest steady governance approach'}
"""
        
        # Temporal strategy
        temporal = activity_patterns['temporal_patterns']
        weekend_pct = temporal['weekend_analysis']['weekend_percentage']
        
        report += f"""
📅 TEMPORAL STRATEGY:
• Weekend activity rate: {weekend_pct:.1f}%
• Most active day: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][temporal['most_active_day']]}
• Strategic implication: {'High weekend activity suggests crisis-driven schedule or public engagement strategy' if weekend_pct > 20 else 'Traditional weekday-focused schedule indicates structured governance approach'}
"""
        
        # Geographic strategy
        if 'state_strategy' in location_strategy:
            state_strategy = location_strategy['state_strategy']
            concentration = state_strategy['concentration_index']
            
            report += f"""
🗺️ GEOGRAPHIC STRATEGY:
• States visited: {state_strategy['total_states_visited']}
• Strategic state focus: {state_strategy['strategic_percentage']:.1f}% of activities
• Geographic concentration index: {concentration:.3f}
• Strategic implication: {'Highly concentrated travel suggests targeted political strategy' if concentration > 0.1 else 'Broad geographic distribution indicates national unity approach'}

Top Target States:
"""
            for state, count in list(state_strategy['top_states'].items())[:5]:
                pct = (count / len(self.df)) * 100
                report += f"  • {state}: {count} visits ({pct:.1f}%)\n"
        
        # Mobility patterns
        if 'mobility_patterns' in activity_patterns:
            mobility = activity_patterns['mobility_patterns']
            dc_pct = mobility['dc_percentage']
            
            report += f"""
🏛️ MOBILITY & PRESENCE STRATEGY:
• Washington DC presence: {dc_pct:.1f}%
• Travel frequency: {mobility['travel_frequency']} trips
• Max consecutive DC days: {mobility['consecutive_dc_days']['max_consecutive']}
• Max consecutive travel days: {mobility['consecutive_travel_days']['max_consecutive']}
• Strategic implication: {'High DC presence suggests policy-focused presidency' if dc_pct > 70 else 'High travel frequency suggests public engagement and coalition-building strategy'}
"""
        
        # Anomaly insights
        if anomalies:
            report += f"""
⚠️ NOTABLE PATTERNS & ANOMALIES:
"""
            if 'high_activity_days' in anomalies:
                high_activity = anomalies['high_activity_days']
                report += f"• Identified {high_activity['count']} unusually high-activity days\n"
                if high_activity['dates']:
                    report += f"• Recent high-activity dates: {', '.join(high_activity['dates'][:3])}\n"
                report += f"• Strategic implication: Cluster of high-activity days may indicate major policy initiatives or crisis response\n"
        
        # Recommendations
        report += f"""
🎯 STRATEGIC RECOMMENDATIONS:
"""
        
        if 'state_strategy' in location_strategy:
            strategic_pct = location_strategy['state_strategy']['strategic_percentage']
            if strategic_pct < 30:
                report += "• Consider increasing visits to strategic swing states for electoral positioning\n"
            else:
                report += "• Strong strategic state engagement - maintain current geographic focus\n"
        
        if 'mobility_patterns' in activity_patterns:
            if activity_patterns['mobility_patterns']['dc_percentage'] > 80:
                report += "• Consider increasing travel to demonstrate national engagement\n"
            elif activity_patterns['mobility_patterns']['dc_percentage'] < 50:
                report += "• Consider balancing travel with Washington DC policy work\n"
        
        if weekend_pct > 25:
            report += "• High weekend activity may impact work-life balance - consider strategic scheduling\n"
        
        return report
    
    def create_advanced_visualizations(self, figsize: tuple = (16, 12)):
        """Create insightful visualizations"""
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('Strategic Presidential Activity Analysis Dashboard', fontsize=18, fontweight='bold')
        
        # 1. Activity intensity heatmap by day and week
        if 'week_of_year' in self.df.columns:
            pivot_data = self.df.groupby(['week_of_year', 'day_of_week']).size().unstack(fill_value=0)
            sns.heatmap(pivot_data, ax=axes[0, 0], cmap='YlOrRd', cbar_kws={'label': 'Activities'})
            axes[0, 0].set_title('Activity Intensity: Week vs Day Pattern')
            axes[0, 0].set_xlabel('Day of Week (0=Mon, 6=Sun)')
            axes[0, 0].set_ylabel('Week of Year')
        
        # 2. Travel vs DC presence over time
        if 'in_washington_dc' in self.df.columns:
            monthly_travel = self.df.groupby(self.df['date'].dt.to_period('M')).agg({
                'in_washington_dc': ['sum', 'count']
            }).round(2)
            monthly_travel.columns = ['DC_days', 'total_days']
            monthly_travel['travel_pct'] = ((monthly_travel['total_days'] - monthly_travel['DC_days']) / monthly_travel['total_days'] * 100)
            
            axes[0, 1].bar(range(len(monthly_travel)), monthly_travel['travel_pct'], color='steelblue', alpha=0.7)
            axes[0, 1].set_title('Travel Percentage by Month')
            axes[0, 1].set_ylabel('% Time Traveling')
            axes[0, 1].set_xlabel('Month')
        
        # 3. Geographic concentration analysis
        if 'location1_state' in self.df.columns:
            state_counts = self.df['location1_state'].value_counts().head(10)
            axes[1, 0].barh(range(len(state_counts)), state_counts.values, color='darkgreen', alpha=0.7)
            axes[1, 0].set_yticks(range(len(state_counts)))
            axes[1, 0].set_yticklabels(state_counts.index)
            axes[1, 0].set_title('Top 10 States by Visit Frequency')
            axes[1, 0].set_xlabel('Number of Visits')
        
        # 4. Activity complexity distribution
        if 'action_complexity_score' in self.df.columns:
            axes[1, 1].hist(self.df['action_complexity_score'], bins=20, color='purple', alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(self.df['action_complexity_score'].mean(), color='red', linestyle='--', 
                              label=f'Mean: {self.df["action_complexity_score"].mean():.1f}')
            axes[1, 1].set_title('Activity Complexity Distribution')
            axes[1, 1].set_xlabel('Complexity Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        
        # 5. Weekend vs Weekday patterns
        weekend_comparison = self.df.groupby('is_weekend').agg({
            'action_complexity_score': 'mean' if 'action_complexity_score' in self.df.columns else lambda x: 0,
            'date': 'count'
        }).round(2)
        
        x_pos = [0, 1]
        axes[2, 0].bar(x_pos, weekend_comparison['date'], color=['lightblue', 'orange'], alpha=0.7)
        axes[2, 0].set_xticks(x_pos)
        axes[2, 0].set_xticklabels(['Weekday', 'Weekend'])
        axes[2, 0].set_title('Activity Count: Weekday vs Weekend')
        axes[2, 0].set_ylabel('Number of Activities')
        
        # 6. Time series with trend
        monthly_activity = self.df.groupby(self.df['date'].dt.to_period('M')).size()
        axes[2, 1].plot(range(len(monthly_activity)), monthly_activity.values, marker='o', linewidth=2)
        
        # Add trend line
        if len(monthly_activity) > 2:
            z = np.polyfit(range(len(monthly_activity)), monthly_activity.values, 1)
            p = np.poly1d(z)
            axes[2, 1].plot(range(len(monthly_activity)), p(range(len(monthly_activity))), "r--", alpha=0.8, linewidth=2)
        
        axes[2, 1].set_title('Activity Trend Over Time')
        axes[2, 1].set_xlabel('Month')
        axes[2, 1].set_ylabel('Activities per Month')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# Usage function
def main():
    """
    Run comprehensive presidential activity analysis
    """
    
    # PASTE YOUR EXCEL FILE PATH HERE
    excel_file_path = r"C:\path\to\your\excel\file.xlsx"  # Replace with your actual file path
    
    try:
        print("🔍 Loading and analyzing presidential activity data...")
        
        # Initialize analyzer
        analyzer = PresidentialDataAnalyzer(r"C:\Users\lamarw\Desktop\ComputerScience\DataScienceProjecs\Biden_Locations.xlsx")
        
        print(f"✅ Successfully loaded {len(analyzer.df)} records")
        print(f"📅 Date range: {analyzer.df['date'].min().strftime('%Y-%m-%d')} to {analyzer.df['date'].max().strftime('%Y-%m-%d')}")
        
        # Generate strategic insights
        strategic_report = analyzer.generate_strategic_insights()
        print(strategic_report)
        
        # Create advanced visualizations
        print("\n📊 Generating strategic visualizations...")
        fig = analyzer.create_advanced_visualizations()
        plt.show()
        
        # Export enhanced data with new features
        analyzer.df.to_csv("enhanced_presidential_data.csv", index=False)
        print("\n💾 Enhanced dataset exported to 'enhanced_presidential_data.csv'")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find the Excel file at: {excel_file_path}")
        print("Please check the file path and make sure the file exists.")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")

if __name__ == "__main__":
    main()