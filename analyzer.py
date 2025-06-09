from typing import List, Dict, Union, Optional
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import os
import openpyxl
import matplotlib

@dataclass
class LocationMetrics:
    mean: float
    median: float
    mode: List[float]
    sum: float

@dataclass
class DispersionMetrics:
    std_dev: float
    variance: float
    range: float
    iqr: float

class StatisticalAnalyzer:
    def __init__(self, data: Union[pd.Series, np.ndarray, List]):
        """Initialize with data that can be converted to pandas Series.

        Args:
            data: Input data (pandas Series, numpy array, or list).
        
        Raises:
            ValueError: If data is empty or non-numeric for statistical analysis.
        """
        if len(data) == 0:
            raise ValueError("Input data cannot be empty.")
        self.data = pd.Series(data) if not isinstance(data, pd.Series) else data
        if not np.issubdtype(np.dtype(self.data.dtype), np.number):
            raise ValueError("Data must be numeric for statistical analysis.")
        else: 
            print(f"Data type is numeric: {self.data.dtype}")
        self._location_metrics: Optional[LocationMetrics] = None
        self._dispersion_metrics: Optional[DispersionMetrics] = None
    
    @property
    def location_metrics(self) -> LocationMetrics:
        """Compute and cache location metrics."""
        if self._location_metrics is None:
            mode_values = list(self.data.mode())
            self._location_metrics = LocationMetrics(
                mean=float(self.data.mean()),
                median=float(self.data.median()),
                mode=mode_values if mode_values else [],
                sum=float(self.data.sum())
            )
        return self._location_metrics
    
    @property
    def dispersion_metrics(self) -> DispersionMetrics:
        """Compute and cache dispersion metrics."""
        if self._dispersion_metrics is None:
            self._dispersion_metrics = DispersionMetrics(
                std_dev=float(self.data.std()),
                variance=float(self.data.var()),
                range=float(self.data.max() - self.data.min()),
                iqr=float(self.data.quantile(0.75) - self.data.quantile(0.25))
            )
        return self._dispersion_metrics

    def normality_test(self) -> Dict[str, float]:
        """Perform Shapiro-Wilk test for normality."""
        stat, p_value = stats.shapiro(self.data)
        return {"statistic": float(stat), "p_value": float(p_value)}

class FrequencyAnalyzer:
    def __init__(self, data: Union[pd.Series, np.ndarray, List]):
        """Initialize with data for frequency analysis."""
        self.data = pd.Series(data)
    
    def frequency_table(self, bins: Optional[int] = None, is_continuous: bool = False) -> pd.DataFrame:
        """Generate frequency table with absolute, relative, and cumulative frequencies.

        Args:
            bins: Number of bins for continuous data (optional).
            is_continuous: Treat data as continuous if True, discrete otherwise.
        
        Returns:
            pd.DataFrame: Frequency table with bin/value, absolute, relative, and cumulative frequencies.
        """
        if is_continuous and bins:
            hist, bin_edges = np.histogram(self.data, bins=bins)
            df = pd.DataFrame({
                'bin_start': bin_edges[:-1].round(2),
                'bin_end': bin_edges[1:].round(2),
                'absolute_freq': hist
            })
            df['relative_freq'] = (df['absolute_freq'] / df['absolute_freq'].sum()).round(4)
            df['cumulative_freq'] = df['absolute_freq'].cumsum()
        else:
            freq = self.data.value_counts().reset_index()
            freq.columns = ['value', 'absolute_freq']
            freq['relative_freq'] = (freq['absolute_freq'] / freq['absolute_freq'].sum()).round(4)
            freq['cumulative_freq'] = freq['absolute_freq'].cumsum()
            df = freq.sort_values('value')
        return df

class Visualizer:
    @staticmethod
    def histogram(data: pd.Series, bins: int = 30, title: str = "Histogram", 
                 save_path: Optional[str] = None, kde: bool = False, show_nominal: bool = False):
        """Create and optionally save a histogram with KDE option.
        
        Args:
            data: Data to plot
            bins: Number of bins for histogram
            title: Plot title
            save_path: Path to save the figure
            kde: Whether to overlay a Kernel Density Estimate
            show_nominal: Whether to overlay a normal distribution curve
        """
        plt.figure(figsize=(10, 6))
        
        # Use seaborn for histogram with KDE option
        if kde:
            ax = sns.histplot(data, bins=bins, kde=True, color='skyblue', edgecolor='black')
        else:
            ax = plt.hist(data, bins=bins, edgecolor='black', color='skyblue')
            
        # Add nominal distribution curve if requested
        if show_nominal:
            x = np.linspace(data.min(), data.max(), 100)
            mean = data.mean()
            std = data.std()
            y = stats.norm.pdf(x, mean, std) * len(data) * (data.max() - data.min()) / bins
            plt.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')
            plt.legend()
            
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        return plt.gcf()
    
    @staticmethod
    def box_plot(data: pd.Series, title: str = "Box Plot", save_path: Optional[str] = None):
        """Create and optionally save a box plot."""
        plt.figure(figsize=(10, 6))
        plt.boxplot(data, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        plt.title(title)
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path)
        plt.show()
        return plt.gcf()
    
    @staticmethod
    def scatter_plot(x: pd.Series, y: pd.Series, title: str = "Scatter Plot", save_path: Optional[str] = None):
        """Create and optionally save a scatter plot."""
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.5, color='purple')
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path)
        plt.show()
        return plt.gcf()

class StatisticalAnalysis:
    def __init__(self, data: Union[pd.DataFrame, pd.Series] = None, file_path: Optional[str] = None):
        """Initialize with data or load from file.

        Args:
            data: Input data (pandas DataFrame or Series).
            file_path: Path to CSV or Excel file (optional).
        
        Raises:
            ValueError: If neither data nor file_path is provided, or if file format is unsupported.
        """
        if file_path:
            print(f"Attempting to load data from: {file_path}")
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
                print(f"Successfully loaded CSV with {len(self.data)} rows")
            elif file_path.endswith(('.xlsx', '.xls')):
                try:
                    print("Trying openpyxl engine...")
                    self.data = pd.read_excel(file_path, engine='openpyxl')
                    print(f"Successfully loaded Excel with openpyxl: {len(self.data)} rows")
                except Exception as e:
                    print(f"Error with openpyxl: {e}")
                    try:
                        print("Trying xlrd engine...")
                        self.data = pd.read_excel(file_path, engine='xlrd')
                        print(f"Successfully loaded Excel with xlrd: {len(self.data)} rows")
                    except Exception as e2:
                        print(f"Error with xlrd: {e2}")
                        # Try to read as CSV as last resort
                        try:
                            print("Trying to read as CSV...")
                            self.data = pd.read_csv(file_path)
                            print(f"Successfully read file as CSV: {len(self.data)} rows")
                        except Exception as e3:
                            print(f"Final error: {e3}")
                            raise ValueError(f"Could not read file {file_path} with any available method")
            else:
                raise ValueError("Unsupported file format. Use .csv, .xlsx, or .xls.")
        elif data is not None:
            self.data = data
            print(f"Using provided data with {len(self.data)} rows")
        else:
            raise ValueError("Must provide either data or file_path.")
        
        self.stats_analyzer = None
        self.freq_analyzer = None
        self.visualizer = Visualizer()
        
        # Print column names to help with debugging
        if isinstance(self.data, pd.DataFrame):
            print(f"Available columns: {list(self.data.columns)}")
    
    def analyze_column(self, column_name: str = None, bins: int = 30, is_continuous: bool = False,
                      save_plots: bool = False, output_dir: str = "plots") -> Dict:
        """Perform complete analysis on a single column.

        Args:
            column_name: Name of the column to analyze (required for DataFrame).
            bins: Number of bins for histogram and frequency table (if continuous).
            is_continuous: Treat data as continuous if True.
            save_plots: Save plots to files if True.
            output_dir: Directory to save plots.

        Returns:
            Dict: Analysis results including metrics, frequency table, and visualizations.
        
        Raises:
            ValueError: If column_name is invalid or data is not suitable.
        """
        if isinstance(self.data, pd.DataFrame):
            if not column_name:
                raise ValueError("Column name must be provided for DataFrame input.")
            if column_name not in self.data.columns:
                raise ValueError(f"Column '{column_name}' not found in data. Available columns: {list(self.data.columns)}")
            series = self.data[column_name]
            print(f"Selected column '{column_name}' with {len(series)} values")
            print(f"Sample values: {series.head().tolist()}")
        else:
            series = self.data
            print(f"Using Series data with {len(series)} values")
            print(f"Sample values: {series.head().tolist()}")
        
        # Check for numeric data
        if not pd.api.types.is_numeric_dtype(series):
            print(f"Warning: Column '{column_name}' is not numeric. Type: {series.dtype}")
            # Try to convert to numeric
            try:
                series = pd.to_numeric(series, errors='coerce')
                print(f"Converted to numeric. {series.isna().sum()} values were set to NaN")
                # Drop NaN values
                series = series.dropna()
                print(f"After dropping NaN values: {len(series)} values remain")
            except Exception as e:
                print(f"Error converting to numeric: {e}")
        
        if len(series) == 0:
            raise ValueError("No valid numeric data to analyze")
        
        analyzer = StatisticalAnalyzer(series)
        freq_analyzer = FrequencyAnalyzer(series)
        
        # Create output directory if saving plots
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            histogram_path = os.path.join(output_dir, f"{column_name or 'data'}_histogram.png")
            box_plot_path = os.path.join(output_dir, f"{column_name or 'data'}_box_plot.png")
            print(f"Plots will be saved to: {output_dir}")
        else:
            histogram_path = box_plot_path = None
        
        # Create histogram
        print("Generating histogram...")
        histogram = self.visualizer.histogram(
            series, 
            bins, 
            f"Histogram of {column_name or 'Data'}", 
            histogram_path,
            kde=True,
            show_nominal=True
        )
        
        # Create box plot
        print("Generating box plot...")
        box_plot = self.visualizer.box_plot(series, f"Box Plot of {column_name or 'Data'}", box_plot_path)
        
        # Calculate metrics
        print("Calculating statistics...")
        results = {
            "location_metrics": analyzer.location_metrics,
            "dispersion_metrics": analyzer.dispersion_metrics,
            "normality_test": analyzer.normality_test(),
            "frequency_table": freq_analyzer.frequency_table(bins if is_continuous else None, is_continuous),
            "visualizations": {
                "histogram": histogram,
                "box_plot": box_plot
            }
        }
        
        return results

    def save_results(self, results: Dict, output_path: str):
        """Save analysis results to a CSV file.

        Args:
            results: Analysis results dictionary.
            output_path: Path to save the results CSV.
        """
        metrics_df = pd.DataFrame({
            "Metric": ["Mean", "Median", "Mode", "Sum", "Std Dev", "Variance", "Range", "IQR", 
                      "Shapiro-Wilk Stat", "Shapiro-Wilk P-Value"],
            "Value": [
                results["location_metrics"].mean,
                results["location_metrics"].median,
                results["location_metrics"].mode[0] if results["location_metrics"].mode and len(results["location_metrics"].mode) > 0 else None,
                results["location_metrics"].sum,
                results["dispersion_metrics"].std_dev,
                results["dispersion_metrics"].variance,
                results["dispersion_metrics"].range,
                results["dispersion_metrics"].iqr,
                results["normality_test"]["statistic"],
                results["normality_test"]["p_value"]
            ]
        })
        metrics_df.to_csv(output_path, index=False)
        results["frequency_table"].to_csv(output_path.replace(".csv", "_freq_table.csv"), index=False)

# Example usage
if __name__ == "__main__":
    # Example 1: Generate and analyze sample data
    try:
        # Create sample data for testing
        sample_data = pd.DataFrame({
            "values": np.random.normal(loc=0, scale=1, size=1000),
            "other_col": np.random.randint(1, 10, 1000)
        })
        
        # Save sample data to CSV (not Excel)
        sample_csv_path = "sample_data.csv"
        sample_data.to_csv(sample_csv_path, index=False)
        print(f"Created sample data file: {sample_csv_path}")
        
        # Initialize with the CSV file
        analysis = StatisticalAnalysis(file_path="sprocket_data.csv")
        print("Successfully loaded data")
        
        # Analyze all numeric columns
        print("Analyzing all numeric columns...")
        
        # Get list of numeric columns
        numeric_columns = analysis.data.select_dtypes(include=['number']).columns.tolist()
        print(f"Found {len(numeric_columns)} numeric columns: {numeric_columns}")
        
        # Skip tolerance flag columns
        analysis_columns = [col for col in numeric_columns if not col.endswith('_InTolerance') and col != 'All_InTolerance']
        print(f"Analyzing {len(analysis_columns)} columns (excluding tolerance flags)")
        
        # Create a combined results dictionary
        all_results = {}
        
        for column in analysis_columns:
            print(f"\nAnalyzing column: {column}")
            results = analysis.analyze_column(column_name=column, bins=30, is_continuous=True, save_plots=True)
            all_results[column] = results
            
            # Print metrics for each column
            print(f"Mean: {results['location_metrics'].mean:.2f}")
            print(f"Standard Deviation: {results['dispersion_metrics'].std_dev:.2f}")
            print(f"Normality Test P-Value: {results['normality_test']['p_value']:.4f}")
        
        # Save results for the first column (for backward compatibility)
        if analysis_columns:
            analysis.save_results(all_results[analysis_columns[0]], "analysis_results.csv")
            print("Analysis complete. Results for first column saved to analysis_results.csv")
        print("Analysis complete. Results saved to analysis_results.csv")
        import matplotlib.pyplot as plt
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")