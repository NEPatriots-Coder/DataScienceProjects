from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import os
import logging
from contextlib import contextmanager
import warnings
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration settings for statistical analysis."""
    bins: int = 30
    alpha: float = 0.05
    figure_size: Tuple[int, int] = (12, 8)
    save_plots: bool = False
    output_dir: str = "analysis_output"
    dpi: int = 300
    outlier_method: str = "iqr"  # "iqr", "zscore", or "both"
    normality_tests: List[str] = field(default_factory=lambda: ["shapiro", "jarque_bera", "anderson"])

@dataclass
class LocationMetrics:
    """Central tendency metrics."""
    mean: float
    median: float
    mode: List[float]
    sum: float
    geometric_mean: Optional[float] = None
    harmonic_mean: Optional[float] = None

@dataclass
class DispersionMetrics:
    """Variability metrics."""
    std_dev: float
    variance: float
    range: float
    iqr: float
    coefficient_variation: float
    mad: float  # Median Absolute Deviation

@dataclass
class ShapeMetrics:
    """Distribution shape metrics."""
    skewness: float
    kurtosis: float
    excess_kurtosis: float

@dataclass
class OutlierResults:
    """Outlier detection results."""
    iqr_outliers: List[float]
    zscore_outliers: List[float]
    outlier_indices: Dict[str, List[int]]
    outlier_percentage: float
    outlier_summary: Dict[str, int]

class EnhancedStatisticalAnalyzer:
    """Enhanced statistical analyzer with comprehensive metrics and tests."""
    
    def __init__(self, data: Union[pd.Series, np.ndarray, List], config: AnalysisConfig = None):
        """Initialize with data and configuration.

        Args:
            data: Input data (pandas Series, numpy array, or list).
            config: Analysis configuration settings.
        
        Raises:
            ValueError: If data is empty or non-numeric for statistical analysis.
        """
        if len(data) == 0:
            raise ValueError("Input data cannot be empty.")
            
        self.data = pd.Series(data) if not isinstance(data, pd.Series) else data.copy()
        self.config = config or AnalysisConfig()
        
        # Clean and validate data
        self._clean_data()
        
        # Cache for computed metrics
        self._location_metrics: Optional[LocationMetrics] = None
        self._dispersion_metrics: Optional[DispersionMetrics] = None
        self._shape_metrics: Optional[ShapeMetrics] = None
        self._outlier_results: Optional[OutlierResults] = None
        
        logger.info(f"Initialized analyzer with {len(self.data)} data points")
    
    def _clean_data(self):
        """Clean and validate input data."""
        original_length = len(self.data)
        
        # Convert to numeric if possible
        if not pd.api.types.is_numeric_dtype(self.data):
            logger.warning(f"Data type {self.data.dtype} is not numeric. Attempting conversion...")
            self.data = pd.to_numeric(self.data, errors='coerce')
        
        # Remove NaN values
        self.data = self.data.dropna()
        
        if len(self.data) == 0:
            raise ValueError("No valid numeric data found after cleaning.")
        
        if len(self.data) < original_length:
            logger.warning(f"Removed {original_length - len(self.data)} invalid/missing values")
        
        # Reset index
        self.data = self.data.reset_index(drop=True)
    
    @property
    def location_metrics(self) -> LocationMetrics:
        """Compute and cache location metrics."""
        if self._location_metrics is None:
            mode_values = list(self.data.mode())
            
            # Geometric and harmonic means (only for positive values)
            geo_mean = None
            harm_mean = None
            if (self.data > 0).all():
                try:
                    geo_mean = float(stats.gmean(self.data))
                    harm_mean = float(stats.hmean(self.data))
                except:
                    logger.warning("Could not compute geometric/harmonic means")
            
            self._location_metrics = LocationMetrics(
                mean=float(self.data.mean()),
                median=float(self.data.median()),
                mode=mode_values,
                sum=float(self.data.sum()),
                geometric_mean=geo_mean,
                harmonic_mean=harm_mean
            )
        return self._location_metrics
    
    @property
    def dispersion_metrics(self) -> DispersionMetrics:
        """Compute and cache dispersion metrics."""
        if self._dispersion_metrics is None:
            mean_val = self.data.mean()
            std_val = self.data.std()
            
            self._dispersion_metrics = DispersionMetrics(
                std_dev=float(std_val),
                variance=float(self.data.var()),
                range=float(self.data.max() - self.data.min()),
                iqr=float(self.data.quantile(0.75) - self.data.quantile(0.25)),
                coefficient_variation=float(std_val / mean_val * 100) if mean_val != 0 else np.inf,
                mad=float(self.data.mad())  # Median Absolute Deviation
            )
        return self._dispersion_metrics
    
    @property
    def shape_metrics(self) -> ShapeMetrics:
        """Compute and cache shape metrics."""
        if self._shape_metrics is None:
            skew_val = float(stats.skew(self.data))
            kurt_val = float(stats.kurtosis(self.data, fisher=False))  # Pearson kurtosis
            
            self._shape_metrics = ShapeMetrics(
                skewness=skew_val,
                kurtosis=kurt_val,
                excess_kurtosis=float(stats.kurtosis(self.data, fisher=True))  # Excess kurtosis
            )
        return self._shape_metrics
    
    def outlier_detection(self) -> OutlierResults:
        """Detect outliers using multiple methods."""
        if self._outlier_results is None:
            # IQR method
            Q1, Q3 = self.data.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            iqr_lower = Q1 - 1.5 * IQR
            iqr_upper = Q3 + 1.5 * IQR
            iqr_mask = (self.data < iqr_lower) | (self.data > iqr_upper)
            iqr_outliers = self.data[iqr_mask]
            iqr_indices = self.data.index[iqr_mask].tolist()
            
            # Z-score method
            z_scores = np.abs(stats.zscore(self.data))
            zscore_mask = z_scores > 3
            zscore_outliers = self.data[zscore_mask]
            zscore_indices = self.data.index[zscore_mask].tolist()
            
            # Combined results
            total_outliers = len(set(iqr_indices + zscore_indices))
            
            self._outlier_results = OutlierResults(
                iqr_outliers=iqr_outliers.tolist(),
                zscore_outliers=zscore_outliers.tolist(),
                outlier_indices={
                    "iqr": iqr_indices,
                    "zscore": zscore_indices
                },
                outlier_percentage=total_outliers / len(self.data) * 100,
                outlier_summary={
                    "iqr_count": len(iqr_indices),
                    "zscore_count": len(zscore_indices),
                    "total_unique": total_outliers
                }
            )
        
        return self._outlier_results
    
    def distribution_tests(self) -> Dict[str, Dict]:
        """Comprehensive normality and distribution tests."""
        results = {}
        
        # Shapiro-Wilk test (works best for n < 5000)
        if "shapiro" in self.config.normality_tests and len(self.data) <= 5000:
            try:
                stat, p_value = stats.shapiro(self.data)
                results['shapiro_wilk'] = {
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "interpretation": "Normal" if p_value > self.config.alpha else "Not Normal"
                }
            except Exception as e:
                logger.warning(f"Shapiro-Wilk test failed: {e}")
        
        # Jarque-Bera test
        if "jarque_bera" in self.config.normality_tests:
            try:
                stat, p_value = stats.jarque_bera(self.data)
                results['jarque_bera'] = {
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "interpretation": "Normal" if p_value > self.config.alpha else "Not Normal"
                }
            except Exception as e:
                logger.warning(f"Jarque-Bera test failed: {e}")
        
        # Anderson-Darling test
        if "anderson" in self.config.normality_tests:
            try:
                result = stats.anderson(self.data, dist='norm')
                # Use 5% significance level (index 2)
                critical_value = result.critical_values[2]
                is_normal = result.statistic < critical_value
                
                results['anderson_darling'] = {
                    "statistic": float(result.statistic),
                    "critical_values": result.critical_values.tolist(),
                    "significance_levels": result.significance_levels.tolist(),
                    "interpretation": "Normal" if is_normal else "Not Normal"
                }
            except Exception as e:
                logger.warning(f"Anderson-Darling test failed: {e}")
        
        # Kolmogorov-Smirnov test against normal distribution
        try:
            mean, std = self.data.mean(), self.data.std()
            stat, p_value = stats.kstest(self.data, lambda x: stats.norm.cdf(x, mean, std))
            results['kolmogorov_smirnov'] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "interpretation": "Normal" if p_value > self.config.alpha else "Not Normal"
            }
        except Exception as e:
            logger.warning(f"Kolmogorov-Smirnov test failed: {e}")
        
        return results
    
    def process_capability_metrics(self, lsl: float = None, usl: float = None, target: float = None) -> Dict:
        """Calculate process capability indices."""
        if lsl is None and usl is None:
            logger.warning("No specification limits provided for capability analysis")
            return {}
        
        mean = self.location_metrics.mean
        std = self.dispersion_metrics.std_dev
        
        metrics = {}
        
        if lsl is not None and usl is not None:
            # Cp - Process Capability
            metrics['Cp'] = (usl - lsl) / (6 * std)
            
            # Cpk - Process Capability Index
            cpu = (usl - mean) / (3 * std)
            cpl = (mean - lsl) / (3 * std)
            metrics['Cpk'] = min(cpu, cpl)
            metrics['CPU'] = cpu
            metrics['CPL'] = cpl
            
            # Pp and Ppk (using sample std dev)
            sample_std = self.data.std(ddof=0)  # Population std dev
            metrics['Pp'] = (usl - lsl) / (6 * sample_std)
            ppu = (usl - mean) / (3 * sample_std)
            ppl = (mean - lsl) / (3 * sample_std)
            metrics['Ppk'] = min(ppu, ppl)
            
        if target is not None:
            # Cpm - Taguchi capability index
            if lsl is not None and usl is not None:
                tau_squared = (mean - target)**2 + std**2
                metrics['Cpm'] = (usl - lsl) / (6 * np.sqrt(tau_squared))
        
        return metrics

class EnhancedFrequencyAnalyzer:
    """Enhanced frequency analysis with statistical insights."""
    
    def __init__(self, data: Union[pd.Series, np.ndarray, List]):
        """Initialize with data for frequency analysis."""
        self.data = pd.Series(data)
    
    def frequency_table(self, bins: Optional[int] = None, is_continuous: bool = False) -> pd.DataFrame:
        """Generate comprehensive frequency table."""
        if is_continuous and bins:
            hist, bin_edges = np.histogram(self.data, bins=bins)
            df = pd.DataFrame({
                'bin_start': bin_edges[:-1].round(4),
                'bin_end': bin_edges[1:].round(4),
                'bin_midpoint': ((bin_edges[:-1] + bin_edges[1:]) / 2).round(4),
                'absolute_freq': hist
            })
            df['relative_freq'] = (df['absolute_freq'] / df['absolute_freq'].sum()).round(6)
            df['percentage'] = (df['relative_freq'] * 100).round(2)
            df['cumulative_freq'] = df['absolute_freq'].cumsum()
            df['cumulative_percentage'] = (df['cumulative_freq'] / df['absolute_freq'].sum() * 100).round(2)
        else:
            freq = self.data.value_counts().reset_index()
            freq.columns = ['value', 'absolute_freq']
            freq['relative_freq'] = (freq['absolute_freq'] / freq['absolute_freq'].sum()).round(6)
            freq['percentage'] = (freq['relative_freq'] * 100).round(2)
            freq = freq.sort_values('value').reset_index(drop=True)
            freq['cumulative_freq'] = freq['absolute_freq'].cumsum()
            freq['cumulative_percentage'] = (freq['cumulative_freq'] / freq['absolute_freq'].sum() * 100).round(2)
            df = freq
        
        return df

class EnhancedVisualizer:
    """Enhanced visualization with quality control focus."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    @contextmanager
    def plot_context(self, figsize=None, save_path=None):
        """Context manager for consistent plot handling."""
        figsize = figsize or self.config.figure_size
        fig, ax = plt.subplots(figsize=figsize)
        try:
            yield fig, ax
        finally:
            if save_path:
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                logger.info(f"Plot saved to {save_path}")
            plt.tight_layout()
            plt.show()
    
    def enhanced_histogram(self, data: pd.Series, bins: int = 30, title: str = "Enhanced Histogram", 
                          save_path: Optional[str] = None, show_stats: bool = True,
                          highlight_outliers: bool = False, outlier_results: OutlierResults = None):
        """Create enhanced histogram with statistical overlays."""
        
        with self.plot_context(save_path=save_path) as (fig, ax):
            # Create histogram with KDE
            n, bins_array, patches = ax.hist(data, bins=bins, density=True, alpha=0.7, 
                                           color='skyblue', edgecolor='black', linewidth=0.5)
            
            # Add KDE
            try:
                kde_x = np.linspace(data.min(), data.max(), 200)
                kde = stats.gaussian_kde(data)
                ax.plot(kde_x, kde(kde_x), 'navy', linewidth=2, label='KDE')
            except:
                logger.warning("Could not compute KDE")
            
            # Add normal distribution overlay
            mean, std = data.mean(), data.std()
            norm_x = np.linspace(data.min(), data.max(), 200)
            norm_y = stats.norm.pdf(norm_x, mean, std)
            ax.plot(norm_x, norm_y, 'red', linewidth=2, linestyle='--', label='Normal Distribution')
            
            # Highlight outliers if requested
            if highlight_outliers and outlier_results:
                for outlier in outlier_results.iqr_outliers:
                    ax.axvline(outlier, color='red', linestyle=':', alpha=0.7)
            
            # Add statistics text box
            if show_stats:
                stats_text = f'Mean: {mean:.3f}\nStd: {std:.3f}\nSkewness: {stats.skew(data):.3f}\nKurtosis: {stats.kurtosis(data):.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel("Value", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def control_chart(self, data: pd.Series, title: str = "Control Chart", 
                     save_path: Optional[str] = None, ucl: float = None, lcl: float = None):
        """Create a basic control chart."""
        
        with self.plot_context(save_path=save_path) as (fig, ax):
            # Calculate control limits if not provided
            mean = data.mean()
            std = data.std()
            
            if ucl is None:
                ucl = mean + 3 * std
            if lcl is None:
                lcl = mean - 3 * std
            
            # Plot data points
            ax.plot(range(len(data)), data, 'bo-', markersize=4, linewidth=1, label='Data')
            
            # Add control lines
            ax.axhline(mean, color='green', linewidth=2, label=f'Mean ({mean:.3f})')
            ax.axhline(ucl, color='red', linewidth=2, linestyle='--', label=f'UCL ({ucl:.3f})')
            ax.axhline(lcl, color='red', linewidth=2, linestyle='--', label=f'LCL ({lcl:.3f})')
            
            # Highlight out-of-control points
            out_of_control = (data > ucl) | (data < lcl)
            if out_of_control.any():
                ooc_indices = data.index[out_of_control]
                ax.scatter(ooc_indices, data[out_of_control], color='red', s=50, 
                          marker='x', linewidth=3, label='Out of Control')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel("Sample Number", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def qq_plot(self, data: pd.Series, title: str = "Q-Q Plot", save_path: Optional[str] = None):
        """Create Q-Q plot for normality assessment."""
        
        with self.plot_context(save_path=save_path) as (fig, ax):
            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

class EnhancedStatisticalAnalysis:
    """Main enhanced statistical analysis class."""
    
    def __init__(self, data: Union[pd.DataFrame, pd.Series] = None, 
                 file_path: Optional[str] = None, config: AnalysisConfig = None):
        """Initialize with data or load from file."""
        
        self.config = config or AnalysisConfig()
        
        if file_path:
            self.data = self._load_data(file_path)
        elif data is not None:
            self.data = data
            logger.info(f"Using provided data with {len(self.data)} rows")
        else:
            raise ValueError("Must provide either data or file_path.")
        
        self.visualizer = EnhancedVisualizer(self.config)
        
        # Create output directory
        if self.config.save_plots:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Print data info
        if isinstance(self.data, pd.DataFrame):
            logger.info(f"Available columns: {list(self.data.columns)}")
    
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file with robust error handling."""
        file_path = Path(file_path)
        logger.info(f"Loading data from: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            engines = ['openpyxl', 'xlrd']
            for engine in engines:
                try:
                    data = pd.read_excel(file_path, engine=engine)
                    logger.info(f"Successfully loaded Excel file using {engine}")
                    return data
                except Exception as e:
                    logger.warning(f"Failed to load with {engine}: {e}")
            
            # Final attempt as CSV
            try:
                logger.info("Attempting to read as CSV...")
                return pd.read_csv(file_path)
            except Exception as e:
                raise ValueError(f"Could not read file {file_path}: {e}")
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def comprehensive_analysis(self, column_name: str = None, 
                             specification_limits: Dict[str, float] = None) -> Dict:
        """Perform comprehensive statistical analysis on a column."""
        
        # Get the data series
        if isinstance(self.data, pd.DataFrame):
            if not column_name:
                raise ValueError("Column name must be provided for DataFrame input.")
            if column_name not in self.data.columns:
                available = list(self.data.columns)
                raise ValueError(f"Column '{column_name}' not found. Available: {available}")
            series = self.data[column_name].copy()
        else:
            series = self.data.copy()
            column_name = column_name or "data"
        
        logger.info(f"Analyzing column: {column_name}")
        
        # Initialize analyzers
        analyzer = EnhancedStatisticalAnalyzer(series, self.config)
        freq_analyzer = EnhancedFrequencyAnalyzer(series)
        
        # Perform analysis
        results = {
            "column_name": column_name,
            "sample_size": len(series),
            "location_metrics": analyzer.location_metrics,
            "dispersion_metrics": analyzer.dispersion_metrics,
            "shape_metrics": analyzer.shape_metrics,
            "outlier_analysis": analyzer.outlier_detection(),
            "normality_tests": analyzer.distribution_tests(),
            "frequency_table": freq_analyzer.frequency_table(
                bins=self.config.bins if len(series) > 50 else None, 
                is_continuous=len(series) > 50
            )
        }
        
        # Process capability if limits provided
        if specification_limits:
            results["process_capability"] = analyzer.process_capability_metrics(**specification_limits)
        
        # Generate visualizations
        plot_name = column_name.replace(" ", "_").lower()
        
        # Enhanced histogram
        hist_path = None
        if self.config.save_plots:
            hist_path = Path(self.config.output_dir) / f"{plot_name}_enhanced_histogram.png"
        
        self.visualizer.enhanced_histogram(
            series, 
            bins=self.config.bins,
            title=f"Enhanced Histogram of {column_name}",
            save_path=hist_path,
            show_stats=True,
            highlight_outliers=True,
            outlier_results=results["outlier_analysis"]
        )
        
        # Control chart
        cc_path = None
        if self.config.save_plots:
            cc_path = Path(self.config.output_dir) / f"{plot_name}_control_chart.png"
        
        self.visualizer.control_chart(
            series,
            title=f"Control Chart of {column_name}",
            save_path=cc_path
        )
        
        # Q-Q plot
        qq_path = None
        if self.config.save_plots:
            qq_path = Path(self.config.output_dir) / f"{plot_name}_qq_plot.png"
        
        self.visualizer.qq_plot(
            series,
            title=f"Q-Q Plot of {column_name}",
            save_path=qq_path
        )
        
        return results
    
    def batch_analysis(self, columns: List[str] = None, 
                      specification_limits: Dict[str, Dict[str, float]] = None) -> Dict[str, Dict]:
        """Perform analysis on multiple columns."""
        
        if isinstance(self.data, pd.Series):
            return {"data": self.comprehensive_analysis()}
        
        if columns is None:
            # Auto-select numeric columns
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            # Filter out tolerance flag columns
            columns = [col for col in numeric_columns 
                      if not col.endswith('_InTolerance') and col != 'All_InTolerance']
        
        results = {}
        
        for column in columns:
            try:
                spec_limits = specification_limits.get(column, {}) if specification_limits else {}
                results[column] = self.comprehensive_analysis(column, spec_limits)
                logger.info(f"Completed analysis for {column}")
            except Exception as e:
                logger.error(f"Failed to analyze {column}: {e}")
                results[column] = {"error": str(e)}
        
        return results
    
    def generate_report(self, results: Dict, output_path: str = None):
        """Generate a comprehensive analysis report."""
        
        if output_path is None:
            output_path = Path(self.config.output_dir) / "analysis_report.html"
        
        # This would generate an HTML report - simplified version here
        logger.info(f"Analysis complete. Report would be generated at: {output_path}")
        
        # For now, save key metrics to CSV
        summary_data = []
        for column, result in results.items():
            if "error" in result:
                continue
                
            # Debug information
            logger.info(f"Processing results for column: {column}")
            logger.info(f"Result keys: {list(result.keys())}")
            
            # Check if we have the expected data structure
            if not all(key in result for key in ["location_metrics", "dispersion_metrics", "shape_metrics", "outlier_analysis"]):
                logger.warning(f"Missing expected metrics for column {column}")
                continue
                
            # Access metrics as objects with attributes
            loc_metrics = result["location_metrics"]
            disp_metrics = result["dispersion_metrics"]
            shape_metrics = result["shape_metrics"]
            outlier_analysis = result["outlier_analysis"]
            
            # Create a dictionary with the metrics
            summary_data.append({
                "Column": column,
                "Sample_Size": result["sample_size"],
                "Mean": loc_metrics.mean if hasattr(loc_metrics, 'mean') else loc_metrics['mean'],
                "Median": loc_metrics.median if hasattr(loc_metrics, 'median') else loc_metrics['median'],
                "Std_Dev": disp_metrics.std_dev if hasattr(disp_metrics, 'std_dev') else disp_metrics['std_dev'],
                "CV_Percent": disp_metrics.coefficient_variation if hasattr(disp_metrics, 'coefficient_variation') else disp_metrics['coefficient_variation'],
                "Skewness": shape_metrics.skewness if hasattr(shape_metrics, 'skewness') else shape_metrics['skewness'],
                "Kurtosis": shape_metrics.kurtosis if hasattr(shape_metrics, 'kurtosis') else shape_metrics['kurtosis'],
                "Outlier_Percentage": outlier_analysis.outlier_percentage if hasattr(outlier_analysis, 'outlier_percentage') else outlier_analysis['outlier_percentage']
            })
        
        # Check if we have any data to save
        if not summary_data:
            logger.error("No data to save to CSV - summary_data is empty!")
            return
            
        logger.info(f"Creating DataFrame with {len(summary_data)} rows")
        summary_df = pd.DataFrame(summary_data)
        csv_path = Path(self.config.output_dir) / "analysis_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Summary saved to: {csv_path} with {len(summary_df)} rows")

# Example usage and demonstration
def main():
    """Main function to run the enhanced statistical analysis."""
    print("Starting Enhanced Statistical Analysis...")
    
    # Configure analysis
    config = AnalysisConfig(
        bins=30,
        save_plots=True,
        output_dir="enhanced_analysis_output",
        normality_tests=["shapiro", "jarque_bera", "anderson"]
    )
    
    try:
        # Try to load existing results.csv file
        data_file = r"C:\Users\lamarw\Desktop\ComputerScience\DataScienceProjects\results.csv"
        
        print(f"Attempting to load data from: {data_file}")
        
        if not os.path.exists(data_file):
            print(f"Warning: {data_file} not found!")
            print("Creating sample data for demonstration...")
            
            # Create sample manufacturing data as fallback
            np.random.seed(42)  # For reproducibility
            
            sample_data = pd.DataFrame({
                "Bore_Diameter": np.random.normal(25.0, 0.5, 100),
                "LTB": np.random.normal(30.0, 1.0, 100),
                "Nominal_Diameter": np.random.normal(342.5, 2.0, 100),
                "Pitch": np.random.normal(0.625, 0.01, 100),
                "Roller_Diameter": np.random.normal(400, 5, 100),
                "Roller_Width": np.random.normal(0.375, 0.005, 100)
            })
            
            # Add some outliers to make it realistic
            sample_data.loc[95:97, "Bore_Diameter"] = [26.5, 23.2, 27.1]
            sample_data.loc[92:94, "Pitch"] = [0.645, 0.605, 0.650]
            
            # Save as results.csv for future use
            sample_data.to_csv(data_file, index=False)
            print(f"Sample data saved to '{data_file}' for future analysis")
            
            # Initialize analysis with sample data
            analysis = EnhancedStatisticalAnalysis(data=sample_data, config=config)
        else:
            # Load actual results.csv file
            print(f"Loading data from {data_file}...")
            analysis = EnhancedStatisticalAnalysis(file_path=data_file, config=config)
            print(f"Successfully loaded data from {data_file}")
            
            # Display basic info about the loaded data
            print(f"Data shape: {analysis.data.shape}")
            print("Data columns:", list(analysis.data.columns))
            print("\nFirst few rows:")
            print(analysis.data.head())
            
            # Show data types
            print("\nData types:")
            print(analysis.data.dtypes)
        
        # Define specification limits for process capability
        spec_limits = {
            "Bore_Diameter": {"lsl": 24.0, "usl": 26.0, "target": 25.0},
            "LTB": {"lsl": 28.0, "usl": 32.0, "target": 30.0},
            "Nominal_Diameter": {"lsl": 338.0, "usl": 347.0, "target": 342.5},
            "Pitch": {"lsl": 0.615, "usl": 0.635, "target": 0.625},
            "Roller_Diameter": {"lsl": 390, "usl": 410, "target": 400},
            "Roller_Width": {"lsl": 0.365, "usl": 0.385, "target": 0.375}
        }
        
        print("Specification limits defined for process capability analysis")
        
        # Perform batch analysis
        print("\nStarting comprehensive batch analysis...")
        all_results = analysis.batch_analysis(specification_limits=spec_limits)
        
        # Generate report
        print("Generating analysis report...")
        analysis.generate_report(all_results)
        
        # Print summary
                
        print("\n" + "="*80)
        print("ENHANCED STATISTICAL ANALYSIS SUMMARY")
        print("="*80)
        
        for column, result in all_results.items():
            if "error" in result:
                print(f"\n{column}: Analysis failed - {result['error']}")
                continue
                
            print(f"\n{column.upper()}:")
            print("-" * 50)
            
            # Location metrics
            loc = result["location_metrics"]
            print(f"  Mean: {loc.mean:.4f}")
            print(f"  Median: {loc.median:.4f}")
            if loc.geometric_mean:
                print(f"  Geometric Mean: {loc.geometric_mean:.4f}")
            
            # Dispersion metrics  
            disp = result["dispersion_metrics"]
            print(f"  Std Dev: {disp.std_dev:.4f}")
            print(f"  CV%: {disp.coefficient_variation:.2f}%")
            print(f"  Range: {disp.range:.4f}")
            
            # Shape metrics
            shape = result["shape_metrics"]
            print(f"  Skewness: {shape.skewness:.4f}")
            print(f"  Kurtosis: {shape.kurtosis:.4f}")
            
            # Outlier analysis
            outliers = result["outlier_analysis"]
            print(f"  Outliers: {outliers.outlier_percentage:.2f}% of data")
            
            # Normality tests
            normality = result["normality_tests"]
            normal_count = sum(1 for test in normality.values() 
                             if test.get("interpretation") == "Normal")
            total_tests = len(normality)
            print(f"  Normality: {normal_count}/{total_tests} tests suggest normal distribution")
            
            # Process capability (if available)
            if "process_capability" in result and result["process_capability"]:
                cap = result["process_capability"]
                if "Cp" in cap:
                    print(f"  Process Capability (Cp): {cap['Cp']:.3f}")
                if "Cpk" in cap:
                    print(f"  Process Capability (Cpk): {cap['Cpk']:.3f}")
                    if cap['Cpk'] >= 1.33:
                        print(f"    → Excellent process capability")
                    elif cap['Cpk'] >= 1.0:
                        print(f"    → Adequate process capability")
                    else:
                        print(f"    → Poor process capability")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS FOR QUALITY TEAM:")
        print("="*80)
        
        for column, result in all_results.items():
            if "error" in result:
                continue
                
            recommendations = []
            
            # Check normality
            normality = result["normality_tests"]
            normal_count = sum(1 for test in normality.values() 
                             if test.get("interpretation") == "Normal")
            total_tests = len(normality)
            
            if normal_count < total_tests / 2:
                recommendations.append("Consider non-parametric control charts")
            else:
                recommendations.append("Suitable for standard SPC methods")
            
            # Check outliers
            outlier_pct = result["outlier_analysis"].outlier_percentage
            if outlier_pct > 5:
                recommendations.append("Investigate outlier sources")
            
            # Check process capability
            if "process_capability" in result and "Cpk" in result["process_capability"]:
                cpk = result["process_capability"]["Cpk"]
                if cpk < 1.0:
                    recommendations.append("Process improvement needed")
                elif cpk < 1.33:
                    recommendations.append("Monitor closely - borderline capability")
            
            # Check variation
            cv = result["dispersion_metrics"].coefficient_variation
            if cv > 10:
                recommendations.append("High variation - investigate causes")
            
            if recommendations:
                print(f"\n{column}:")
                for rec in recommendations:
                    print(f"  • {rec}")
        
        print("\n" + "="*80)
        print("FILES GENERATED:")
        print("="*80)
        print(f"  • Analysis summary: {config.output_dir}/analysis_summary.csv")
        print(f"  • Enhanced histograms: {config.output_dir}/*_enhanced_histogram.png")
        print(f"  • Control charts: {config.output_dir}/*_control_chart.png") 
        print(f"  • Q-Q plots: {config.output_dir}/*_qq_plot.png")
        print(f"  • Sample data: sample_sprocket_data.csv")
        print(f"  • Full logs available in console output")
        
        print("\nAnalysis completed successfully!")
        print("Check the 'enhanced_analysis_output' folder for all generated files and plots.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Show any remaining plots
        try:
            plt.show()
        except:
            pass

if __name__ == "__main__":
    main()