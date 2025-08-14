import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from scipy.stats import shapiro
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Multivariate Linear Regression Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border: 2px solid #1f77b4;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem 0;
        border-bottom: 2px solid #3498db;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #fdcb6e;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def load_sample_data(dataset='tips'):
    """Load sample datasets"""
    if dataset == 'tips':
        return sns.load_dataset('tips')
    elif dataset == 'mpg':
        return sns.load_dataset('mpg').dropna()
    else:
        return sns.load_dataset('tips')

def detect_categorical_columns(df):
    """Detect categorical columns automatically"""
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_cols.append(col)
    return categorical_cols

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def perform_shapiro_test(residuals):
    """Perform Shapiro-Wilk test for normality"""
    stat, p_value = shapiro(residuals)
    return stat, p_value

def create_residual_plots(residuals, fitted_values):
    """Create residual analysis plots"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Residuals Histogram', 'Q-Q Plot', 'Residuals vs Fitted', 'Residuals Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Histogram of residuals
    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=20, name='Residuals', showlegend=False,
                    marker_color='skyblue', opacity=0.7),
        row=1, col=1
    )
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=None)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    sample_quantiles = np.sort(residuals)
    
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers',
                  name='Q-Q Plot', showlegend=False, marker=dict(color='red', size=8)),
        row=1, col=2
    )
    
    # Add diagonal line for Q-Q plot
    min_val, max_val = min(theoretical_quantiles.min(), sample_quantiles.min()), max(theoretical_quantiles.max(), sample_quantiles.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                  name='45° Line', line=dict(color='blue', dash='dash', width=2), showlegend=False),
        row=1, col=2
    )
    
    # Residuals vs Fitted
    fig.add_trace(
        go.Scatter(x=fitted_values, y=residuals, mode='markers',
                  name='Residuals vs Fitted', showlegend=False, 
                  marker=dict(color='green', size=8)),
        row=2, col=1
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2, row=2, col=1)
    
    # Box plot of residuals
    fig.add_trace(
        go.Box(x=residuals, name='', showlegend=False,
               marker_color='orange'),
        row=2, col=2
    )
    
    # Update layout with black font and larger sizes
    fig.update_layout(
        height=600,
        title_text="Residual Analysis",
        title_font=dict(color='black', size=24),
        showlegend=False,
        font=dict(color='black', size=14)
    )
    
    # Update subplot titles with larger font
    fig.update_annotations(font=dict(color='black', size=16))
    
    # Update axis labels and ticks with black font and larger size
    fig.update_xaxes(
        title_font=dict(color='black', size=18),
        tickfont=dict(color='black', size=14)
    )
    fig.update_yaxes(
        title_font=dict(color='black', size=18),
        tickfont=dict(color='black', size=14)
    )
    
    # Update specific axis titles
    fig.update_xaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_xaxes(title_text="Predicted Values", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)
    
    return fig

# Main app
def main():
    st.markdown('<div class="main-header">📊 Multivariate Linear Regression Explorer</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = load_sample_data('tips')
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Configuration Panel</div>', unsafe_allow_html=True)
        
        # Data loading options
        st.subheader("📁 Data Input")
        data_option = st.radio("Choose data source:", 
                              ["Sample Data (Tips)", "Sample Data (MPG)", "Upload CSV File"])
        
        if data_option == "Upload CSV File":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    st.session_state.df = pd.read_csv(uploaded_file)
                    st.success("File uploaded successfully!")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        elif data_option == "Sample Data (Tips)":
            if st.button("🔄 Load Tips Dataset"):
                st.session_state.df = load_sample_data('tips')
                st.success("Tips dataset loaded!")
        elif data_option == "Sample Data (MPG)":
            if st.button("🔄 Load MPG Dataset"):
                st.session_state.df = load_sample_data('mpg')
                st.success("MPG dataset loaded!")
        
    
    # Main content area
    st.markdown('<div class="section-header">📋 Data Preview</div>', unsafe_allow_html=True)
    
    # Display editable dataframe
    
    # st.write("**Editable Data Table** (Double-click cells to edit)")
    edited_df = st.data_editor(
        st.session_state.df,
        use_container_width=True,
        num_rows="dynamic",
        key="data_editor"
    )
    st.session_state.df = edited_df
    
    
    # Configuration panel continued in sidebar
    if len(st.session_state.df) > 0:
        with st.sidebar:
            st.markdown("---")
            
            # Get numeric columns for dependent variable
            numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                # Dependent variable selection
                st.subheader("🎯 Dependent Variable")
                dependent_var = st.radio(
                    "Select dependent variable (Y):",
                    numeric_cols,
                    key="dependent_var"
                )
                
                # Independent variables selection
                st.subheader("📊 Independent Variables")
                available_cols = [col for col in st.session_state.df.columns if col != dependent_var]
                independent_vars = st.multiselect(
                    "Select independent variables (X):",
                    available_cols,
                    default=[col for col in available_cols[:3] if col in available_cols],
                    key="independent_vars"
                )
                
                if independent_vars:
                    # Categorical variables identification
                    st.subheader("🏷️ Categorical Variables")
                    st.write("Select which variables should be treated as categorical:")
                    
                    # Auto-detect categorical columns
                    auto_categorical = detect_categorical_columns(st.session_state.df[independent_vars])
                    
                    categorical_vars = []
                    for var in independent_vars:
                        is_categorical = st.checkbox(
                            f"{var} (auto-detected: {'Yes' if var in auto_categorical else 'No'})",
                            value=var in auto_categorical,
                            key=f"cat_{var}"
                        )
                        if is_categorical:
                            categorical_vars.append(var)
                    
                    st.markdown("---")
                    # Sample size selection
                    if len(st.session_state.df) > 0:
                        st.subheader("📊 Sample Size")
                        max_rows = len(st.session_state.df)
                        n_rows = st.slider(
                            "Number of rows to include in regression:",
                            min_value=10,
                            max_value=max_rows,
                            value=min(max_rows, 100),
                            step=1,
                            help=f"Total available rows: {max_rows}"
                        )
                        st.session_state.n_rows = n_rows
                    
                    # Analysis trigger
                    if st.button("🚀 Calculate Linear Regression Parameters", type="primary", use_container_width=True):
                        
                        # Prepare data for regression
                        try:
                            # Use only selected number of rows
                            df_sample = st.session_state.df.head(st.session_state.n_rows)
                            st.info(f"Using {len(df_sample)} rows for regression analysis")
                            
                            # Create feature matrix
                            X = df_sample[independent_vars].copy()
                            y = df_sample[dependent_var].copy()
                            
                            # Remove rows with missing values
                            mask = ~(X.isnull().any(axis=1) | y.isnull())
                            X = X[mask]
                            y = y[mask]
                            
                            if len(X) == 0:
                                st.error("No valid data remaining after removing missing values.")
                                return
                            
                            st.success(f"Final dataset: {len(X)} rows with complete data")
                            
                            # Store original column names for interpretation
                            original_columns = independent_vars.copy()
                            
                            # Apply one-hot encoding to categorical variables using pandas get_dummies
                            if categorical_vars:
                                # Separate categorical and numerical variables
                                cat_cols = [col for col in independent_vars if col in categorical_vars]
                                num_cols = [col for col in independent_vars if col not in categorical_vars]
                                
                                # Create dataframe for processing
                                X_processed_df = pd.DataFrame(index=X.index)
                                
                                # Add numerical columns as-is
                                if num_cols:
                                    X_processed_df = pd.concat([X_processed_df, X[num_cols]], axis=1)
                                
                                # One-hot encode categorical variables with get_dummies
                                for cat_col in cat_cols:
                                    cat_dummies = pd.get_dummies(X[cat_col], prefix=cat_col, drop_first=True)
                                    X_processed_df = pd.concat([X_processed_df, cat_dummies], axis=1)
                                
                                # Convert to numpy array and get feature names
                                X_processed = X_processed_df.values.astype(float)
                                feature_names = X_processed_df.columns.tolist()
                                
                            else:
                                X_processed = X.values.astype(float)
                                feature_names = independent_vars
                            
                            # Fit the regression model
                            model = LinearRegression()
                            model.fit(X_processed, y)
                            
                            # Make predictions
                            y_pred = model.predict(X_processed)
                            residuals = y - y_pred
                            
                            # Store results in session state
                            st.session_state.model = model
                            st.session_state.y_true = y
                            st.session_state.y_pred = y_pred
                            st.session_state.residuals = residuals
                            st.session_state.X_processed = X_processed
                            st.session_state.feature_names = feature_names
                            st.session_state.dependent_var_name = dependent_var
                            st.session_state.analysis_complete = True
                            
                        except Exception as e:
                            st.error(f"Error in regression analysis: {str(e)}")
                            return
            else:
                st.warning("No numeric columns available for regression analysis.")
    
    # Display results if analysis is complete
    if hasattr(st.session_state, 'analysis_complete') and st.session_state.analysis_complete:
        
        st.markdown('<div class="section-header">📈 Results</div>', unsafe_allow_html=True)
        y_true = st.session_state.y_true
        y_pred = st.session_state.y_pred
        residuals = st.session_state.residuals
        model = st.session_state.model

        # Display equation
        # Escape underscores in variable names for LaTeX
        dep_var_escaped = st.session_state.dependent_var_name.replace("_", r"\_")
        
        # Build the LaTeX equation
        equation = rf"\text{{{dep_var_escaped}}}_\text{{model}} = {model.intercept_:.4f}"
        
        for i, (name, coef) in enumerate(zip(st.session_state.feature_names, model.coef_)):
            # Escape underscores in feature names
            name_escaped = name.replace("_", r"\_")
            sign = "+" if coef >= 0 else "-"
            equation += rf" {sign} {abs(coef):.4f}\cdot \text{{{name_escaped}}}"
        
        # Display as LaTeX
        st.latex(equation)     

        # Model Performance Metrics
        st.markdown("### 🎯 Model Performance Metrics")
                
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = calculate_mape(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Display metrics in columns
        metric_cols = st.columns(5)
        
        with metric_cols[0]:
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="margin:0; color:#2c3e50;">MSE</h4>
                <h2 style="margin:0; color:#e74c3c;">{mse:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[1]:
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="margin:0; color:#2c3e50;">RMSE</h4>
                <h2 style="margin:0; color:#e74c3c;">{rmse:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[2]:
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="margin:0; color:#2c3e50;">MAE</h4>
                <h2 style="margin:0; color:#e74c3c;">{mae:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[3]:
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="margin:0; color:#2c3e50;">MAPE (%)</h4>
                <h2 style="margin:0; color:#e74c3c;">{mape:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_cols[4]:
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="margin:0; color:#2c3e50;">R²</h4>
                <h2 style="margin:0; color:#e74c3c;">{r2:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
                
        # Create and display residual plots
        residual_fig = create_residual_plots(residuals, y_pred)
        st.plotly_chart(residual_fig, use_container_width=True)


        st.subheader("Normality Assessment")        
        # Residual statistics table
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Residual Statistics:**")
            residual_stats = pd.DataFrame({
                'Statistic': ['Mean', 'Standard Deviation', 'Skewness', 'Kurtosis'],
                'Value': [
                    np.mean(residuals),
                    np.std(residuals),
                    stats.skew(residuals),
                    stats.kurtosis(residuals, fisher=False)
                ]
            })
            st.dataframe(residual_stats, hide_index=True)
        
        with col2:
            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = perform_shapiro_test(residuals)
            
            st.markdown("**Shapiro-Wilk Test:**")
            st.markdown("**H0:** Residuals are normally distributed")
            st.markdown("**HA:** Residuals are not normally distributed")
            st.write(f"**p-value:** {shapiro_p:.6f}")
            
            if shapiro_p > 0.05:
                st.markdown(f"""
                <div class="success-box">
                    <strong>Conclusion:</strong> Fail to reject H₀<br>
                    Residuals appear to be normally distributed (p > 0.05)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>Conclusion:</strong> Reject H₀<br>
                    Residuals may not be normally distributed (p ≤ 0.05)
                </div>
                """, unsafe_allow_html=True)
        
        # Statistical Significance Tests
        st.subheader("📈 Statistical Significance Tests")
        
        # Calculate statistical tests
        n = len(y_true)
        k = len(st.session_state.feature_names)
        
        # F-test for overall model significance
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        ss_reg = ss_tot - ss_res
        
        f_statistic = (ss_reg / k) / (ss_res / (n - k - 1))
        f_p_value = 1 - stats.f.cdf(f_statistic, k, n - k - 1)
        
        # t-tests for coefficients
        # Initialize variables first
        all_coefficients = np.concatenate([[model.intercept_], model.coef_])
        X_with_intercept = np.column_stack([np.ones(len(st.session_state.X_processed)), st.session_state.X_processed])
        
        try:
            # Ensure all data is float type for matrix operations
            X_with_intercept = X_with_intercept.astype(float)
            y_array = np.array(y_true, dtype=float)
            
            # Calculate covariance matrix
            mse_model = ss_res / (n - k - 1)
            XTX = X_with_intercept.T @ X_with_intercept
            
            # Check if matrix is invertible
            if np.linalg.det(XTX) != 0:
                cov_matrix = mse_model * np.linalg.inv(XTX)
                std_errors = np.sqrt(np.diag(cov_matrix))
                
                t_statistics = all_coefficients / std_errors
                t_p_values = 2 * (1 - stats.t.cdf(np.abs(t_statistics), n - k - 1))
            else:
                raise ValueError("Matrix is singular (not invertible)")
            
        except Exception as e:
            # Fallback if matrix inversion fails
            st.warning(f"Could not calculate standard errors: {str(e)}")
            std_errors = np.full(k + 1, np.nan)
            t_statistics = np.full(k + 1, np.nan)
            t_p_values = np.full(k + 1, np.nan)
        
        # Display F-test results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**F-Test (Overall Coefficient Significance):**")
            st.markdown("**H0:** All slope equals zero (β1 = β2 = ... = βn= 0)")
            st.markdown("**HA:** At least one slope is nonzero")

            st.write(f"**p-value:** {f_p_value:.6f}")
            
            if f_p_value < 0.05:
                st.markdown(f"""
                <div class="success-box">
                    <strong>Conclusion:</strong> The model is statistically significant (p < 0.05)<br>
                    At least one predictor variable is significantly related to the outcome.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>Conclusion:</strong> The model is not statistically significant (p ≥ 0.05)<br>
                    No predictor variables are significantly related to the outcome.
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**T-Test (Individual Coefficient Significance Tests):**")
            
            # Create coefficient table
            coef_names = ['Intercept'] + st.session_state.feature_names
            n = len(y_true)
            k = len(st.session_state.feature_names)

            # Calculate 95% confidence intervals
            alpha = 0.05  # for 95% CI
            t_critical = stats.t.ppf(1 - alpha/2, n - k - 1)
            
            # Calculate confidence intervals
            ci_lower = all_coefficients - t_critical * std_errors
            ci_upper = all_coefficients + t_critical * std_errors

            # Format confidence intervals as strings
            confidence_intervals = [f"[{lower:.3f}, {upper:.3f}]" 
                                for lower, upper in zip(ci_lower, ci_upper)]

            coef_df = pd.DataFrame({
                'Variable': coef_names,
                'β Coefficient': all_coefficients,
                'β Conf. Interval': confidence_intervals,
                'p-value': t_p_values,
                'Significant': ['Yes' if p < 0.05 else 'No' for p in t_p_values]
            })
            
            st.dataframe(
                coef_df.round(6),
                hide_index=True,
                use_container_width=True
            )
        
        # Model equation
        st.subheader("📝 Model Equation")
        
        # Escape underscores in variable names for LaTeX
        dep_var_escaped = st.session_state.dependent_var_name.replace("_", r"\_")
        
        # Build the LaTeX equation
        equation = rf"\text{{{dep_var_escaped}}}_\text{{model}} = {model.intercept_:.4f}"
        
        for i, (name, coef) in enumerate(zip(st.session_state.feature_names, model.coef_)):
            # Escape underscores in feature names
            name_escaped = name.replace("_", r"\_")
            sign = "+" if coef >= 0 else "-"
            equation += rf" {sign} {abs(coef):.4f} \  \text{{{name_escaped}}}"
        
        # Display as LaTeX
        st.latex(equation)        
        
        with st.expander("📚 Statistical Interference Concepts", expanded=False):
            st.subheader("Statistical Inference for Linear Regression")
            st.markdown("- Required Assumption: **Errors are normally distributed**")
            st.markdown("- For **multivariate linear regression**:")
            st.latex(r"y_{\text{model}} = \beta_0 + \beta_1 x + ... + \beta_n x")


            st.markdown("***F-test for Multivariate Linear Regression:***")
            st.markdown("- $H_0: \\text{ All slope equals zero} \\left(\\beta_1 = \\beta_2 = ... = \\beta_n = 0\\right)$")
            st.markdown("- $H_A: \\text{At least one slope is nonzero} $")
            st.markdown("**Note: For univariate linear regression, F-test is just t-test with only one slope $\\beta_1$**")


            st.markdown("***t-test for Slope ($\\beta_i$):***")
            st.markdown("- $H_0: \\beta_i = 0 \\text{ (no linear relationship)}$")
            st.markdown("- $H_A: \\beta_i \\neq 0 \\text{ (linear relationship exists)}$")
            st.markdown("- $\\text{Where } i > 0$")

            st.markdown("***t-test for Intercept ($\\beta_0$):***")
            st.markdown("- $H_0: \\beta_0 = 0 \\text{ (intercept equals zero)}$")
            st.markdown("- $H_A: \\beta_0 \\neq 0 \\text{ (intercept does not equal zero)}$")






if __name__ == "__main__":
    main()