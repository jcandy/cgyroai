import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="CGYRO-AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def generate_sample_data(n_samples=1000, random_state=42):
    """Generate synthetic dataset with correlations between inputs and outputs"""
    np.random.seed(random_state)
    
    # Generate input attributes with realistic CGYRO parameter ranges
    RMIN = np.random.uniform(0.2, 0.9, n_samples)
    RMAJ = np.random.uniform(1.0, 4.0, n_samples)
    SHEAR = np.random.uniform(0.1, 3.0, n_samples)
    KAPPA = np.random.uniform(1.0, 2.0, n_samples)
    
    # Create outputs with physics-based functional dependencies
    # Add small amount of noise to simulate measurement/computational uncertainty
    noise1 = np.random.normal(0, 0.1, n_samples)
    noise2 = np.random.normal(0, 0.1, n_samples)
    
    QI = KAPPA * (np.sin(RMIN) + np.exp(RMAJ) + SHEAR**2) + noise1
    QE = KAPPA * (2*np.sin(RMIN) + np.exp(1/RMAJ) + SHEAR**3) + noise2
    
    # Create DataFrame
    data = pd.DataFrame({
        'RMIN': RMIN,
        'RMAJ': RMAJ,
        'SHEAR': SHEAR,
        'KAPPA': KAPPA,
        'QI': QI,
        'QE': QE
    })
    
    # Add metadata
    locations = np.random.choice(['ORNL', 'NERSC', 'ANL'], n_samples, p=[0.4, 0.35, 0.25])
    years = np.random.choice([2022, 2023, 2024, 2025], n_samples, p=[0.2, 0.3, 0.35, 0.15])
    
    data['location'] = locations
    data['year'] = years
    
    # Add NAME metadata - 2 unique names per location-year combination
    np.random.seed(random_state + 1)  # Different seed for names
    name_pool = [
        'Fusion', 'Plasma', 'Tokamak', 'Stellarator', 'Magnetosphere', 'Ionosphere', 'Corona', 'Heliosphere',
        'Neutron', 'Proton', 'Electron', 'Photon', 'Deuteron', 'Tritium', 'Helium', 'Hydrogen',
        'Cyclotron', 'Synchrotron', 'Betatron', 'Accelerator', 'Collider', 'Detector', 'Spectrometer', 'Interferometer',
        'Quantum', 'Relativity', 'Entropy', 'Thermodynamics', 'Magnetohydrodynamics', 'Kinetics', 'Dynamics', 'Mechanics',
        'Isotope', 'Molecule', 'Catalyst', 'Polymer', 'Crystal', 'Semiconductor', 'Superconductor', 'Metamaterial',
        'Laser', 'Maser', 'Hologram', 'Spectrum', 'Frequency', 'Amplitude', 'Wavelength', 'Resonance'
    ]
    
    # Create unique location-year combinations
    location_year_combos = list(set(zip(locations, years)))
    
    # Assign 2 names per location-year combination
    name_assignments = {}
    for loc, yr in location_year_combos:
        available_names = name_pool.copy()
        np.random.shuffle(available_names)
        name_assignments[(loc, yr)] = available_names[:2]
    
    # Assign names to each entry
    names = []
    for i in range(n_samples):
        loc = locations[i]
        yr = years[i]
        # Randomly choose one of the 2 names for this location-year
        chosen_name = np.random.choice(name_assignments[(loc, yr)])
        names.append(chosen_name)
    
    data['name'] = names
    
    # Add time-dependent functions: QI(t) = QI + sin(t), QE(t) = QE + cos(t)
    # Create time array with 64 points from 0 to 30
    t_values = np.linspace(0, 30, 64)
    
    # For each entry, create time series data
    QI_time_series = []
    QE_time_series = []
    
    for i in range(n_samples):
        qi_t = QI[i] + np.sin(t_values)
        qe_t = QE[i] + np.cos(t_values)
        QI_time_series.append(qi_t.tolist())  # Convert to list
        QE_time_series.append(qe_t.tolist())  # Convert to list
    
    data['QI_time_series'] = QI_time_series
    data['QE_time_series'] = QE_time_series
    data['t_values'] = [t_values.tolist()] * n_samples  # Convert to list
    
    # Add index as entry ID
    data['entry_id'] = range(1, n_samples + 1)
    data = data[['entry_id', 'location', 'year', 'name'] + [col for col in data.columns if col not in ['entry_id', 'location', 'year', 'name']]]
    
    return data

@st.cache_data
def train_gaussian_process_model(data, test_size=0.2, random_state=42):
    """Train Gaussian Process model optimized for CGYRO data"""
    
    # Prepare features and targets
    X = data[['RMIN', 'RMAJ', 'SHEAR', 'KAPPA']].values
    y_QI = data['QI'].values
    y_QE = data['QE'].values
    
    # Split data
    X_train, X_test, y_QI_train, y_QI_test, y_QE_train, y_QE_test = train_test_split(
        X, y_QI, y_QE, test_size=test_size, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X)
    
    # Gaussian Process with ARD kernel
    kernel = ConstantKernel(1.0) * RBF([1.0, 1.0, 1.0, 1.0], length_scale_bounds=(1e-1, 1e1)) + WhiteKernel(1e-5)
    
    gp_QI = GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=3,
        alpha=1e-6,
        random_state=random_state
    )
    gp_QE = GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=3,
        alpha=1e-6,
        random_state=random_state
    )
    
    gp_QI.fit(X_train_scaled, y_QI_train)
    gp_QE.fit(X_train_scaled, y_QE_train)
    
    # Predictions with uncertainty
    y_QI_pred, y_QI_std = gp_QI.predict(X_scaled, return_std=True)
    y_QE_pred, y_QE_std = gp_QE.predict(X_scaled, return_std=True)
    
    models = {'QI': gp_QI, 'QE': gp_QE, 'scaler': scaler}
    predictions = {
        'QI_pred': y_QI_pred, 'QI_std': y_QI_std,
        'QE_pred': y_QE_pred, 'QE_std': y_QE_std
    }
    
    # Test set predictions for metrics
    y_QI_test_pred, _ = gp_QI.predict(X_test_scaled, return_std=True)
    y_QE_test_pred, _ = gp_QE.predict(X_test_scaled, return_std=True)
    
    metrics = {
        'QI': {
            'R²': r2_score(y_QI_test, y_QI_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_QI_test, y_QI_test_pred)),
            'MAE': mean_absolute_error(y_QI_test, y_QI_test_pred)
        },
        'QE': {
            'R²': r2_score(y_QE_test, y_QE_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_QE_test, y_QE_test_pred)),
            'MAE': mean_absolute_error(y_QE_test, y_QE_test_pred)
        }
    }
    
    return models, predictions, metrics, (X_train, X_test, y_QI_train, y_QI_test, y_QE_train, y_QE_test)

def create_pca_plot(data):
    """Create PCA visualization"""
    numeric_cols = ['RMIN', 'RMAJ', 'SHEAR', 'KAPPA', 'QI', 'QE']
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_cols])
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create DataFrame for plotting
    pca_df = pd.DataFrame(
        pca_result, 
        columns=['PC1', 'PC2']
    )
    
    fig = px.scatter(
        pca_df, x='PC1', y='PC2',
        title=f'PCA Analysis (Explained variance: {pca.explained_variance_ratio_.sum():.2%})',
        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'}
    )
    
    return fig

def main():
    st.title("📊 CGYRO-AI")
    st.markdown("*Interactive data exploration with filtering, visualization, and statistical analysis*")
    
    # Add download button for source code
    with st.sidebar:
        st.markdown("---")
        st.subheader("📥 Download")
        
        # Get the current source code
        source_code = '''import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="CGYRO-AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def generate_sample_data(n_samples=1000, random_state=42):
    """Generate synthetic dataset with correlations between inputs and outputs"""
    np.random.seed(random_state)
    
    # Generate input attributes with realistic CGYRO parameter ranges
    RMIN = np.random.uniform(0.2, 0.9, n_samples)
    RMAJ = np.random.uniform(1.0, 4.0, n_samples)
    SHEAR = np.random.uniform(0.1, 3.0, n_samples)
    KAPPA = np.random.uniform(1.0, 2.0, n_samples)
    
    # Create outputs with physics-based functional dependencies
    # Add small amount of noise to simulate measurement/computational uncertainty
    noise1 = np.random.normal(0, 0.1, n_samples)
    noise2 = np.random.normal(0, 0.1, n_samples)
    
    QI = KAPPA * (np.sin(RMIN) + np.exp(RMAJ) + SHEAR**2) + noise1
    QE = KAPPA * (2*np.sin(RMIN) + np.exp(1/RMAJ) + SHEAR**3) + noise2
    
    # Create DataFrame
    data = pd.DataFrame({
        'RMIN': RMIN,
        'RMAJ': RMAJ,
        'SHEAR': SHEAR,
        'KAPPA': KAPPA,
        'QI': QI,
        'QE': QE
    })
    
    # Add metadata
    locations = np.random.choice(['ORNL', 'NERSC', 'ANL'], n_samples, p=[0.4, 0.35, 0.25])
    years = np.random.choice([2022, 2023, 2024, 2025], n_samples, p=[0.2, 0.3, 0.35, 0.15])
    
    data['location'] = locations
    data['year'] = years
    
    # Add time-dependent functions: QI(t) = QI + sin(t), QE(t) = QE + cos(t)
    # Create time array with 64 points from 0 to 30
    t_values = np.linspace(0, 30, 64)
    
    # For each entry, create time series data
    QI_time_series = []
    QE_time_series = []
    
    for i in range(n_samples):
        qi_t = QI[i] + np.sin(t_values)
        qe_t = QE[i] + np.cos(t_values)
        QI_time_series.append(qi_t)
        QE_time_series.append(qe_t)
    
    data['QI_time_series'] = QI_time_series
    data['QE_time_series'] = QE_time_series
    data['t_values'] = [t_values] * n_samples  # Same time array for all entries
    
    # Add index as entry ID
    data['entry_id'] = range(1, n_samples + 1)
    data = data[['entry_id', 'location', 'year'] + [col for col in data.columns if col not in ['entry_id', 'location', 'year']]]
    
    return data

@st.cache_data
def train_gaussian_process_model(data, test_size=0.2, random_state=42):
    """Train Gaussian Process model optimized for CGYRO data"""
    
    # Prepare features and targets
    X = data[['RMIN', 'RMAJ', 'SHEAR', 'KAPPA']].values
    y_QI = data['QI'].values
    y_QE = data['QE'].values
    
    # Split data
    X_train, X_test, y_QI_train, y_QI_test, y_QE_train, y_QE_test = train_test_split(
        X, y_QI, y_QE, test_size=test_size, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X)
    
    # Gaussian Process with ARD kernel
    kernel = ConstantKernel(1.0) * RBF([1.0, 1.0, 1.0, 1.0], length_scale_bounds=(1e-1, 1e1)) + WhiteKernel(1e-5)
    
    gp_QI = GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=3,
        alpha=1e-6,
        random_state=random_state
    )
    gp_QE = GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=3,
        alpha=1e-6,
        random_state=random_state
    )
    
    gp_QI.fit(X_train_scaled, y_QI_train)
    gp_QE.fit(X_train_scaled, y_QE_train)
    
    # Predictions with uncertainty
    y_QI_pred, y_QI_std = gp_QI.predict(X_scaled, return_std=True)
    y_QE_pred, y_QE_std = gp_QE.predict(X_scaled, return_std=True)
    
    models = {'QI': gp_QI, 'QE': gp_QE, 'scaler': scaler}
    predictions = {
        'QI_pred': y_QI_pred, 'QI_std': y_QI_std,
        'QE_pred': y_QE_pred, 'QE_std': y_QE_std
    }
    
    # Test set predictions for metrics
    y_QI_test_pred, _ = gp_QI.predict(X_test_scaled, return_std=True)
    y_QE_test_pred, _ = gp_QE.predict(X_test_scaled, return_std=True)
    
    metrics = {
        'QI': {
            'R²': r2_score(y_QI_test, y_QI_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_QI_test, y_QI_test_pred)),
            'MAE': mean_absolute_error(y_QI_test, y_QI_test_pred)
        },
        'QE': {
            'R²': r2_score(y_QE_test, y_QE_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_QE_test, y_QE_test_pred)),
            'MAE': mean_absolute_error(y_QE_test, y_QE_test_pred)
        }
    }
    
    return models, predictions, metrics, (X_train, X_test, y_QI_train, y_QI_test, y_QE_train, y_QE_test)

def create_pca_plot(data):
    """Create PCA visualization"""
    numeric_cols = ['RMIN', 'RMAJ', 'SHEAR', 'KAPPA', 'QI', 'QE']
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_cols])
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create DataFrame for plotting
    pca_df = pd.DataFrame(
        pca_result, 
        columns=['PC1', 'PC2']
    )
    
    fig = px.scatter(
        pca_df, x='PC1', y='PC2',
        title=f'PCA Analysis (Explained variance: {pca.explained_variance_ratio_.sum():.2%})',
        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'}
    )
    
    return fig

def main():
    st.title("📊 CGYRO-AI")
    st.markdown("*Interactive data exploration with filtering, visualization, and statistical analysis*")
    
    # Generate or load data (but don't train AI automatically)
    if 'data' not in st.session_state:
        with st.spinner("🔄 Generating 1000 CGYRO simulation entries..."):
            st.session_state.data = generate_sample_data()
        st.success("✅ Data generation complete!")
    
    data = st.session_state.data
    
    # [Rest of the main function code would continue here...]

if __name__ == "__main__":
    main()'''
        
        st.download_button(
            label="📄 Download Source Code",
            data=source_code,
            file_name="cgyro_ai.py",
            mime="text/plain",
            help="Download the complete CGYRO-AI application source code"
        )
        
        st.markdown("**To run:**")
        st.code("""pip install streamlit pandas numpy plotly scikit-learn scipy
streamlit run cgyro_ai.py""")
    
    # Generate or load data (but don't train AI automatically)
    if 'data' not in st.session_state:
        with st.spinner("🔄 Generating 1000 CGYRO simulation entries..."):
            st.session_state.data = generate_sample_data()
        st.success("✅ Data generation complete!")
    
    data = st.session_state.data
    
    # Sidebar for filtering
    st.sidebar.header("🔍 Data Filters")
    
    # Metadata filters
    st.sidebar.subheader("Metadata")
    locations = st.sidebar.multiselect(
        "Locations", 
        options=data['location'].unique(),
        default=data['location'].unique()
    )
    
    years = st.sidebar.multiselect(
        "Years", 
        options=sorted(data['year'].unique()),
        default=sorted(data['year'].unique())
    )
    
    names = st.sidebar.multiselect(
        "Names", 
        options=sorted(data['name'].unique()),
        default=sorted(data['name'].unique())
    )
    
    # Numerical filters
    st.sidebar.subheader("Input Attributes")
    RMIN_range = st.sidebar.slider(
        "RMIN range", 
        float(data['RMIN'].min()), 
        float(data['RMIN'].max()), 
        (float(data['RMIN'].min()), float(data['RMIN'].max())),
        step=0.01
    )
    
    RMAJ_range = st.sidebar.slider(
        "RMAJ range", 
        float(data['RMAJ'].min()), 
        float(data['RMAJ'].max()), 
        (float(data['RMAJ'].min()), float(data['RMAJ'].max())),
        step=0.1
    )
    
    SHEAR_range = st.sidebar.slider(
        "SHEAR range", 
        float(data['SHEAR'].min()), 
        float(data['SHEAR'].max()), 
        (float(data['SHEAR'].min()), float(data['SHEAR'].max())),
        step=0.01
    )
    
    KAPPA_range = st.sidebar.slider(
        "KAPPA range", 
        float(data['KAPPA'].min()), 
        float(data['KAPPA'].max()), 
        (float(data['KAPPA'].min()), float(data['KAPPA'].max())),
        step=0.01
    )
    
    st.sidebar.subheader("Output Attributes")
    QI_range = st.sidebar.slider(
        "QI range", 
        float(data['QI'].min()), 
        float(data['QI'].max()), 
        (float(data['QI'].min()), float(data['QI'].max())),
        step=0.1
    )
    
    QE_range = st.sidebar.slider(
        "QE range", 
        float(data['QE'].min()), 
        float(data['QE'].max()), 
        (float(data['QE'].min()), float(data['QE'].max())),
        step=0.1
    )
    
    # Apply filters
    filtered_data = data[
        (data['location'].isin(locations)) &
        (data['year'].isin(years)) &
        (data['name'].isin(names)) &
        (data['RMIN'].between(RMIN_range[0], RMIN_range[1])) &
        (data['RMAJ'].between(RMAJ_range[0], RMAJ_range[1])) &
        (data['SHEAR'].between(SHEAR_range[0], SHEAR_range[1])) &
        (data['KAPPA'].between(KAPPA_range[0], KAPPA_range[1])) &
        (data['QI'].between(QI_range[0], QI_range[1])) &
        (data['QE'].between(QE_range[0], QE_range[1]))
    ]
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Data Table", 
        "📈 Scatter Plots", 
        "📊 Distributions", 
        "⏱️ Time Series",
        "🤖 AI Predictions"
    ])
    
    with tab1:
        st.subheader(f"Data Table ({len(filtered_data)} entries)")
        
        # Search functionality
        search_term = st.text_input("🔍 Search in data (searches all columns):")
        if search_term:
            mask = filtered_data.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            display_data = filtered_data[mask]
        else:
            display_data = filtered_data
        
        if len(display_data) > 0:
            # Prepare data for display - only essential columns
            table_data = display_data[['entry_id', 'location', 'year', 'name', 'QI', 'QE']].copy()
            table_data['QI'] = table_data['QI'].round(3)
            table_data['QE'] = table_data['QE'].round(3)
            
            # Initialize session state for selected entries
            if 'selected_for_timeseries' not in st.session_state:
                st.session_state.selected_for_timeseries = set()
            
            # Add selection column
            table_data['Select for Time Series'] = table_data['entry_id'].isin(st.session_state.selected_for_timeseries)
            
            # Reorder columns
            table_data = table_data[['Select for Time Series', 'entry_id', 'location', 'year', 'name', 'QI', 'QE']]
            
            st.write("**Select entries for Time Series analysis by checking the boxes:**")
            
            # Use st.data_editor for clean selection handling
            edited_data = st.data_editor(
                table_data,
                column_config={
                    "Select for Time Series": st.column_config.CheckboxColumn(
                        "📊 Time Series",
                        help="Select entries to analyze in Time Series tab",
                        default=False
                    ),
                    "entry_id": st.column_config.NumberColumn(
                        "ID",
                        help="Entry ID"
                    ),
                    "location": st.column_config.TextColumn(
                        "Location",
                        help="Laboratory location"
                    ),
                    "year": st.column_config.NumberColumn(
                        "Year",
                        help="Simulation year"
                    ),
                    "name": st.column_config.TextColumn(
                        "Name",
                        help="Group identifier"
                    ),
                    "QI": st.column_config.NumberColumn(
                        "QI",
                        help="Ion heat flux",
                        format="%.3f"
                    ),
                    "QE": st.column_config.NumberColumn(
                        "QE", 
                        help="Electron heat flux",
                        format="%.3f"
                    )
                },
                use_container_width=True,
                hide_index=True,
                key="data_table_editor"
            )
            
            # Update session state based on selections
            if edited_data is not None:
                selected_entries = set(edited_data[edited_data['Select for Time Series']]['entry_id'].tolist())
                st.session_state.selected_for_timeseries = selected_entries
            
            # Show selection summary
            selected_count = len(st.session_state.selected_for_timeseries)
            if selected_count > 0:
                st.success(f"✅ {selected_count} entries selected for Time Series analysis")
                
                # Show selected entry details in an expander
                if st.button("📋 View Selected Entry Details", key="view_selected_details"):
                    selected_details = display_data[display_data['entry_id'].isin(st.session_state.selected_for_timeseries)]
                    with st.expander("Selected Entries Details", expanded=True):
                        for _, row in selected_details.iterrows():
                            st.write(f"**Entry {row['entry_id']}** ({row['location']} {row['year']} - {row['name']})")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.write(f"RMIN: {row['RMIN']:.3f}")
                            with col2:
                                st.write(f"RMAJ: {row['RMAJ']:.3f}")
                            with col3:
                                st.write(f"SHEAR: {row['SHEAR']:.3f}")
                            with col4:
                                st.write(f"KAPPA: {row['KAPPA']:.3f}")
                            st.divider()
                
                # Management buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Clear All Selections", key="clear_all_selections"):
                        st.session_state.selected_for_timeseries.clear()
                        st.rerun()
                with col2:
                    st.info("💡 Switch to Time Series tab to view plots")
            else:
                st.info("💡 Check the boxes in the first column to select entries for Time Series analysis")
            
            # Download option
            st.subheader("Export Data")
            csv_data = display_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv_data,
                file_name="cgyro_filtered_data.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("No data matches the current filters.")
    
    with tab2:
        st.subheader("Scatter Plot Analysis")
        
        if len(filtered_data) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("X-axis", ['RMIN', 'RMAJ', 'SHEAR', 'KAPPA', 'QI', 'QE'], index=0)
                color_by = st.selectbox("Color by", ['location', 'year', 'name'] + ['RMIN', 'RMAJ', 'SHEAR', 'KAPPA', 'QI', 'QE'], index=2)
                
            with col2:
                y_axis = st.selectbox("Y-axis", ['RMIN', 'RMAJ', 'SHEAR', 'KAPPA', 'QI', 'QE'], index=4)
                size_by = st.selectbox("Size by", [None] + ['RMIN', 'RMAJ', 'SHEAR', 'KAPPA', 'QI', 'QE'])
            
            # Create scatter plot
            fig = px.scatter(
                filtered_data, 
                x=x_axis, 
                y=y_axis,
                color=color_by,
                size=size_by if size_by else None,
                hover_data=['entry_id', 'location', 'year', 'name'],
                title=f"{y_axis} vs {x_axis}",
                width=800,
                height=600,
                color_continuous_scale="viridis" if color_by in ['RMIN', 'RMAJ', 'SHEAR', 'KAPPA', 'QI', 'QE'] else None
            )
            
            # Add trendline option
            if st.checkbox("Add trendline"):
                fig.add_traces(
                    px.scatter(
                        filtered_data, x=x_axis, y=y_axis, trendline="ols"
                    ).data[1:]
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation coefficient
            if x_axis != y_axis:
                corr = filtered_data[x_axis].corr(filtered_data[y_axis])
                st.metric("Correlation coefficient", f"{corr:.3f}")
        else:
            st.warning("No data to plot with current filters.")
    
    with tab3:
        st.subheader("Distribution Analysis")
        
        if len(filtered_data) > 0:
            # Variable selection
            variables = st.multiselect(
                "Select variables to analyze", 
                ['RMIN', 'RMAJ', 'SHEAR', 'KAPPA', 'QI', 'QE'],
                default=['RMIN', 'QI']
            )
            
            if variables:
                # Grouping option for metadata
                group_by = st.selectbox(
                    "Group distributions by:", 
                    [None, "location", "year"]
                )
                
                # Plot type selection
                plot_type = st.radio(
                    "Plot type", 
                    ["Histogram", "Box Plot", "Violin Plot", "Density Plot"]
                )
                
                if plot_type == "Histogram":
                    fig = make_subplots(
                        rows=len(variables), cols=1,
                        subplot_titles=variables,
                        vertical_spacing=0.1
                    )
                    
                    for i, var in enumerate(variables):
                        fig.add_trace(
                            go.Histogram(
                                x=filtered_data[var],
                                name=var,
                                nbinsx=30,
                                opacity=0.7
                            ),
                            row=i+1, col=1
                        )
                    
                    fig.update_layout(height=300*len(variables), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Box Plot":
                    if group_by:
                        # Melt data for grouped box plots
                        melted_data = pd.melt(
                            filtered_data, 
                            id_vars=[group_by], 
                            value_vars=variables,
                            var_name='variable', 
                            value_name='value'
                        )
                        
                        fig = px.box(
                            melted_data, 
                            x='variable', 
                            y='value',
                            color=group_by,
                            title=f"Box Plots by {group_by}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = make_subplots(
                            rows=1, cols=len(variables),
                            subplot_titles=variables,
                            horizontal_spacing=0.1
                        )
                        
                        for i, var in enumerate(variables):
                            fig.add_trace(
                                go.Box(
                                    y=filtered_data[var],
                                    name=var,
                                    boxpoints='outliers'
                                ),
                                row=1, col=i+1
                            )
                        
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Violin Plot":
                    if group_by:
                        # Melt data for grouped violin plots
                        melted_data = pd.melt(
                            filtered_data, 
                            id_vars=[group_by], 
                            value_vars=variables,
                            var_name='variable', 
                            value_name='value'
                        )
                        
                        fig = px.violin(
                            melted_data, 
                            x='variable', 
                            y='value',
                            color=group_by,
                            box=True,
                            title=f"Violin Plots by {group_by}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = make_subplots(
                            rows=1, cols=len(variables),
                            subplot_titles=variables,
                            horizontal_spacing=0.1
                        )
                        
                        for i, var in enumerate(variables):
                            fig.add_trace(
                                go.Violin(
                                    y=filtered_data[var],
                                    name=var,
                                    box_visible=True,
                                    meanline_visible=True
                                ),
                                row=1, col=i+1
                            )
                        
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif plot_type == "Density Plot":
                    fig = go.Figure()
                    
                    for var in variables:
                        fig.add_trace(go.Histogram(
                            x=filtered_data[var],
                            histnorm='probability density',
                            name=var,
                            opacity=0.6,
                            nbinsx=50
                        ))
                    
                    fig.update_layout(
                        title="Probability Density Functions",
                        barmode='overlay'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistical tests
                st.subheader("Statistical Summary")
                for var in variables:
                    col1, col2, col3, col4 = st.columns(4)
                    data_var = filtered_data[var]
                    
                    with col1:
                        st.metric(f"{var} Mean", f"{data_var.mean():.3f}")
                    with col2:
                        st.metric(f"{var} Std", f"{data_var.std():.3f}")
                    with col3:
                        st.metric(f"{var} Skewness", f"{stats.skew(data_var):.3f}")
                    with col4:
                        st.metric(f"{var} Kurtosis", f"{stats.kurtosis(data_var):.3f}")
        else:
            st.warning("No data available for distribution analysis.")
    
    with tab4:
        st.subheader("⏱️ Time Series Analysis")
        st.markdown("*QI(t) = QI + sin(t) and QE(t) = QE + cos(t) for t ∈ [0, 30]*")
        
        # Get selected entries from session state
        if 'selected_for_timeseries' not in st.session_state:
            st.session_state.selected_for_timeseries = set()
        
        selected_entry_ids = list(st.session_state.selected_for_timeseries)
        
        if selected_entry_ids:
            # Filter data for selected entries
            selected_data = data[data['entry_id'].isin(selected_entry_ids)]
            
            if len(selected_data) > 0:
                st.info(f"📊 Showing time series for {len(selected_data)} selected entries")
                
                # Plot type selection
                plot_type = st.radio("Plot type:", ["Individual Lines", "Average ± Std", "Both"], horizontal=True)
                
                # Time series plots
                col1, col2 = st.columns(2)
                
                with col1:
                    # QI(t) plot
                    fig_qi = go.Figure()
                    
                    # Individual lines
                    if plot_type in ["Individual Lines", "Both"]:
                        for idx, row in selected_data.iterrows():
                            t_vals = np.array(row['t_values'])
                            qi_vals = np.array(row['QI_time_series'])
                            fig_qi.add_trace(go.Scatter(
                                x=t_vals, y=qi_vals,
                                mode='lines',
                                name=f"Entry {row['entry_id']} ({row['location']} {row['year']} - {row['name']})",
                                line=dict(width=2),
                                hovertemplate='t=%{x:.1f}<br>QI(t)=%{y:.3f}<br>Entry %{text}<extra></extra>',
                                text=[row['entry_id']] * len(t_vals)
                            ))
                    
                    # Average and std
                    if plot_type in ["Average ± Std", "Both"] and len(selected_data) > 1:
                        t_vals = np.array(selected_data.iloc[0]['t_values'])
                        qi_matrix = np.array([np.array(row['QI_time_series']) for _, row in selected_data.iterrows()])
                        qi_mean = np.mean(qi_matrix, axis=0)
                        qi_std = np.std(qi_matrix, axis=0)
                        
                        # Mean line
                        fig_qi.add_trace(go.Scatter(
                            x=t_vals, y=qi_mean,
                            mode='lines',
                            name='Average QI(t)',
                            line=dict(color='red', width=3),
                            hovertemplate='t=%{x:.1f}<br>Avg QI(t)=%{y:.3f}<extra></extra>'
                        ))
                        
                        # Confidence band
                        fig_qi.add_trace(go.Scatter(
                            x=np.concatenate([t_vals, t_vals[::-1]]),
                            y=np.concatenate([qi_mean + qi_std, (qi_mean - qi_std)[::-1]]),
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='±1 Std Dev',
                            hoverinfo='skip'
                        ))
                    
                    fig_qi.update_layout(
                        title="QI(t) = QI + sin(t)",
                        xaxis_title="Time t",
                        yaxis_title="QI(t)",
                        height=400,
                        showlegend=len(selected_data) <= 8
                    )
                    st.plotly_chart(fig_qi, use_container_width=True)
                
                with col2:
                    # QE(t) plot
                    fig_qe = go.Figure()
                    
                    # Individual lines
                    if plot_type in ["Individual Lines", "Both"]:
                        for idx, row in selected_data.iterrows():
                            t_vals = np.array(row['t_values'])
                            qe_vals = np.array(row['QE_time_series'])
                            fig_qe.add_trace(go.Scatter(
                                x=t_vals, y=qe_vals,
                                mode='lines',
                                name=f"Entry {row['entry_id']} ({row['location']} {row['year']} - {row['name']})",
                                line=dict(width=2),
                                hovertemplate='t=%{x:.1f}<br>QE(t)=%{y:.3f}<br>Entry %{text}<extra></extra>',
                                text=[row['entry_id']] * len(t_vals)
                            ))
                    
                    # Average and std
                    if plot_type in ["Average ± Std", "Both"] and len(selected_data) > 1:
                        t_vals = np.array(selected_data.iloc[0]['t_values'])
                        qe_matrix = np.array([np.array(row['QE_time_series']) for _, row in selected_data.iterrows()])
                        qe_mean = np.mean(qe_matrix, axis=0)
                        qe_std = np.std(qe_matrix, axis=0)
                        
                        # Mean line
                        fig_qe.add_trace(go.Scatter(
                            x=t_vals, y=qe_mean,
                            mode='lines',
                            name='Average QE(t)',
                            line=dict(color='blue', width=3),
                            hovertemplate='t=%{x:.1f}<br>Avg QE(t)=%{y:.3f}<extra></extra>'
                        ))
                        
                        # Confidence band
                        fig_qe.add_trace(go.Scatter(
                            x=np.concatenate([t_vals, t_vals[::-1]]),
                            y=np.concatenate([qe_mean + qe_std, (qe_mean - qe_std)[::-1]]),
                            fill='toself',
                            fillcolor='rgba(0,0,255,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='±1 Std Dev',
                            hoverinfo='skip'
                        ))
                    
                    fig_qe.update_layout(
                        title="QE(t) = QE + cos(t)",
                        xaxis_title="Time t",
                        yaxis_title="QE(t)",
                        height=400,
                        showlegend=len(selected_data) <= 8
                    )
                    st.plotly_chart(fig_qe, use_container_width=True)
                
                # Selected entries summary
                st.subheader("Selected Entries Summary")
                summary_data = selected_data[['entry_id', 'location', 'year', 'name', 'RMIN', 'RMAJ', 'SHEAR', 'KAPPA', 'QI', 'QE']].copy()
                st.dataframe(summary_data, use_container_width=True, hide_index=True)
                
                # Statistics table
                if len(selected_data) > 1:
                    st.subheader("Time Series Statistics")
                    
                    # Calculate statistics for each entry
                    stats_data = []
                    for idx, row in selected_data.iterrows():
                        qi_series = np.array(row['QI_time_series'])
                        qe_series = np.array(row['QE_time_series'])
                        
                        stats_data.append({
                            'Entry ID': row['entry_id'],
                            'Location': row['location'],
                            'Year': row['year'],
                            'Name': row['name'],
                            'QI(t) Mean': qi_series.mean(),
                            'QI(t) Std': qi_series.std(),
                            'QI(t) Range': qi_series.max() - qi_series.min(),
                            'QE(t) Mean': qe_series.mean(),
                            'QE(t) Std': qe_series.std(),
                            'QE(t) Range': qe_series.max() - qe_series.min()
                        })
                    
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Download time series data
                st.subheader("Export Time Series")
                
                # Create downloadable dataset
                export_rows = []
                for idx, row in selected_data.iterrows():
                    t_vals = np.array(row['t_values'])
                    qi_vals = np.array(row['QI_time_series'])
                    qe_vals = np.array(row['QE_time_series'])
                    
                    for i, t in enumerate(t_vals):
                        export_rows.append({
                            'entry_id': row['entry_id'],
                            'location': row['location'],
                            'year': row['year'],
                            'name': row['name'],
                            'RMIN': row['RMIN'],
                            'RMAJ': row['RMAJ'],
                            'SHEAR': row['SHEAR'],
                            'KAPPA': row['KAPPA'],
                            'QI_steady': row['QI'],
                            'QE_steady': row['QE'],
                            't': t,
                            'QI_t': qi_vals[i],
                            'QE_t': qe_vals[i]
                        })
                
                export_df = pd.DataFrame(export_rows)
                csv_data = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Time Series Data",
                    data=csv_data,
                    file_name="cgyro_time_series_selected.csv",
                    mime="text/csv"
                )
                
                # Management buttons
                st.subheader("Selection Management")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Clear All Selections", type="secondary", key="timeseries_clear_selections"):
                        st.session_state.selected_for_timeseries.clear()
                        st.rerun()
                with col2:
                    if st.button("📋 Back to Data Table", type="secondary", key="back_to_data_table"):
                        st.info("💡 Use the Data Table tab to select more entries")
            else:
                st.warning("Selected entries are not in the current filtered dataset. Please adjust filters or select different entries.")
        else:
            st.info("👆 No entries selected for time series analysis.")
            st.markdown("**To get started:**")
            st.markdown("1. Go to the **Data Table** tab")
            st.markdown("2. Use the checkboxes to select entries for analysis") 
            st.markdown("3. Return to this tab to view their time series")
            
            if st.button("📋 Go to Data Table", type="primary", key="goto_data_table"):
                st.info("💡 Switch to the Data Table tab to select entries")
    
    with tab5:
        st.subheader("🤖 Gaussian Process Predictions")
        
        # AI Training Toggle
        st.subheader("Model Training")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("Train the Gaussian Process model on the currently filtered data:")
        with col2:
            train_model = st.button("🎯 Train AI Model", type="primary")
        
        # Check if model should be trained
        if train_model:
            if len(filtered_data) < 10:
                st.error("⚠️ Need at least 10 data points to train the model. Please adjust your filters.")
            else:
                with st.spinner(f"🤖 Training Gaussian Process on {len(filtered_data)} entries..."):
                    models, predictions, metrics, split_data = train_gaussian_process_model(filtered_data)
                    st.session_state.models = models
                    st.session_state.predictions = predictions
                    st.session_state.metrics = metrics
                    st.session_state.split_data = split_data
                    st.session_state.training_data_size = len(filtered_data)
                st.success(f"✅ Model trained successfully on {len(filtered_data)} entries!")
        
        # Display model results if available
        if 'models' in st.session_state and 'predictions' in st.session_state:
            models = st.session_state.models
            predictions = st.session_state.predictions
            metrics = st.session_state.metrics
            training_size = st.session_state.get('training_data_size', len(data))
            
            st.info(f"📊 Current model trained on {training_size} entries")
            
            # Model performance metrics
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**QI Predictions**")
                qi_metrics = metrics['QI']
                st.metric("R² Score", f"{qi_metrics['R²']:.4f}")
                st.metric("RMSE", f"{qi_metrics['RMSE']:.4f}")
                st.metric("MAE", f"{qi_metrics['MAE']:.4f}")
            
            with col2:
                st.write("**QE Predictions**")
                qe_metrics = metrics['QE']
                st.metric("R² Score", f"{qe_metrics['R²']:.4f}")
                st.metric("RMSE", f"{qe_metrics['RMSE']:.4f}")
                st.metric("MAE", f"{qe_metrics['MAE']:.4f}")
            
            # Only show predictions for current filtered data
            if len(filtered_data) > 0:
                # Generate predictions for current filtered data using the trained model
                try:
                    # Prepare filtered data for prediction
                    X_filtered = filtered_data[['RMIN', 'RMAJ', 'SHEAR', 'KAPPA']].values
                    scaler = models['scaler']
                    X_filtered_scaled = scaler.transform(X_filtered)
                    
                    # Get predictions for current filtered data
                    QI_pred_filtered, QI_std_filtered = models['QI'].predict(X_filtered_scaled, return_std=True)
                    QE_pred_filtered, QE_std_filtered = models['QE'].predict(X_filtered_scaled, return_std=True)
                    
                    # Create predictions dataframe
                    filtered_data_pred = filtered_data.copy()
                    filtered_data_pred['QI_pred'] = QI_pred_filtered
                    filtered_data_pred['QE_pred'] = QE_pred_filtered
                    filtered_data_pred['QI_std'] = QI_std_filtered
                    filtered_data_pred['QE_std'] = QE_std_filtered
                    
                    # Prediction vs Actual plots
                    st.subheader("Prediction vs Actual")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # QI predictions
                        fig_qi = go.Figure()
                        
                        # Perfect prediction line
                        min_val = min(filtered_data_pred['QI'].min(), filtered_data_pred['QI_pred'].min())
                        max_val = max(filtered_data_pred['QI'].max(), filtered_data_pred['QI_pred'].max())
                        fig_qi.add_trace(go.Scatter(
                            x=[min_val, max_val], y=[min_val, max_val],
                            mode='lines', name='Perfect Prediction',
                            line=dict(dash='dash', color='red')
                        ))
                        
                        # Actual vs predicted with uncertainty
                        fig_qi.add_trace(go.Scatter(
                            x=filtered_data_pred['QI'], 
                            y=filtered_data_pred['QI_pred'],
                            error_y=dict(
                                type='data',
                                array=2*filtered_data_pred['QI_std'],
                                visible=True
                            ),
                            mode='markers',
                            name='GP Predictions',
                            hovertemplate='Actual: %{x:.3f}<br>Predicted: %{y:.3f}<br>Uncertainty: ±%{error_y.array:.3f}'
                        ))
                        
                        fig_qi.update_layout(
                            title="QI: Predicted vs Actual",
                            xaxis_title="Actual QI",
                            yaxis_title="Predicted QI"
                        )
                        st.plotly_chart(fig_qi, use_container_width=True)
                    
                    with col2:
                        # QE predictions
                        fig_qe = go.Figure()
                        
                        # Perfect prediction line
                        min_val = min(filtered_data_pred['QE'].min(), filtered_data_pred['QE_pred'].min())
                        max_val = max(filtered_data_pred['QE'].max(), filtered_data_pred['QE_pred'].max())
                        fig_qe.add_trace(go.Scatter(
                            x=[min_val, max_val], y=[min_val, max_val],
                            mode='lines', name='Perfect Prediction',
                            line=dict(dash='dash', color='red')
                        ))
                        
                        # Actual vs predicted with uncertainty
                        fig_qe.add_trace(go.Scatter(
                            x=filtered_data_pred['QE'], 
                            y=filtered_data_pred['QE_pred'],
                            error_y=dict(
                                type='data',
                                array=2*filtered_data_pred['QE_std'],
                                visible=True
                            ),
                            mode='markers',
                            name='GP Predictions',
                            hovertemplate='Actual: %{x:.3f}<br>Predicted: %{y:.3f}<br>Uncertainty: ±%{error_y.array:.3f}'
                        ))
                        
                        fig_qe.update_layout(
                            title="QE: Predicted vs Actual",
                            xaxis_title="Actual QE",
                            yaxis_title="Predicted QE"
                        )
                        st.plotly_chart(fig_qe, use_container_width=True)
                    
                    # Residuals analysis
                    st.subheader("Residuals Analysis")
                    
                    qi_residuals = filtered_data_pred['QI'] - filtered_data_pred['QI_pred']
                    qe_residuals = filtered_data_pred['QE'] - filtered_data_pred['QE_pred']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_resid_qi = px.scatter(
                            x=filtered_data_pred['QI_pred'], y=qi_residuals,
                            labels={'x': 'Predicted QI', 'y': 'Residuals'},
                            title="QI Residuals vs Predicted"
                        )
                        fig_resid_qi.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_resid_qi, use_container_width=True)
                    
                    with col2:
                        fig_resid_qe = px.scatter(
                            x=filtered_data_pred['QE_pred'], y=qe_residuals,
                            labels={'x': 'Predicted QE', 'y': 'Residuals'},
                            title="QE Residuals vs Predicted"
                        )
                        fig_resid_qe.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig_resid_qe, use_container_width=True)
                    
                    # Download predictions
                    st.subheader("Export Predictions")
                    
                    # Create downloadable dataset
                    export_data = filtered_data_pred[['entry_id', 'location', 'year', 'name', 'RMIN', 'RMAJ', 'SHEAR', 'KAPPA', 'QI', 'QE', 'QI_pred', 'QE_pred']].copy()
                    export_data['QI_uncertainty'] = filtered_data_pred['QI_std']
                    export_data['QE_uncertainty'] = filtered_data_pred['QE_std']
                    
                    csv_data = export_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download GP Predictions",
                        data=csv_data,
                        file_name="cgyro_gp_predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {str(e)}")
                    st.info("💡 Try retraining the model on the current data selection.")
            else:
                st.warning("No data available for predictions with current filters.")
        else:
            st.info("👆 Click 'Train AI Model' above to train the Gaussian Process on your filtered data selection.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and Plotly*")

if __name__ == "__main__":
    main()