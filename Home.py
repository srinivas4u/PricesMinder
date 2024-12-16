import streamlit as st
import pandas as pd
from datetime import datetime
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

# Set page configuration
page_title = "Prices Minder"
st.set_page_config(
    page_title=page_title,
    page_icon="shark",  # You can use an emoji or a URL to an image
    layout="wide",
    initial_sidebar_state="auto",
)

description = "Prices Minder"
keywords = "Prices Minder"

# Inject custom HTML for meta tags
st.markdown(f"""
    <meta name="description" content="{description}">
    <meta name="keywords" content="{keywords}">
""", unsafe_allow_html=True)

# Your Streamlit app content goes here
#st.title("Welcome to Prices Minder")
#st.write("This is a simple app to track prices.")
# Inject custom CSS for font size and table styles
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        font-size: 10px;
        max-width: 100%;
        margin-left: 0;
        margin-right: auto;
    }
    .sidebar .sidebar-content {
        font-size: 10px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
    }
    th {
        background-color: #f2f2f2;
        text-align: left;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    tr:hover {
        background-color: #ddd;
    }
    </style>
""",unsafe_allow_html=True)

# Load paths from secrets
static_files_path = st.secrets["paths"]["static_files"]
data_files_path = st.secrets["paths"]["data_files"]
csv_files = st.secrets["csv_files"]

# Add session state for selected file and modify option
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = 'in'
if 'modify' not in st.session_state:
    st.session_state.modify = True

# Display the selected file
st.write(f"Disk Prices Minder- {st.session_state.selected_file.upper()}")

with st.sidebar:
    file_keys = list(csv_files.keys())
    display_keys = [f"amazon.{key}" for key in file_keys]
    selected_display_key = st.radio("Data Source", display_keys, index=display_keys.index(f"amazon.{st.session_state.selected_file}"))
    st.session_state.selected_file = selected_display_key.replace("amazon.", "")
    st.session_state.modify = st.checkbox("Price Filters", value=st.session_state.modify)

    # Add radio button for Price per GB or Price per TB selection, default to Price per TB
    price_option = st.radio("Select Price Option", ("Price per GB", "Price per TB"), index=1)
# Add caching for data loading
@st.cache_data
def load_data(file_key):
    file_path = csv_files[file_key]
    # Read the CSV file
    df = pd.read_csv(file_path, sep=',', encoding='utf-8', low_memory=False, on_bad_lines='skip')

    # Clean Capacity column to handle cases like '64 GB x2', '128 GB', '123 GB'
    def clean_capacity(capacity):
        capacity_value, capacity_str = None, None
        base_capacity_value = None  # Initialize base_capacity_value

        if isinstance(capacity, str):
            capacity = capacity.strip()  # Remove leading and trailing spaces
            if ' x' in capacity:
                parts = capacity.split(' x')
                base_capacity = parts[0].strip()
                multiplier = int(parts[1].strip())
                if 'TB' in base_capacity:
                    base_capacity_value = float(base_capacity.replace('TB', '').strip()) * 1024
                    capacity_str = f"{base_capacity_value / 1024} TB"
                elif 'GB' in base_capacity:
                    base_capacity_value = float(base_capacity.replace('GB', '').strip())
                    capacity_str = f"{base_capacity_value} GB"
                if base_capacity_value is not None:
                    capacity_value = base_capacity_value * multiplier
            elif 'TB' in capacity:
                base_capacity_value = float(capacity.replace('TB', '').strip()) * 1024
                capacity_str = f"{base_capacity_value / 1024} TB"
                capacity_value = base_capacity_value
            # Clean Capacity column to handle cases like '64 GB x2', '128 GB', '123 GB'
            elif 'GB' in capacity:
                base_capacity_value = float(capacity.replace('GB', '').strip())
                capacity_str = f"{base_capacity_value} GB"
                capacity_value = base_capacity_value
        return capacity_value, capacity_str

    df[['Capacity_GB', 'Capacity']] = df['Capacity'].apply(clean_capacity).apply(pd.Series)

    # Remove rupees symbol, dot, and comma in Price column
    df['Price'] = df['Price'].str.replace('₹', '').str.replace('£', '').str.replace('€', '').str.replace('A$', '').str.replace('$', '').str.replace(' ', '')

    # Ensure Price column is numeric
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    df['Price per GB'] = df['Price'] / df['Capacity_GB']
    df['Price per TB'] = df['Price'] / (df['Capacity_GB'] / 1024)

    return df

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not st.session_state.modify:
        return df

    df = df.copy()

    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])

            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    # Filter columns in sidebar
    with st.sidebar:
        to_filter_columns = df.columns.tolist()
        to_filter_columns.remove('Affiliate Link')
        if 'Affiliate Link HTML' in to_filter_columns:
            to_filter_columns.remove('Affiliate Link HTML')
        if 'Capacity_GB' in to_filter_columns:
            to_filter_columns.remove('Capacity_GB')

            # Sort columns in ascending order
        to_filter_columns.sort()

        for column in to_filter_columns:
            if is_categorical_dtype(df[column]) or df[column].nunique() < 20:
                st.markdown(f"**{column}**")
                unique_values = df[column].unique()
                selected_values = []
                for value in unique_values:
                    if st.checkbox(f"{value}", key=f"{column}_{value}"):
                        selected_values.append(value)
                if selected_values:
                    df = df[df[column].isin(selected_values)]
                
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100

                # Apply slider logic only if the column is selected in the sidebar toggle
                if (column == price_option):
                    user_num_input = st.slider(
                        f"Based on {column}",
                        _min,
                        _max,
                        (_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]

            elif column == "Capacity_GB":
                capacity_min = float(df[column].min())
                capacity_max = float(df[column].max())
                capacity_value = st.number_input(
                    f"Capacity_GB ({column})",
                    min_value=capacity_min,
                    max_value=capacity_max,
                    value=capacity_min,
                    step=1.0,
                )
                df = df[df[column] >= capacity_value]

            elif is_datetime64_any_dtype(df[column]):
                user_date_input = st.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = st.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df

df = load_data(st.session_state.selected_file)

if df is not None and not df.empty:
    df_reset = df.reset_index(drop=True)
else:
    st.write("No data available to display.")
    df_reset = pd.DataFrame()

filtered_df = filter_dataframe(df_reset)

# Define a dictionary to map countries to their currency symbols
currency_symbols = {
    "in": "₹",
    "us": "$",
    "au": "$",
    "es": "€",
    "it": "€",
    "de": "€",
    "uk": "£",
    "fr": "€"
}

# Get the selected currency symbol
currency_symbol = currency_symbols[st.session_state.selected_file]

# Add the currency symbol as prefix and include comma in Price per GB and Price per TB columns
filtered_df['Price per GB'] = filtered_df['Price per GB'].apply(lambda x: f"{currency_symbol}{x:,.2f}" if pd.notnull(x) else "N/A")
filtered_df['Price per TB'] = filtered_df['Price per TB'].apply(lambda x: f"{currency_symbol}{x:,.2f}" if pd.notnull(x) else "N/A")
filtered_df['Price'] = filtered_df['Price'].apply(lambda x: f"{currency_symbol}{x:,.2f}" if pd.notnull(x) else "N/A")

# Remove 'Affiliate Link' column and make 'Affiliate Link HTML' clickable
filtered_df.drop(columns=['Affiliate Link'], inplace=True)

# Remove columns based on Price per GB and Price per TB selection
if price_option == "Price per GB":
    filtered_df.drop(columns=['Price per TB'], inplace=True)
else:
    filtered_df.drop(columns=['Price per GB'], inplace=True)

#Drop Capacity_GB, Prices per GB and Prices Per TB
    
filtered_df.drop(columns=['Capacity_GB','Price per GB','Price per TB'], inplace=True)


# Display the filtered DataFrame using HTML markdown

html_table=filtered_df.to_html(escape=False,index=False)
st.markdown(html_table,unsafe_allow_html=True)
           