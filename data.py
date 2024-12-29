import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import warnings

### METHOD DEFINITIONS ###
def graph(file1: str, file2: str, subsample_rate: int, x_label_rotation: int, y_label_rotation: int, y_tick_interval: int, graph_title: str, x_axis_label: str, y_axis_label: str, line1_name: str, line2_name: str):
    """
    Plot two time series from two CSV files with customizable formatting options.

    Parameters:
    file1 (str): Path to the first CSV file containing the data.
    file2 (str): Path to the second CSV file containing the data.
    subsample_rate (int): The rate at which to subsample the data (every Nth row).
    x_label_rotation (int): The rotation angle for the x-axis labels (in degrees).
    y_label_rotation (int): The rotation angle for the y-axis labels (in degrees).
    y_tick_interval (int): The interval at which to place y-axis ticks.
    graph_title (str): The title of the graph.
    x_axis_label (str): The label for the x-axis (date or time column).
    y_axis_label (str): The label for the y-axis (the values to be plotted).
    line1_name (str): The name for the first line in the legend.
    line2_name (str): The name for the second line in the legend.

    Returns:
    None

    This function reads two time series from CSV files, subsamples them according to the given rate, 
    and plots them on the same graph. The x-axis is formatted with custom rotation, and the y-axis 
    ticks are placed based on the provided interval. The graph includes a title, axis labels, a legend, 
    and gridlines for improved readability.

    The x-axis is adjusted to include both time series and ensure the data is properly aligned.
    """

    # Read the CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Convert the x-axis values to datetime format
    df1[x_axis_label] = pd.to_datetime(df1[x_axis_label], format='%m/%d/%Y')
    df2[x_axis_label] = pd.to_datetime(df2[x_axis_label], format='%m/%d/%Y')

    # Subsample the DataFrames by the specified rate (e.g., every Nth row)
    df1_subsampled = df1.iloc[::subsample_rate]
    df2_subsampled = df2.iloc[::subsample_rate]

    # Plot the first dataset (df1) with the provided line1_name for the legend
    plt.plot(df1_subsampled[x_axis_label], df1_subsampled[y_axis_label], label=line1_name, color='#ECA400', linewidth=2)

    # Plot the second dataset (df2) with the provided line2_name for the legend
    plt.plot(df2_subsampled[x_axis_label], df2_subsampled[y_axis_label], label=line2_name, color='#8AC6D0', linewidth=2)

    # Determine the x-axis limits based on the shortest line (furthest data point of the shortest line)
    min_x = min(df1_subsampled[x_axis_label].iloc[-1], df2_subsampled[x_axis_label].iloc[-1])
    max_x = max(df1_subsampled[x_axis_label].iloc[-1], df2_subsampled[x_axis_label].iloc[-1])

    # Set custom x-ticks (every Nth x-axis label) with rotation
    # Use the union of both dataframes' x-axis labels to ensure both lines are aligned
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=subsample_rate * 5))  # 'nbins' controls the max number of x-ticks
    #ax.set_facecolor('#B0C0BC')

    # Set the x-tick interval as a day frequency, use date formatting
    plt.xticks(rotation=x_label_rotation)

    # Set custom y-ticks with rotation, and intervals based on the provided y_tick_interval
    yticks_range = range(0,
                     int(max(df1[y_axis_label].max(), df2[y_axis_label].max()) // y_tick_interval + 1) * y_tick_interval, 
                     y_tick_interval
                     )


    plt.yticks(yticks_range, rotation=y_label_rotation)

    # Set axis labels and title with improved font size
    plt.xlabel(x_axis_label, fontsize=16)
    plt.ylabel(y_axis_label, fontsize=16)
    plt.title(graph_title, fontsize=20, fontweight='bold')

    # Add a legend with a specific location using line1_name and line2_name
    plt.legend(loc='best', fontsize=10)

    # Add gridlines (subtle but helpful for readability)
    plt.grid(True, linestyle='-', alpha=0.3)

    # Set the x-axis limits to the furthest data point of the shortest line
    plt.xlim(min(df1_subsampled[x_axis_label].iloc[0], df2_subsampled[x_axis_label].iloc[0]), min_x)

    # Adjust layout to prevent label clipping
    plt.tight_layout(pad=1.5)  # You can increase/decrease pad to control spacing

    # Adjust the left/right margins to avoid extra space
    plt.subplots_adjust(left=0.07, right=0.93)  # Reduces the space on left and right

    # Display the plot
    plt.show()

import pandas as pd
import numpy as np

def calculate_correlation(file1: str, file2: str, column1: str, column2: str, x_axis_label: str, CPI=False):
    """
    Calculate the Pearson correlation and its squared value (R^2) between two columns from two time series CSV files.
    
    Parameters:
    file1 (str): Path to the first CSV file containing the data.
    file2 (str): Path to the second CSV file containing the data.
    column1 (str): The name of the column in the first CSV file to correlate.
    column2 (str): The name of the column in the second CSV file to correlate.
    x_axis_label (str): The name of the column containing the date or time values, used as the common key for merging.
    CPI (bool): If True, merge based on year; if False, merge based on exact date.

    Returns:
    tuple: A tuple containing the Pearson correlation and its squared value (R^2).
        - correlation (float): The Pearson correlation coefficient between the two specified columns.
        - correlation_squared (float): The square of the correlation coefficient (R^2), indicating the proportion 
          of variance in one variable explained by the other.
    
    The function assumes that both datasets share the same date format ('%m/%d/%Y') and will only keep rows 
    with matching date values from both CSV files.
    """

    # Read the CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Clean column names by stripping any leading/trailing spaces
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # Convert the date columns (if they're strings) to datetime objects for correct alignment
    df1[x_axis_label] = pd.to_datetime(df1[x_axis_label], format='%m/%d/%Y')
    df2[x_axis_label] = pd.to_datetime(df2[x_axis_label], format='%m/%d/%Y')

    # If CPI is True, extract the year from the date column
    if CPI:
        df1['Year'] = df1[x_axis_label].dt.year
        df2['Year'] = df2[x_axis_label].dt.year
        # Merge on the Year column, taking the average for each year
        df1 = df1.groupby('Year').agg({column1: 'mean'}).reset_index()
        df2 = df2.groupby('Year').agg({column2: 'mean'}).reset_index()
        df_merged = pd.merge(df1, df2, on='Year', how='inner')
    else:
        # Merge both DataFrames on the exact date
        df_merged = pd.merge(df1, df2, on=x_axis_label, how='inner')

    # Calculate the correlation between the two specified columns
    correlation = df_merged[column1].corr(df_merged[column2])

    # Calculate the correlation squared (R^2)
    correlation_squared = correlation ** 2

    return correlation, correlation_squared


def calculate_line(file1: str, file2: str, x_axis_label: str, CPI=False):
    """
    Calculate the slope and intercept of the linear relationship between two time series from CSV files.
    
    Parameters:
    file1 (str): Path to the first CSV file containing the data.
    file2 (str): Path to the second CSV file containing the data.
    x_axis_label (str): Name of the column containing the date or time values, used as the common key for merging.
    CPI (bool): If True, merge based on year; if False, merge based on exact date.

    Returns:
    tuple: A tuple containing the slope and intercept of the linear regression line.
        - slope (float): The rate of change between the two variables (Value1 and Value2).
        - intercept (float): The value of Value2 when Value1 is zero.
    
    The function assumes that the two datasets share the same date column format ('%m/%d/%Y') and that the 
    target columns are labeled 'Value_x' and 'Value_y', which are renamed to 'Value1' and 'Value2' before 
    performing the regression.
    """

    # Read the CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Clean column names by stripping any leading/trailing spaces
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # Convert the date columns (if they're strings) to datetime objects for correct alignment
    df1[x_axis_label] = pd.to_datetime(df1[x_axis_label], format='%m/%d/%Y')
    df2[x_axis_label] = pd.to_datetime(df2[x_axis_label], format='%m/%d/%Y')

    # If CPI is True, extract the year from the date column
    if CPI:
        df1['Year'] = df1[x_axis_label].dt.year
        df2['Year'] = df2[x_axis_label].dt.year
        # Merge on the Year column
        df_merged = pd.merge(df1, df2, left_on='Year', right_on='Year', how='inner')
    else:
        # Merge both DataFrames on the exact date
        df_merged = pd.merge(df1, df2, on=x_axis_label, how='inner')

    # Rename the 'Value_x' and 'Value_y' columns to a common name
    df_merged.rename(columns={'Value_x': 'Value1', 'Value_y': 'Value2'}, inplace=True)

    # Perform linear regression to get the slope and intercept
    x = df_merged['Value1'].values  # Independent variable (Value1)
    y = df_merged['Value2'].values  # Dependent variable (Value2)

    # Use numpy's polyfit to perform a linear regression (1st degree polynomial = line)
    slope, intercept = np.polyfit(x, y, 1)  # The "1" specifies a linear fit (degree 1)

    return slope, intercept

def graph_lag(csv_file1, csv_file2, title, max_lag_months=60):
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Degrees of freedom <= 0 for slice.*")
    """
    Calculate and plot lag correlation between two time series from CSV files.

    Parameters:
    csv_file1 (str): Path to the first CSV file containing the data.
    csv_file2 (str): Path to the second CSV file containing the data.
    title (str): Title for the plot.
    max_lag_months (int, optional): Maximum number of months to calculate the lag for. Default is 60.

    Returns:
    tuple: A tuple containing:
        - max_corr_lag (int): The lag (in months) at which the maximum correlation occurs.
        - max_correlation (float): The maximum correlation value at that lag.

    This function reads two time series from CSV files, calculates the correlation for different lag values 
    (from 0 to max_lag_months), and plots the correlation as a function of the lag. The plot highlights the 
    point of maximum correlation, with the lag and value displayed in the legend. The function also prints the 
    length of the datasets and the number of data points used for analysis.
    """

    
    # Read CSV files
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    
    # Print original lengths
    print(f"Original lengths - Dataset 1: {len(df1)}, Dataset 2: {len(df2)}")
    
    # Get the shorter length
    min_length = min(len(df1), len(df2))
    
    # Trim both datasets to the shorter length
    values1 = df1.iloc[:min_length, 1].values
    values2 = df2.iloc[:min_length, 1].values
    
    # Adjust max_lag_months if it's larger than the data length
    max_lag_months = min(max_lag_months, min_length - 1)
    
    print(f"Using {min_length} data points and maximum lag of {max_lag_months} months")
    
    # Calculate correlations
    correlations = []
    lags = range(0, max_lag_months + 1)
    
    for lag in lags:
        if lag == 0:
            corr = np.corrcoef(values1, values2)[0,1]
        else:
            corr = np.corrcoef(values1[:-lag], values2[lag:])[0,1]
        correlations.append(corr)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(lags, correlations, '-', color='#8AC6D0')
    plt.title(title)
    plt.xlabel('Lag (months)')
    plt.ylabel('Correlation')
    plt.grid(True)
    
    # Add max correlation point
    max_corr_lag = lags[np.argmax(correlations)]
    max_correlation = max(correlations)
    plt.plot(max_corr_lag, max_correlation, 'o', color='#ECA400', label=f'Max correlation at lag {max_corr_lag}')
    plt.legend()
    
    plt.show()
    
    return max_corr_lag, max_correlation

### RUN CODE ###
# CPI and HPI #
print("---------------- CPI and HPI ----------------")
graph(
    file1="data_files/cpi/cpi_data.csv",
    file2="data_files/hpi/hpi_data.csv",
    subsample_rate=10,  # Subsample every 10th value
    x_label_rotation=45,  # Rotate x-axis labels by 45 degrees
    y_label_rotation=0,  # No rotation for y-axis labels
    y_tick_interval=50,  # Y-ticks every 50 units
    graph_title="CPI Vs. HPI",  # Title of the graph
    x_axis_label="Date",  # Label for x-axis
    y_axis_label="Value",  # Label for y-axis
    line1_name="CPI",
    line2_name="HPI"
)
r, r_squared = calculate_correlation(
    file1="data_files/cpi/cpi_data.csv",
    file2="data_files/hpi/hpi_data.csv",
    column1="Date",
    column2="Value",
    x_axis_label="Date",
    CPI=True
)
slope, y_intercept = calculate_line(
    file1="data_files/cpi/cpi_data.csv",
    file2="data_files/hpi/hpi_data.csv",
    x_axis_label="Date",
    CPI=True
)
print("Slope:", slope)
print("Y-Intercept:", y_intercept)
print("Correlation(r): ", r)
print("Coefficient of Determination(R^2): ", r_squared)
max_corr_lag, max_corr = graph_lag(
    csv_file1="data_files/cpi/cpi_data.csv",
    csv_file2="data_files/hpi/hpi_data.csv",
    title="Lag Between CPI and HPI"
)
print("Max Lag Correlation: ", max_corr_lag)
print("Max Correlation", max_corr)

# GDP and HPI #
print("\n---------------- GDP and HPI ----------------")
graph(
    file1="data_files/gdp/gdp_data.csv",
    file2="data_files/hpi/hpi_data.csv",
    subsample_rate=10,  # Subsample every 10th value
    x_label_rotation=45,  # Rotate x-axis labels by 45 degrees
    y_label_rotation=0,  # No rotation for y-axis labels
    y_tick_interval=50,  # Y-ticks every 50 units
    graph_title="GDP Vs. HPI",  # Title of the graph
    x_axis_label="Date",  # Label for x-axis
    y_axis_label="Value",  # Label for y-axis
    line1_name="GDP",
    line2_name="HPI"
)
r, r_squared = calculate_correlation(
    file1="data_files/gdp/gdp_data.csv",
    file2="data_files/hpi/hpi_data.csv",
    column1="Value_x",
    column2="Value_y",
    x_axis_label="Date"
)
slope, y_intercept = calculate_line(
    file1="data_files/gdp/gdp_data.csv",
    file2="data_files/hpi/hpi_data.csv",
    x_axis_label="Date"
)
print("Slope:", slope)
print("Y-Intercept:", y_intercept)
print("Correlation(r): ", r)
print("Coefficient of Determination(R^2): ", r_squared)
max_corr_lag, max_corr = graph_lag(
    csv_file1="data_files/gdp/gdp_data.csv",
    csv_file2="data_files/hpi/hpi_data.csv",
    title="Lag Between CPI and HPI"
)
print("Max Lag Correlation: ", max_corr_lag)
print("Max Correlation", max_corr)

# S&P500 and HPI #
print("\n---------------- S&P500 and HPI ----------------")
graph(
    file1="data_files/sp500/sp500_data.csv",
    file2="data_files/hpi/hpi_data.csv",
    subsample_rate=10,  # Subsample every 10th value
    x_label_rotation=45,  # Rotate x-axis labels by 45 degrees
    y_label_rotation=0,  # No rotation for y-axis labels
    y_tick_interval=50,  # Y-ticks every 50 units
    graph_title="S&P500 Vs. HPI",  # Title of the graph
    x_axis_label="Date",  # Label for x-axis
    y_axis_label="Value",  # Label for y-axis
    line1_name="S&P500",
    line2_name="HPI"
)
r, r_squared = calculate_correlation(
    file1="data_files/sp500/sp500_data.csv",
    file2="data_files/hpi/hpi_data.csv",
    column1="Value_x",
    column2="Value_y",
    x_axis_label="Date"
)
slope, y_intercept = calculate_line(
    file1="data_files/sp500/sp500_data.csv",
    file2="data_files/hpi/hpi_data.csv",
    x_axis_label="Date"
)
print("Slope:", slope)
print("Y-Intercept:", y_intercept)
print("Correlation(r): ", r)
print("Coefficient of Determination(R^2): ", r_squared)
max_corr_lag, max_corr = graph_lag(
    csv_file1="data_files/sp500/sp500_data.csv",
    csv_file2="data_files/hpi/hpi_data.csv",
    title="Lag Between CPI and HPI"
)
print("Max Lag Correlation: ", max_corr_lag)
print("Max Correlation", max_corr)

####### CODE BY OM PATEL 12/24/2024 DO NOT USE WITHOUT PERMISISON #######