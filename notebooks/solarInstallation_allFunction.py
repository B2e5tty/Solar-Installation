import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import boxcox
from IPython.display import display, HTML
from matplotlib import cm


# function for the summarty statistics
def summary_statistice(df,region):
    display(HTML(f'<span style="font-size: 20px;">{region}</span>'))
    df_column = df.iloc[:,1:19].columns.tolist()
    df_min = df.iloc[:,1:19].min().tolist()
    df_list = list(zip(df_column,df_min))
    print(f"Minimum:\n {df_list}")
    
    print('\n')
    
    df_column = df.iloc[:,1:19].columns.tolist()
    df_max = df.iloc[:,1:19].max().tolist()
    df_list = list(zip(df_column,df_max))
    print(f"Maximum:\n {df_list}")
    
    print('\n')
    
    df_column = df.iloc[:,1:19].columns.tolist()
    df_mean = df.iloc[:,1:19].mean().round(2).tolist()
    df_list = list(zip(df_column,df_mean))
    print(f"Mean:\n {df_list}")
    
    print('\n')
    
    df_column = df.iloc[:,1:19].columns.tolist()
    df_median = df.iloc[:,1:19].median().tolist()
    df_list = list(zip(df_column,df_median))
    print(f"Median:\n {df_list}")
    
    print('\n')
    
    df_column = df.iloc[:,1:19].columns.tolist()
    df_std = df.iloc[:,1:19].std().round(2).tolist()
    df_list = list(zip(df_column,df_std))
    print(f"Standard deviation:\n {df_list}")
    
    print('\n')
    
    df_column = df.iloc[:,1:19].columns.tolist()
    df_25th = df.iloc[:,1:19].quantile(0.25).tolist()
    df_list = list(zip(df_column,df_25th))
    print(f"25th Percentile:\n {df_list}")
    
    print('\n')
    
    df_column = df.iloc[:,1:19].columns.tolist()
    df_50th = df.iloc[:,1:19].quantile(0.5).tolist()
    df_list = list(zip(df_column,df_50th))
    print(f"50th Percentile:\n {df_list}")
    
    print('\n')
    
    df_column = df.iloc[:,1:19].columns.tolist()
    df_75th = df.iloc[:,1:19].quantile(0.75).tolist()
    df_list = list(zip(df_column,df_75th))
    print(f"75th Percentile:\n {df_list}")
    
# functions for data quality check
def check_negative_values(df,region):
    display(HTML(f'<span style="font-size: 20px;">{region}</span>'))
    len_negative_ghi = len(df[df['GHI'] < 0])
    len_negative_dni = len(df[df['DNI'] < 0])
    len_negative_dhi = len(df[df['DHI'] < 0])
    
    if (len_negative_ghi != 0) & (len_negative_dni != 0) & (len_negative_dhi != 0):
        print("All the three columns have negative values")
        
    elif (len_negative_ghi != 0) & (len_negative_dni == 0) & (len_negative_dhi == 0):
        print("Column GHI have negative values")
        
    elif (len_negative_ghi == 0) & (len_negative_dni != 0) & (len_negative_dhi == 0):
        print("Column DNI have negative values")
        
    elif (len_negative_ghi == 0) & (len_negative_dni == 0) & (len_negative_dhi != 0):
        print("Column DHI have negative values")
        
    else:
        print("All the three columns don't have negative values")
        

# using boxplot to check for outliers
def check_outliers(df,region):
    display(HTML(f'<span style="font-size: 20px;">{region}</span>'))
    sns.boxplot(data=df[['GHI','DNI','DHI','ModA','ModB','WS','WSgust']])
    plt.show()


# check further outliers
def histo_gram(df,col,region):
    display(HTML(f'<span style="font-size: 20px;">{region}</span>'))
    plt.figure(figsize=(5,3))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(col)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    df_skewness = skew(df[col])
    print(f"Skewness: {df_skewness}")    
    
    
# functions for data cleaning
# changing negative values to zero
def negative_to_zero(df):
    df['GHI'] = df['GHI'].apply(lambda x: max(x,0))
    df['DNI'] = df['GHI'].apply(lambda x: max(x,0))
    df['DHI'] = df['GHI'].apply(lambda x: max(x,0))    

# removing outliers
def removing_outlier(df,col,upper_limit):
    df.loc[df[col] > upper_limit, col] = df[col].median()
    
# function for time series analysis and evaluating the impact of cleaning
def month_hour(df):
    df['Month'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.to_period('M')
    df['Hour'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.hour
    
def plot_timeSeries(df,region):
    # call the above function to create Month and Hour column
    month_hour(df)
    monthly_data = df.groupby('Month').agg({'GHI': 'mean', 'DNI': 'mean', 'DHI': 'mean', 'Tamb': 'mean'}).reset_index()  
    hour_data = df.groupby('Hour').agg({'GHI': 'mean', 'DNI': 'mean', 'DHI': 'mean', 'Tamb': 'mean'}).reset_index()  
    
    fig, axes = plt.subplots(2, 4, figsize=(30,15))
    column = ['GHI','DNI','DHI','Tamb']
    colors = ['blue', 'orange', 'green', 'red']
    titles = ['GHI', 'DNI', 'DHI', 'Tamb']
    
    # monthly trend
    
    for i, col in enumerate(column):
        axes[0,i].plot(monthly_data['Month'].astype(str), monthly_data[col], label=col, color = colors[i])
        axes[0,i].set_title(f'Monthly {region} {titles[i]} Trend')
        axes[0,i].set_ylabel(col)
        plt.xticks(rotation=90)
    
    # hourly trend
    for i, col in enumerate(column):
        axes[1,i].plot(hour_data['Hour'].astype(str), hour_data[col], label=col, color = colors[i])
        axes[1,i].set_title(f'Hourly {region} {titles[i]} Trend')
        axes[1,i].set_ylabel(col)
        plt.xticks(rotation=90)    
        
# Evaluating Impact
def impact_of_cleaning(df,region):
    display(HTML(f'<span style="font-size: 20px;">{region}</span>'))
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    cleaning_events = df[df['Cleaning'] == 1]['Timestamp']

    pre_cleaning_modA = []
    post_cleaning_modA = []
    pre_cleaning_modB = []
    post_cleaning_modB = []

    time_window = pd.Timedelta(days=1)
    
    for event in cleaning_events:
        # Filter data within the time window before and after the cleaning event
        pre_cleaning_data = df[(df['Timestamp'] >= event - time_window) & (df['Timestamp'] < event)]
        post_cleaning_data = df[(df['Timestamp'] > event) & (df['Timestamp'] <= event + time_window)]

        # Calculate the mean sensor readings before and after cleaning
        pre_cleaning_modA.append(pre_cleaning_data['ModA'].mean())
        post_cleaning_modA.append(post_cleaning_data['ModA'].mean())
        pre_cleaning_modB.append(pre_cleaning_data['ModB'].mean())
        post_cleaning_modB.append(post_cleaning_data['ModB'].mean())

    # Convert results to DataFrame for easier plotting
    impact_df = pd.DataFrame({
        'Cleaning_Event': cleaning_events,
        'Pre_Cleaning_ModA': pre_cleaning_modA,
        'Post_Cleaning_ModA': post_cleaning_modA,
        'Pre_Cleaning_ModB': pre_cleaning_modB,
        'Post_Cleaning_ModB': post_cleaning_modB
    })

    # Calculate the improvement or change
    impact_df['ModA_Change'] = impact_df['Post_Cleaning_ModA'] - impact_df['Pre_Cleaning_ModA']
    impact_df['ModB_Change'] = impact_df['Post_Cleaning_ModB'] - impact_df['Pre_Cleaning_ModB']

    # Visualize the impact
    plt.figure(figsize=(14, 7))

    # # Plot ModA
    plt.subplot(2, 1, 1)
    plt.plot(impact_df['Cleaning_Event'], impact_df['Post_Cleaning_ModA'], color='blue', label='Post-Cleaning ModA')
    plt.xlabel('Time')
    plt.ylabel('ModA Readings')
    plt.title('Impact of Cleaning on ModA')
    plt.legend()

    # Plot ModB
    plt.subplot(2, 1, 2)
    plt.plot(impact_df['Cleaning_Event'], impact_df['Post_Cleaning_ModB'], color='green', label='Post-Cleaning ModB')
    plt.xlabel('Time')
    plt.ylabel('ModB Readings')
    plt.title('Impact of Cleaning on ModB')
    plt.legend()

    plt.tight_layout()
    plt.show()


# function for correlation analysis and relation betweeen wind and solar radiance
def correlation_heatmap(df,columns,task,region):
    display(HTML(f'<span style="font-size: 20px;">{region}</span>'))
    correlation_matrix = df[columns].corr()

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title(task)
    plt.show()

# function for wind analysis
def polar_plot_windAnalysis(df,region):
    display(HTML(f'<span style="font-size: 20px;">{region}</span>'))
    df['wd_rad'] = np.deg2rad(df['WD'])
    
    plt.figure(figsize=(10,8))
    ax = plt.subplot(111, polar=True)
    
    sc = ax.scatter(df['wd_rad'], df['WS'], c=df['WSgust'], s=df['WSstdev']*100, cmap=cm.plasma, alpha=0.75)

    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Wind Gust Speed (m/s)')

    # Set the title and labels
    ax.set_title('Wind Speed, Gusts, and Direction Distribution', va='bottom')
    ax.set_theta_direction(-1)  # Set the direction of the theta (clockwise)
    ax.set_theta_offset(np.pi/2)  # Set 0 degrees (North) to the top

    # Show the plot
    plt.show()


# function for temperature analysis
def polar_plot_windAnalysis(df,region):
    display(HTML(f'<span style="font-size: 20px;">{region}</span>'))
    df['wd_rad'] = np.deg2rad(df['WD'])
    
    plt.figure(figsize=(10,8))
    ax = plt.subplot(111, polar=True)
    
    sc = ax.scatter(df['wd_rad'], df['WS'], c=df['WSgust'], s=df['WSstdev']*100, cmap=cm.plasma, alpha=0.75)

    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Wind Gust Speed (m/s)')

    # Set the title and labels
    ax.set_title('Wind Speed, Gusts, and Direction Distribution', va='bottom')
    ax.set_theta_direction(-1)  # Set the direction of the theta (clockwise)
    ax.set_theta_offset(np.pi/2)  # Set 0 degrees (North) to the top

    # Show the plot
    plt.show()

# function for histogram plot 
def histogram(df,region):
    display(HTML(f'<span style="font-size: 20px;">{region}</span>'))
    columns = ['GHI', 'DNI', 'DHI', 'WS', 'Tamb', 'TModA', 'TModB']
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))
    axs = axs.flatten()
    
    for i, col in enumerate(columns):
        axs[i].hist(df[col], bins=30, color='skyblue', edgecolor='black')
        axs[i].set_title(f'Histogram of {col}')
        axs[i].set_xlabel(col)
        axs[i].set_ylabel('Frequency')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()      
    
# function for z-score analysis
def z_score(df,region):
    mean = df.select_dtypes(include=[np.number]).mean()
    std = df.select_dtypes(include=[np.number]).std()
    
    z_score = (df.select_dtypes(include=[np.number]) - mean) / std
    flags = (z_score.abs() > mean)
    
    print("Columns with data points significantly different from the mean")
    for col in flags.columns:
        if flags[col].any() == True:
            print(f"{col}")
          
    print('\n')
    
# function for bubble chart
def bubble_charts(df,region):
    display(HTML(f'<span style="font-size: 20px;">{region}</span>'))
    plt.figure(figsize=(12, 8))
    bubble = plt.scatter(df['GHI'], df['Tamb'], s=df['RH'], c=df['WS'], cmap='viridis', alpha=0.6, edgecolors="w", linewidth=2)

    plt.colorbar(bubble, label='Wind Speed (WS)')
    plt.xlabel('Global Horizontal Irradiance (GHI)')
    plt.ylabel('Ambient Temperature (Tamb)')
    plt.title('Bubble Chart: GHI vs. Tamb vs. WS (Bubble Size: RH)')
    plt.show()
