#!/usr/bin/env python
# coding: utf-8

# bclc
# # Performance Framework

# ## i) Libraries & Dependencies

# In[1]:


#Dependencies
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 1000)

#Visualization
import matplotlib.pyplot as plt
import plotly as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots 

#Data processing
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy.stats import ttest_rel
from scipy import stats


# ## ii) Data Wrangling

# In[2]:


df = pd.read_csv('Online_Sales.csv', index_col=0).reset_index()

df.shape


# In[3]:


df.info()


# In[4]:


df


# In[5]:


df.isna().sum()


# ## iii) Feature Engineering (Adding columns)

# In[6]:


date_format='%m/%d/%Y'
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'],format=date_format)


# In[7]:


df['Sales'] = df['Quantity'] * df['Avg_Price']


# In[8]:


df['Year'] = df['Transaction_Date'].dt.year


# In[9]:


df['Month'] = df['Transaction_Date'].dt.month


# In[10]:


df['Coupon_Number'] = np.where(df['Coupon_Status'] == 'Used', 1, 0)


# In[11]:


df_product_category = df['Product_Category'].value_counts()
df_product_category


# In[12]:


# Create a new column 'Product_Category_Grouped' to combine categories
df['Product_Category_Grouped'] = df['Product_Category'].apply(lambda x: x if x in ['Apparel', 'Nest-USA'] else 'Other')

# Create a pivot table to transform the data
pivot_df = pd.pivot_table(df, 
                          index=['CustomerID', 'Transaction_ID', 'Transaction_Date','Month','Year'],
                          values=['Quantity', 'Avg_Price', 'Sales', 'Coupon_Number'],
                          columns='Product_Category_Grouped',
                          aggfunc={
                              'Quantity': 'sum',
                              'Avg_Price': 'mean',
                              'Sales': 'sum',
                              'Coupon_Number': 'sum'
                          },
                          fill_value=0)

# Flatten the multi-level columns
pivot_df.columns = ['_'.join(map(str, col)).strip('_') for col in pivot_df.columns.values]

# Reset index to make it flat
pivot_df.reset_index(inplace=True)

pivot_df


# In[13]:


pivot_df['Quantity_All'] = pivot_df['Quantity_Apparel'] + pivot_df['Quantity_Nest-USA'] + pivot_df['Quantity_Other']
pivot_df['Sales_All'] = pivot_df['Sales_Apparel'] + pivot_df['Sales_Nest-USA'] + pivot_df['Sales_Other']
pivot_df['Coupon_Number_All'] = pivot_df['Coupon_Number_Apparel'] + pivot_df['Coupon_Number_Nest-USA'] + pivot_df['Coupon_Number_Other']
pivot_df['Avg_Price_All'] = pivot_df['Sales_All'] / pivot_df['Quantity_All']


# In[14]:


pivot_df.info()


# In[15]:


agg_functions = {
    'Transaction_ID': 'nunique',
    'Quantity_All': 'sum',
    'Sales_All': 'sum',
    'Coupon_Number_All': 'sum',
    'Quantity_Apparel': 'sum',
    'Sales_Apparel': 'sum',
    'Coupon_Number_Apparel': 'sum',
    'Quantity_Nest-USA': 'sum',
    'Sales_Nest-USA': 'sum',
    'Coupon_Number_Nest-USA': 'sum',
    'Quantity_Other': 'sum',
    'Sales_Other': 'sum',
    'Coupon_Number_Other': 'sum',
}


# In[16]:


agg_columns = ['Transaction_ID', 'Quantity_All', 'Sales_All', 'Coupon_Number_All', 'Quantity_Apparel', 'Sales_Apparel', 'Coupon_Number_Apparel','Quantity_Nest-USA', 'Sales_Nest-USA', 'Coupon_Number_Nest-USA','Quantity_Other', 'Sales_Other', 'Coupon_Number_Other']


# In[17]:


daily_df = pivot_df.groupby(['CustomerID','Transaction_Date','Month','Year']).agg(agg_functions).reset_index()


# In[18]:


daily_df.rename(columns={
    'Transaction_ID': 'Transactions',
}, inplace=True)


# In[19]:


daily_df


# # 1. Data Exploration

# In[20]:


dfexp0=daily_df.copy()


# In[21]:


dfexp1=dfexp0.groupby(['Year'])['Sales_All'].sum().reset_index()
dfexp1


# In[22]:


dfexp2=dfexp0.groupby(['Month'])['Sales_All'].sum().reset_index()
dfexp2


# In[23]:


# Initialize figure
figexp2 = go.Figure()


# Add traces
figexp2 .add_trace(go.Bar(
        x=dfexp2['Month'],
        y=dfexp2['Sales_All'],
        marker_color='#F58518'
        ))

# Update xaxis properties
figexp2 .update_xaxes(title_text='Month',
                 tick0=0,
                 dtick=1,
                # autorange='reversed' 
                )


# Update yaxis properties
figexp2 .update_yaxes(title_text='Sales',
                tickformat=",",
                tickprefix="$")


# Update general layout
figexp2 .update_layout(title_text='Monthly Sales',
                 showlegend=False)

figexp2 .show()


# In[24]:


dfexp3=dfexp0.groupby(['Transaction_Date','Month'])['Sales_All'].sum().reset_index()
dfexp3


# In[25]:


# Create a line chart
figexp3 = go.Figure()

# Add a trace for the line chart with area
figexp3.add_trace(go.Scatter(x=dfexp3['Transaction_Date'], y=dfexp3['Sales_All'],
                             mode='lines',
                             line=dict(color='#F58518'),
                             fill='tozeroy',  # Fill area below the line for positive values
                             fillcolor='rgba(75, 120, 168, 0.2)',  # Fill color for positive values
                             name='FY24'))

# Update layout
figexp3.update_layout(title='Monthly Sales - Time Series Line Chart',
                      xaxis_title='Date',
                      yaxis_title='Sales',
                      template='simple_white',  # Use the 'simple_white' template
                      xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black', range=[dfexp3['Transaction_Date'].min(), dfexp3['Transaction_Date'].max()]),  # Show x-axis at zero
                      yaxis=dict(gridcolor='lightgrey'),  # Set the y-axis grid color
                      xaxis_showgrid=True, yaxis_showgrid=True,  # Show the gridlines
                      plot_bgcolor='white',  # Set background color to white
                      paper_bgcolor='white',  # Set paper color to white
                      font=dict(color='black'),  # Set font color to black
                      )

# Show the plot
figexp3.show()


# # 2. Functions Definition

# ### Period Pre vs Post

# In[26]:


def pre_post_period(pre_ini_date, pre_end_date, post_ini_date, post_end_date, dfa):
    
    # Define the date ranges
    pre_ini_date = pd.to_datetime(pre_ini_date)
    pre_end_date = pd.to_datetime(pre_end_date)

    # Define the date ranges
    post_ini_date = pd.to_datetime(post_ini_date)
    post_end_date = pd.to_datetime(post_end_date)
    
    # Create the 'Period' column based on conditions
    dfa['Period'] = np.select(
        [
            (dfa['Transaction_Date'] >= pre_ini_date) & (dfa['Transaction_Date'] <= pre_end_date),
            (dfa['Transaction_Date'] >= post_ini_date) & (dfa['Transaction_Date'] <= post_end_date)
        ],
        ['Pre', 'Post'],
        default='Out'
    )
    
    return dfa


# In[27]:


# Define the date ranges
pre_ini_date = '2019-02-01'
pre_end_date = '2019-05-30'

# Define the date ranges
post_ini_date = '2019-07-01'
post_end_date = '2019-11-30'


# In[28]:


df_with_period = pre_post_period(pre_ini_date, pre_end_date, post_ini_date, post_end_date, daily_df)


# In[29]:


df_with_period.info()


# ### Promo Segmentation

# In[30]:


def promo_segmentation(dfa):
    dfa['Promo_Segment'] = 'Other'  # Default to 'Other'
    
    #Grouping for Promo_segmentation
    grouped_df = pd.pivot_table(data=dfa,
                                index='CustomerID',
                                columns='Period',
                                values=['Coupon_Number_All'],
                                aggfunc={'Coupon_Number_All':np.sum})
    grouped_df=grouped_df.fillna(0)    
    grouped_df.columns = ['{}_{}'.format(level2, level1) for level1, level2 in grouped_df.columns]
    grouped_df= grouped_df.reset_index()
    grouped_df
    
    #Promo_Segmentation

    # Segment 1
    dfa.loc[(dfa['CustomerID'].isin(grouped_df[(grouped_df['Pre_Coupon_Number_All'] != 0) & (grouped_df['Post_Coupon_Number_All'] != 0)]['CustomerID'])), 'Promo_Segment'] = 'Promo'

    # Segment 2
    dfa.loc[(dfa['CustomerID'].isin(grouped_df[(grouped_df['Pre_Coupon_Number_All'] != 0) & (grouped_df['Post_Coupon_Number_All'] == 0)]['CustomerID'])), 'Promo_Segment'] = 'PromoPre'

    # Segment 3
    dfa.loc[(dfa['CustomerID'].isin(grouped_df[(grouped_df['Pre_Coupon_Number_All'] == 0) & (grouped_df['Post_Coupon_Number_All'] != 0)]['CustomerID'])), 'Promo_Segment'] = 'PromoPost'

    # Segment 4
    dfa.loc[(dfa['CustomerID'].isin(grouped_df[(grouped_df['Pre_Coupon_Number_All'] == 0) & (grouped_df['Post_Coupon_Number_All'] == 0)]['CustomerID'])), 'Promo_Segment'] = 'NoPromo'
    
    
    return dfa


# In[31]:


df_with_segmentation = promo_segmentation(df_with_period)


# In[32]:


df_with_segmentation


# In[33]:


segmentation = df_with_segmentation.groupby(['Period','Promo_Segment'])['Sales_All'].sum()


# In[34]:


segmentation


# ### Hypothesis testing

# In[35]:


def weighted_ttest_1samp(data, hypothesized_mean, weights):
    # Calculate the weighted mean
    weighted_mean = np.average(data, weights=weights)
    
    # Calculate the weighted standard deviation
    weighted_std = np.sqrt(np.average((data - weighted_mean)**2, weights=weights))
    
    # Calculate the t-statistic
    t_stat = (weighted_mean - hypothesized_mean) / (weighted_std / np.sqrt(len(data)))
    
    # Calculate the degrees of freedom
    degrees_of_freedom = len(data) - 1
    
    # Calculate the p-value using the t-distribution
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=degrees_of_freedom))
    
    return t_stat, p_value


# In[36]:


def hypothesis_testing_customer(dfa,pre_ini_date,pre_end_date,post_ini_date,post_end_date):
    # I will be using One-Sample t_Test so we can use a different number of customers for Pre and Post period
    # Ho: Promo does nothing vs Ha: Promo does something
    ################################### Per Player ##########################################
    
    # Define the date ranges
    pre_ini_date = pd.to_datetime(pre_ini_date)
    pre_end_date = pd.to_datetime(pre_end_date)

    # Define the date ranges
    post_ini_date = pd.to_datetime(post_ini_date)
    post_end_date = pd.to_datetime(post_end_date)
    
    
    ### Feature Engineering: Frequency and Recency, and New Product ###
    
    # Calculate the difference in days
    pre_days_length = (pre_end_date - pre_ini_date).days+1
    post_days_length = (post_end_date - post_ini_date).days+1
    
    
    
    # First, let's calculate the mean aggregated columns

    agg_columns = ['Transactions', 'Quantity_All', 'Sales_All', 'Coupon_Number_All', 'Quantity_Apparel', 'Sales_Apparel', 'Coupon_Number_Apparel','Quantity_Nest-USA', 'Sales_Nest-USA', 'Coupon_Number_Nest-USA','Quantity_Other', 'Sales_Other', 'Coupon_Number_Other']
    dfpre_player =  dfa[(dfa['Period'] == 'Pre')].groupby(['Promo_Segment','Period','CustomerID'])[agg_columns].mean().reset_index()
    dfpost_player =  dfa[(dfa['Period'] == 'Post')].groupby(['Promo_Segment','Period','CustomerID'])[agg_columns].mean().reset_index()
  
    ### Pre
    
    # Filter data for the Pre and Post periods
    pre_data = dfa[dfa['Period'] == 'Pre']

    # Then, calculate Frequency and Recency
    freq_rec_df = pre_data.groupby(['CustomerID']).agg({'Transaction_Date': ['count', 'max']}).reset_index()
    freq_rec_df.columns = ['CustomerID', 'Frequency', 'MaxDate']

    # Calculate Frequency as a %
    days_count = pre_days_length
    freq_rec_df['Frequency%'] = freq_rec_df['Frequency']/days_count

    # Calculate the Recency column
    max_date = pre_end_date
    freq_rec_df['Recency'] = (max_date - freq_rec_df['MaxDate']).dt.days

    # Merge the two DataFrames
    dfpre_player = pd.merge(dfpre_player, freq_rec_df[['CustomerID', 'Frequency%', 'Recency']], on='CustomerID')

    # Now, dfpre contains Frequency and Recency
    
    
    ### Post
    
    # Filter data for the Pre and Post periods
    post_data = dfa[dfa['Period'] == 'Post']

    # Then, calculate Frequency and Recency
    freq_rec_df = post_data.groupby(['CustomerID']).agg({'Transaction_Date': ['count']}).reset_index()
    freq_rec_df.columns = ['CustomerID', 'Frequency']

    # Calculate Frequency as a %
    days_count = post_days_length
    freq_rec_df['Frequency%'] = freq_rec_df['Frequency']/days_count

    # Merge the two DataFrames
    dfpost_player = pd.merge(dfpost_player, freq_rec_df[['CustomerID', 'Frequency%']], on='CustomerID')

    # Now, dfpost contains Frequency
    
    #For Post_Recency calculation only
    
    # Filter the DataFrame
    start_date = post_end_date - pd.Timedelta(days=pre_days_length)
    post_data = dfa[(dfa['Transaction_Date'] >= start_date) & (dfa['Transaction_Date'] <= post_end_date)]

    # Then, calculate Frequency and Recency
    freq_rec_df = post_data.groupby(['CustomerID']).agg({'Transaction_Date': ['max']}).reset_index()
    freq_rec_df.columns = ['CustomerID', 'MaxDate']

    # Calculate the Recency column
    max_date = post_end_date
    freq_rec_df['Recency'] = (max_date - freq_rec_df['MaxDate']).dt.days

    # Merge the two DataFrames
    dfpost_player = pd.merge(dfpost_player, freq_rec_df[['CustomerID', 'Recency']], on='CustomerID')

    # Now, dfpost contains aggregated columns, Frequency, and Recency
    
    
    ### Hypothesis Testing ###
    
    
    # Filter data for the Pre and Post periods
    pre_data = dfpre_player
    post_data = dfpost_player

    agg_columns = ['Transactions', 'Quantity_All', 'Sales_All', 'Coupon_Number_All', 'Quantity_Apparel', 'Sales_Apparel', 'Coupon_Number_Apparel','Quantity_Nest-USA', 'Sales_Nest-USA', 'Coupon_Number_Nest-USA','Quantity_Other', 'Sales_Other', 'Coupon_Number_Other', 'Frequency%', 'Recency']
    

    # Calculate weighted average for every metric except Recency which is average
    pre_metrics = pre_data.groupby('Promo_Segment').apply(lambda x: pd.Series({
        col: (x[col] * x['Frequency%']).sum() / x['Frequency%'].sum() if col in agg_columns and col != 'Recency' and col != 'Frequency%' else x[col].mean()
        for col in agg_columns
    })).reset_index()

    # PlayerCount: count the occurrences of each segment and store it in a DataFrame
    pre_segment_counts = pre_data['Promo_Segment'].value_counts().reset_index()
    pre_segment_counts.columns = ['Promo_Segment', 'CustomerCount']
    pre_metrics = pd.merge(pre_metrics,pre_segment_counts,on='Promo_Segment',how='left')

    # Calculate weighted average for every metric except Recency which is average
    post_metrics = post_data.groupby('Promo_Segment').apply(lambda x: pd.Series({
        col: (x[col] * x['Frequency%']).sum() / x['Frequency%'].sum() if col in agg_columns and col != 'Recency' and col != 'Frequency%' else x[col].mean()
        for col in agg_columns
    })).reset_index()

    # PlayerCount: count the occurrences of each segment and store it in a DataFrame
    post_segment_counts = post_data['Promo_Segment'].value_counts().reset_index()
    post_segment_counts.columns = ['Promo_Segment', 'CustomerCount']
    post_metrics = pd.merge(post_metrics,post_segment_counts,on='Promo_Segment',how='left')
    
    # Perform one-sample t-test for each metric
    alpha = 0.05
    results = {'Promo_Segment': [], 'Metrics': [], 'Pre': [], 'Post': [], 'Var': [], 'Var%': [], 'Sign': []}

    columns = ['Transactions', 'Quantity_All', 'Sales_All', 'Coupon_Number_All', 'Quantity_Apparel', 'Sales_Apparel', 'Coupon_Number_Apparel','Quantity_Nest-USA', 'Sales_Nest-USA', 'Coupon_Number_Nest-USA','Quantity_Other', 'Sales_Other', 'Coupon_Number_Other', 'Frequency%', 'Recency','CustomerCount']

    for metric in columns:
        for segment in dfa['Promo_Segment'].unique():

            if metric == 'CustomerCount':
                 #Calculate the difference
                difference = post_metrics.loc[post_metrics['Promo_Segment'] == segment, metric].values[0] - pre_metrics.loc[pre_metrics['Promo_Segment'] == segment, metric].values[0]

                # Store results
                results['Promo_Segment'].append(segment)
                results['Metrics'].append(f'{metric}')
                results['Pre'].append(pre_metrics.loc[pre_metrics['Promo_Segment'] == segment, metric].values[0])
                results['Post'].append(post_metrics.loc[post_metrics['Promo_Segment'] == segment, metric].values[0])
                results['Var'].append(difference)
                results['Sign'].append('NA')

                continue

            if metric == 'Frequency%' or metric == 'Recency':
                post_values = post_data[post_data['Promo_Segment'] == segment][metric]
                pre_weighted_avg = pre_metrics.loc[pre_metrics['Promo_Segment'] == segment, metric].values[0]
                t_stat, p_value = stats.ttest_1samp(post_values, pre_weighted_avg)
            else:
                # Extract data for the segment
                post_values = post_data[post_data['Promo_Segment'] == segment][metric]
                weights = post_data[post_data['Promo_Segment'] == segment]['Frequency%']

                # Calculate the weighted average for the metric from pre_metrics
                pre_weighted_avg = pre_metrics.loc[pre_metrics['Promo_Segment'] == segment, metric].values[0]

                # Perform one-sample t-test using the weighted average
                t_stat, p_value = weighted_ttest_1samp(post_values, pre_weighted_avg, weights)


            # Calculate the difference
            difference = post_metrics.loc[post_metrics['Promo_Segment'] == segment, metric].values[0] - pre_metrics.loc[pre_metrics['Promo_Segment'] == segment, metric].values[0]
            difference_percent = difference/pre_metrics.loc[pre_metrics['Promo_Segment'] == segment, metric].values[0]
            # Store results
            results['Promo_Segment'].append(segment)
            results['Metrics'].append(f'{metric}')
            results['Pre'].append(pre_metrics.loc[pre_metrics['Promo_Segment'] == segment, metric].values[0])
            results['Post'].append(post_metrics.loc[post_metrics['Promo_Segment'] == segment, metric].values[0])
            results['Var'].append(difference)
            results['Sign'].append('Yes' if p_value < alpha else 'No')


    length = len(results['Promo_Segment'])        
    # Iterate through rows
    for i in range(length):
        pre_val = results['Pre'][i]
        post_val = results['Post'][i]

        # Check if Pre or Post is zero
        if pre_val == 0 or post_val == 0:
            results['Var%'].append('NA')
        else:
            # Calculate Var%
            var_percentage = (post_val / pre_val - 1) * 100
            results['Var%'].append('{:.2f}%'.format(var_percentage))

    # Create the result dataframe
    player_df = pd.DataFrame(results)

    # Sort the result dataframe by 'Promo_Segment'
    player_df = player_df.sort_values(by='Promo_Segment')
    
    return player_df,dfpre_player,dfpost_player


# In[37]:


player_df,dfpre_player,dfpost_player = hypothesis_testing_customer(df_with_segmentation,pre_ini_date,pre_end_date,post_ini_date,post_end_date)


# In[38]:


player_df


# In[39]:


# Function to calculate days count
def calculate_days_count(group):
    return (group['Transaction_Date'].max() - group['Transaction_Date'].min()).days+1


# In[40]:


def hypothesis_testing_days(dfa,pre_ini_date,pre_end_date,post_ini_date,post_end_date):
    # I will be using One-Sample t_Test so we can use a different number of players for Pre and Post period
    # Ho: Promo does nothing vs Ha: Promo does something
    ################################### Per Day ##########################################
    
    # Define the date ranges
    pre_ini_date = pd.to_datetime(pre_ini_date)
    pre_end_date = pd.to_datetime(pre_end_date)

    # Define the date ranges
    post_ini_date = pd.to_datetime(post_ini_date)
    post_end_date = pd.to_datetime(post_end_date)
    
    
    ### Feature Engineering: Frequency and Recency, and New Product ###
    
    # First, let's calculate the mean aggregated columns
    agg_columns = ['Transactions', 'Quantity_All', 'Sales_All', 'Coupon_Number_All', 'Quantity_Apparel', 'Sales_Apparel', 'Coupon_Number_Apparel','Quantity_Nest-USA', 'Sales_Nest-USA', 'Coupon_Number_Nest-USA','Quantity_Other', 'Sales_Other', 'Coupon_Number_Other']
    dfpre_day =  dfa[dfa['Period'] == 'Pre'].groupby(['Promo_Segment','Period','Transaction_Date'])[agg_columns].mean().reset_index()
    dfpost_day =  dfa[dfa['Period'] == 'Post'].groupby(['Promo_Segment','Period','Transaction_Date'])[agg_columns].mean().reset_index()

    active_customers_pre = dfa[dfa['Period'] == 'Pre'].groupby(['Promo_Segment', 'Transaction_Date'])['CustomerID'].count().reset_index(name='ActiveCustomers')
    active_customers_post = dfa[dfa['Period'] == 'Post'].groupby(['Promo_Segment', 'Transaction_Date'])['CustomerID'].count().reset_index(name='ActiveCustomers')

    # Calculate ActivePlayer as a %
    total_active_customers_pre = active_customers_pre.groupby('Promo_Segment')['ActiveCustomers'].sum().reset_index(name='TotalActiveCustomers')
    total_active_customers_post = active_customers_post.groupby('Promo_Segment')['ActiveCustomers'].sum().reset_index(name='TotalActiveCustomers')
    
    

    # Step 2: Merge with the aggregated DataFrames
    dfpre_day = pd.merge(dfpre_day, active_customers_pre, on=['Promo_Segment', 'Transaction_Date'], how='left')
    dfpre_day = pd.merge(dfpre_day, total_active_customers_pre, on=['Promo_Segment'], how='left')
    dfpre_day['ActiveCustomers%'] = dfpre_day['ActiveCustomers']/dfpre_day['TotalActiveCustomers']
    dfpre_day = dfpre_day.drop('TotalActiveCustomers', axis=1)
    dfpost_day = pd.merge(dfpost_day, active_customers_post, on=['Promo_Segment', 'Transaction_Date'], how='left')
    dfpost_day = pd.merge(dfpost_day, total_active_customers_post, on=['Promo_Segment'], how='left')
    dfpost_day['ActiveCustomers%'] = dfpost_day['ActiveCustomers']/dfpost_day['TotalActiveCustomers']
    dfpost_day = dfpost_day.drop('TotalActiveCustomers', axis=1)
    
    ########################################################## RECENCY CALC ################################
    # Step 3: Calculate the recency per day per player
    recency_raw_df= dfa.copy()
    
    # Calculate the difference in days
    post_days_length = (post_end_date - post_ini_date).days+1
    pre_days_length = (pre_end_date - pre_ini_date).days+1
    
    # we will calculate recency for both pre and post period based on post duration
    # Calculate the initial day
    
    if post_days_length > pre_days_length:
        
        post_end_date_recency = post_ini_date + timedelta(days=pre_days_length)

        # Create the 'Period' column based on condition
        recency_raw_df['Period'] = np.select(
            [
                (recency_raw_df['Transaction_Date'] >= pre_ini_date) & (recency_raw_df['Transaction_Date'] <= pre_end_date),
                (recency_raw_df['Transaction_Date'] >= post_ini_date) & (recency_raw_df['Transaction_Date'] <= post_end_date_recency)
            ],
            ['Pre', 'Post'],
            default='Out'
        )
        
        recency_pre_df = recency_raw_df[recency_raw_df['Period'] == 'Pre']
        recency_post_df = recency_raw_df[recency_raw_df['Period'] == 'Post']
        
        # Find the minimum and maximum Datetime for each player
        min_max_datetime_post = recency_post_df.groupby('CustomerID')['Transaction_Date'].agg(['min']).reset_index()
        min_max_datetime_post.columns = ['CustomerID', 'MinDatetime']
        min_max_datetime_post['MaxDatetime'] = post_end_date_recency

        min_max_datetime_pre = recency_pre_df.groupby('CustomerID')['Transaction_Date'].agg(['min']).reset_index()
        min_max_datetime_pre.columns = ['CustomerID', 'MinDatetime']
        min_max_datetime_pre['MaxDatetime'] = pre_end_date
        
    else:  
        pre_ini_date_recency = pre_end_date - timedelta(days=post_days_length)

        # Create the 'Period' column based on condition
        recency_raw_df['Period'] = np.select(
            [
                (recency_raw_df['Transaction_Date'] >= pre_ini_date_recency) & (recency_raw_df['Transaction_Date'] <= pre_end_date),
                (recency_raw_df['Transaction_Date'] >= post_ini_date) & (recency_raw_df['Transaction_Date'] <= post_end_date)
            ],
            ['Pre', 'Post'],
            default='Out'
        )
        
        recency_pre_df = recency_raw_df[recency_raw_df['Period'] == 'Pre']
        recency_post_df = recency_raw_df[recency_raw_df['Period'] == 'Post']
    
        # Find the minimum and maximum Datetime for each player
        min_max_datetime_post = recency_post_df.groupby('CustomerID')['Transaction_Date'].agg(['min']).reset_index()
        min_max_datetime_post.columns = ['CustomerID', 'MinDatetime']
        min_max_datetime_post['MaxDatetime'] = post_end_date

        min_max_datetime_pre = recency_pre_df.groupby('CustomerID')['Transaction_Date'].agg(['min']).reset_index()
        min_max_datetime_pre.columns = ['CustomerID', 'MinDatetime']
        min_max_datetime_pre['MaxDatetime'] = pre_end_date

    #Create the entire Player date_range
    post_date_range_list = []
    pre_date_range_list = []

    for row in min_max_datetime_post.itertuples(index=False):
        date_range = pd.date_range(row.MinDatetime, row.MaxDatetime, name='Transaction_Date')
        customer_dates = pd.DataFrame({'CustomerID': [row.CustomerID] * len(date_range), 'Transaction_Date': date_range})
        post_date_range_list.append(customer_dates)

    post_date_range_df = pd.concat(post_date_range_list, ignore_index=True)

    for row in min_max_datetime_pre.itertuples(index=False):
        date_range = pd.date_range(row.MinDatetime, row.MaxDatetime, name='Transaction_Date')
        customer_dates = pd.DataFrame({'CustomerID': [row.CustomerID] * len(date_range), 'Transaction_Date': date_range})
        pre_date_range_list.append(customer_dates)

    pre_date_range_df = pd.concat(pre_date_range_list, ignore_index=True)
    
    
    
    #Merge with other necessary data
    recency_pre_df['TransactionDate2'] = recency_pre_df['Transaction_Date']
    recency_post_df['TransactionDate2'] = recency_post_df['Transaction_Date']
    # Select columns for the left dataframe
    columns_to_merge = ['Promo_Segment','CustomerID', 'Transaction_Date', 'TransactionDate2']
    subset_recency_pre_df = recency_pre_df[columns_to_merge]
    subset_recency_post_df = recency_post_df[columns_to_merge]
    pre_date_range_merged_df = pd.merge(pre_date_range_df, subset_recency_pre_df, on=['CustomerID','Transaction_Date'], how='left')
    post_date_range_merged_df = pd.merge(post_date_range_df, subset_recency_post_df, on=['CustomerID','Transaction_Date'], how='left')

    # Fill forward the values
    pre_date_range_merged_df = pre_date_range_merged_df.sort_values(by=['CustomerID', 'Transaction_Date'])
    pre_date_range_merged_df = pre_date_range_merged_df.fillna(method='ffill')
    post_date_range_merged_df = post_date_range_merged_df.sort_values(by=['CustomerID', 'Transaction_Date'])
    post_date_range_merged_df = post_date_range_merged_df.fillna(method='ffill')

    pre_date_range_merged_df['Recency'] = (pre_date_range_merged_df['Transaction_Date'] - pre_date_range_merged_df['TransactionDate2']).dt.days
    pre_date_range_merged_df['Recency'] = pre_date_range_merged_df['Recency']*-1
    post_date_range_merged_df['Recency'] = (post_date_range_merged_df['Transaction_Date'] - post_date_range_merged_df['TransactionDate2']).dt.days
    post_date_range_merged_df['Recency'] = post_date_range_merged_df['Recency']*-1
    #Grouped by Day, Period, Promo_Segment
    df_recency_pre =  pre_date_range_merged_df.groupby(['Promo_Segment','Transaction_Date'])['Recency'].mean().reset_index()
    df_recency_post =  post_date_range_merged_df.groupby(['Promo_Segment','Transaction_Date'])['Recency'].mean().reset_index()
    
    #Merging with dfpre and dfpost
    dfpre_day = pd.merge(dfpre_day, df_recency_pre, on=['Promo_Segment', 'Transaction_Date'], how='left')
    dfpost_day = pd.merge(dfpost_day, df_recency_post, on=['Promo_Segment', 'Transaction_Date'], how='left')
    
    # Filter data for the Pre and Post periods
    pre_data = dfpre_day
    post_data = dfpost_day

    # First, let's calculate the mean aggregated columns
    agg_columns = ['Transactions', 'Quantity_All', 'Sales_All', 'Coupon_Number_All', 'Quantity_Apparel', 'Sales_Apparel', 'Coupon_Number_Apparel','Quantity_Nest-USA', 'Sales_Nest-USA', 'Coupon_Number_Nest-USA','Quantity_Other', 'Sales_Other', 'Coupon_Number_Other', 'ActiveCustomers', 'Recency']

    # Calculate weighted average for every metric except Recency which is average
    pre_metrics = pre_data.groupby('Promo_Segment').apply(lambda x: pd.Series({
        col: (x[col] * x['ActiveCustomers%']).sum()  if col in agg_columns and col != 'ActiveCustomers' and col != 'ActiveCustomers%' and col != 'Recency' else x[col].mean()
        for col in agg_columns
    })).reset_index()

    # Calculate weighted average for every metric except Recency which is average
    post_metrics = post_data.groupby('Promo_Segment').apply(lambda x: pd.Series({
        col: (x[col] * x['ActiveCustomers%']).sum()  if col in agg_columns and col != 'ActiveCustomers' and col != 'ActiveCustomers%' and col != 'Recency' else x[col].mean()
        for col in agg_columns
    })).reset_index()
    
    # Calculate days count for pre period
    pre_days_count = dfpre_day.groupby('Promo_Segment').apply(calculate_days_count).reset_index()
    pre_days_count.columns = ['Promo_Segment', 'Days_Count']

    # Merge pre_metrics with days count
    pre_metrics = pd.merge(pre_metrics, pre_days_count, on='Promo_Segment')

    # Calculate days count for post period
    post_days_count = dfpost_day.groupby('Promo_Segment').apply(calculate_days_count).reset_index()
    post_days_count.columns = ['Promo_Segment', 'Days_Count']

    # Merge post_metrics with days count
    post_metrics = pd.merge(post_metrics, post_days_count, on='Promo_Segment')
    
    # Perform one-sample t-test for each metric
    alpha = 0.05
    results = {'Promo_Segment': [], 'Metrics': [], 'Pre': [], 'Post': [], 'Var': [], 'Var%': [], 'Sign': []}

    columns = ['Transactions', 'Quantity_All', 'Sales_All', 'Coupon_Number_All', 'Quantity_Apparel', 'Sales_Apparel', 'Coupon_Number_Apparel','Quantity_Nest-USA', 'Sales_Nest-USA', 'Coupon_Number_Nest-USA','Quantity_Other', 'Sales_Other', 'Coupon_Number_Other', 'ActiveCustomers', 'Recency','Days_Count']
    
    for metric in columns:
        for segment in dfa['Promo_Segment'].unique():

            if metric == 'ActiveCustomers' or metric == 'Recency':
                post_values = post_data[post_data['Promo_Segment'] == segment][metric]
                pre_weighted_avg = pre_metrics.loc[pre_metrics['Promo_Segment'] == segment, metric].values[0]
                t_stat, p_value = stats.ttest_1samp(post_values, pre_weighted_avg)
                results['Sign'].append('Yes' if p_value < alpha else 'No')

            elif metric == 'Days_Count':
                results['Sign'].append('NA')

            else:
                # Extract data for the segment
                post_values = post_data[post_data['Promo_Segment'] == segment][metric]
                weights = post_data[post_data['Promo_Segment'] == segment]['ActiveCustomers%']

                # Calculate the weighted average for the metric from pre_metrics
                pre_weighted_avg = pre_metrics.loc[pre_metrics['Promo_Segment'] == segment, metric].values[0]

                # Perform one-sample t-test using the weighted average
                t_stat, p_value = weighted_ttest_1samp(post_values, pre_weighted_avg, weights)

                results['Sign'].append('Yes' if p_value < alpha else 'No')


            # Calculate the difference
            difference = post_metrics.loc[post_metrics['Promo_Segment'] == segment, metric].values[0] - pre_metrics.loc[pre_metrics['Promo_Segment'] == segment, metric].values[0]
            difference_percent = difference/pre_metrics.loc[pre_metrics['Promo_Segment'] == segment, metric].values[0]

            # Store results
            results['Promo_Segment'].append(segment)
            results['Metrics'].append(f'{metric}')
            results['Pre'].append(pre_metrics.loc[pre_metrics['Promo_Segment'] == segment, metric].values[0])
            results['Post'].append(post_metrics.loc[post_metrics['Promo_Segment'] == segment, metric].values[0])
            results['Var'].append(difference)


    length = len(results['Promo_Segment'])        
    # Iterate through rows
    for i in range(length):
        pre_val = results['Pre'][i]
        post_val = results['Post'][i]

        # Check if Pre or Post is zero
        if pre_val == 0 or post_val == 0:
            results['Var%'].append('NA')
        else:
            # Calculate Var%
            var_percentage = (post_val-pre_val) / np.abs(pre_val)  * 100
            results['Var%'].append('{:.2f}%'.format(var_percentage))

    # Create the result dataframe
    day_df = pd.DataFrame(results)

    # Sort the result dataframe by 'Promo_Segment'
    day_df = day_df.sort_values(by='Promo_Segment')
    
    return day_df, dfpre_day, dfpost_day


# In[41]:


day_df, dfpre_day, dfpost_day = hypothesis_testing_days(df_with_segmentation,pre_ini_date,pre_end_date,post_ini_date,post_end_date)


# In[42]:


day_df


# ### Adding Playing and Day metrics together

# In[43]:


def results(player_df,day_df):
    # Extracting the rows from player_df based on the specified condition
    selected_rows = player_df[player_df['Metrics'].isin(['Frequency%', 'CustomerCount'])]

    # Concatenating the two dataframes vertically
    result_df = pd.concat([day_df, selected_rows], ignore_index=True)

    # If you want to sort the resulting dataframe based on a specific column (e.g., 'Metrics')
    result_df = result_df.sort_values(by='Metrics')

    # Resetting the index of the resulting dataframe
    result_df.reset_index(drop=True, inplace=True)  
    
    # Extracting the rows from player_df based on the specified condition
    selected_rows = player_df[player_df['Metrics'].isin(['Frequency%', 'CustomerCount'])]

    # Concatenating the two dataframes vertically
    result_df = pd.concat([day_df, selected_rows], ignore_index=True)

    # Defining the custom sorting order for the 'Metrics' column
    custom_metric_order = [
        'Transactions', 'Sales_All', 'Sales_Apparel', 'Sales_Nest-USA','Sales_Other',  
        'Quantity_All',  'Quantity_Apparel','Quantity_Nest-USA','Quantity_Other', 
        'Coupon_Number_All','Coupon_Number_Apparel','Coupon_Number_Nest-USA','Coupon_Number_Other', 
        'ActiveCustomers', 'Frequency%', 'Recency', 'CustomerCount', 'Days_Count'
    ]

    # Sorting the resulting dataframe by 'Promo_Segment' and then by the custom order of 'Metrics'
    result_df = result_df.sort_values(by=['Promo_Segment', 'Metrics'], key=lambda x: x.map({metric: i for i, metric in enumerate(custom_metric_order)}))

    # Resetting the index of the resulting dataframe
    result_df.reset_index(drop=True, inplace=True)
    
    result_with_format_df = result_df.copy()

    # Define a formatting function for currency
    format_currency = lambda x: "${:,.0f}".format(x) 

    # Define a formatting function for general
    format_general = lambda x: "{:,.0f}".format(x)

    # Define a formatting function for %
    format_percent = lambda x: "{:.2%}".format(x) 

    # Create a mask for the 'Metrics' column
    is_general = result_with_format_df['Metrics'].isin(['Transactions', 'Quantity_All', 'Coupon_Number_All', 'Quantity_Apparel', 'Coupon_Number_Apparel','Quantity_Nest-USA', 'Coupon_Number_Nest-USA','Quantity_Other','Coupon_Number_Other', 'ActiveCustomers', 'Recency', 'CustomerCount', 'Days_Count' ])
    is_currency = result_with_format_df['Metrics'].isin(['Sales_All','Sales_Apparel','Sales_Nest-USA','Sales_Other'])
    is_percent = result_with_format_df['Metrics'].isin(['Frequency%'])

    # Apply formatting based on the mask
    result_with_format_df.loc[is_currency, ['Pre', 'Post','Var']] = result_with_format_df.loc[is_currency, ['Pre', 'Post','Var']].applymap(format_currency)
    result_with_format_df.loc[is_general, ['Pre', 'Post','Var']] = result_with_format_df.loc[is_general, ['Pre', 'Post','Var']].applymap(format_general)
    result_with_format_df.loc[is_percent, ['Pre', 'Post','Var']] = result_with_format_df.loc[is_percent, ['Pre', 'Post','Var']].applymap(format_percent)
    
    return result_df, result_with_format_df


# In[44]:


result_df, result_with_format_df = results(player_df,day_df)


# In[45]:


result_with_format_df


# In[46]:


def data_for_radar_and_table(result_df, segment) :
    data_df = result_df.copy()
    data_df = data_df[data_df['Promo_Segment'] == segment]
    
    data_with_format_df = data_df.copy()

    # Define a formatting function for currency
    format_currency = lambda x: "${:,.0f}".format(x) 

    # Define a formatting function for general
    format_general = lambda x: "{:,.0f}".format(x)

    # Define a formatting function for %
    format_percent = lambda x: "{:.2%}".format(x) 

    # Create a mask for the 'Metrics' column
    is_general = result_with_format_df['Metrics'].isin(['Transactions', 'Quantity_All', 'Coupon_Number_All', 'Quantity_Apparel', 'Coupon_Number_Apparel','Quantity_Nest-USA', 'Coupon_Number_Nest-USA','Quantity_Other','Coupon_Number_Other', 'ActiveCustomers', 'Recency', 'CustomerCount', 'Days_Count' ])
    is_currency = result_with_format_df['Metrics'].isin(['Sales_All','Sales_Apparel','Sales_Nest-USA','Sales_Other'])
    is_percent = result_with_format_df['Metrics'].isin(['Frequency%'])

    # Apply formatting based on the mask
    data_with_format_df.loc[is_currency, ['Pre', 'Post','Var']] = data_with_format_df.loc[is_currency, ['Pre', 'Post','Var']].applymap(format_currency)
    data_with_format_df.loc[is_general, ['Pre', 'Post','Var']] = data_with_format_df.loc[is_general, ['Pre', 'Post','Var']].applymap(format_general)
    data_with_format_df.loc[is_percent, ['Pre', 'Post','Var']] = data_with_format_df.loc[is_percent, ['Pre', 'Post','Var']].applymap(format_percent)

    # Use the drop method to remove the specified column
    data_with_format_df = data_with_format_df.drop('Promo_Segment', axis=1)
    data_with_format_df['Var2'] = data_df['Var']
    
    data_percent_df = data_df.copy()
    data_percent_df = data_percent_df.drop('Var', axis=1)
    data_percent_df = data_percent_df.drop('Var%', axis=1)
    data_percent_df = data_percent_df.drop('Sign', axis=1)
    data_percent_df = data_percent_df.drop('Promo_Segment', axis=1)
    keep = (data_percent_df['Metrics'] != 'Days_Count') & (data_percent_df['Metrics'] != 'PlayerCount')
    data_percent_df = data_percent_df[keep]
    data_percent_df = data_percent_df.set_index('Metrics')
        
    data_percent_df['Total'] = data_percent_df.sum(axis=1)  # Calculate the total for each row

    # Calculate the percentage of the total for each row
    #data_percent_df.iloc[:, :] = (data_percent_df.iloc[:, :].div(data_percent_df['Total'], axis=0) * 100).round(2)
    
    
    # Check if 'Metrics' column is equal to 'Recency'
    is_recency = data_percent_df.index == 'Recency'

    # Apply different logic for 'Recency'
    data_percent_df.loc[is_recency, :] = 100 + (data_percent_df.loc[is_recency, :]
                                             .div(np.abs(data_percent_df.loc[is_recency, 'Total']), axis=0)
                                             .mul(100)
                                             .round(2))

    # Apply the original logic for other metrics
    data_percent_df.loc[~is_recency, :] = (data_percent_df.loc[~is_recency, :]
                                           .div(data_percent_df.loc[~is_recency, 'Total'], axis=0)
                                           .mul(100)
                                           .round(2))
    
    
    data_percent_df.replace([np.inf, -np.inf], 100, inplace=True)
    data_percent_df = data_percent_df.drop(columns='Total')  # Drop the 'Total' column
    data_percent_df = data_percent_df.reset_index()
    # Replace NaN values with zero
    data_percent_df = data_percent_df.fillna(0)
    
    concatenated_values = pd.concat([data_percent_df['Pre'], data_percent_df['Post']], ignore_index=True)
    min_value = 0
    max_value = concatenated_values.max()
    
    return data_with_format_df, data_percent_df, min_value, max_value


# In[47]:


NoPromo_with_format_df, NoPromo_percent_df, min_value, max_value = data_for_radar_and_table(result_df, 'NoPromo')


# In[48]:


NoPromo_with_format_df


# In[49]:


def player_table_data (dfpre_player,dfpost_player):
    # Merge dataframes on the 'Player' and 'CRM' columns using outer join
    merged_df = pd.merge(dfpre_player, dfpost_player, on=['CustomerID', 'Promo_Segment'], how='outer', suffixes=('_pre', '_post'))

    # Calculate the differences for the specified columns
    merged_df['Sales_All_D'] = merged_df['Sales_All_post'] - merged_df['Sales_All_pre']
    merged_df['Quantity_All_D'] = merged_df['Quantity_All_post'] - merged_df['Quantity_All_pre']
    merged_df['Frequency%_D'] = merged_df['Frequency%_post'] - merged_df['Frequency%_pre']
    merged_df['Recency_D'] = merged_df['Recency_post'] - merged_df['Recency_pre']
    merged_df['Coupon_Number_All_D'] = merged_df['Coupon_Number_All_post'] - merged_df['Coupon_Number_All_pre']
    merged_df['Transactions_D'] = merged_df['Transactions_post'] - merged_df['Transactions_pre']

    # Format columns with 2 decimals
    numeric_columns = ['Transactions_post', 'Quantity_All_post', 'Coupon_Number_All_post', 'Recency_post', 'Sales_All_post',
                      'Sales_All_D','Quantity_All_D','Recency_D','Coupon_Number_All_D','Transactions_D']
    merged_df[numeric_columns] = merged_df[numeric_columns].round(2)

    # Format 'Frequency%' and 'Frequency%_D' as percentages
    merged_df[['Frequency%_post', 'Frequency%_D']] = merged_df[['Frequency%_post', 'Frequency%_D']].apply(lambda x: x * 100).round(2)

    # Create the final dataframe with the desired columns
    df_playertable = merged_df[[
        'Promo_Segment', 'CustomerID', 
        'Sales_All_post', 'Sales_All_D',
        'Quantity_All_post', 'Quantity_All_D',
        'Frequency%_post', 'Frequency%_D',
        'Recency_post', 'Recency_D',
        'Coupon_Number_All_post', 'Coupon_Number_All_D',
        'Transactions_post', 'Transactions_D',
    ]]
    return df_playertable


# # Analysis Process

# In[50]:


dfa=daily_df.copy()


# In[51]:


# Define the date ranges
pre_ini_date = '2019-02-01'
pre_end_date = '2019-05-30'

# Define the date ranges
post_ini_date = '2019-07-01'
post_end_date = '2019-11-30'


# In[52]:


df_period = pre_post_period(pre_ini_date, pre_end_date, post_ini_date, post_end_date, dfa)


# In[53]:


df_segmentation = promo_segmentation(df_period)


# In[54]:


player_df,dfpre_player,dfpost_player = hypothesis_testing_customer(df_segmentation,pre_ini_date,pre_end_date,post_ini_date,post_end_date)


# In[55]:


day_df, dfpre_day, dfpost_day = hypothesis_testing_days(df_segmentation,pre_ini_date,pre_end_date,post_ini_date,post_end_date)


# In[56]:


result_df, result_with_format_df = results(player_df,day_df)


# In[57]:


df_playertable = player_table_data(dfpre_player,dfpost_player)


# In[58]:


df_playertable


# In[59]:


result_df[result_df['Promo_Segment'] == 'Promo']


# In[60]:


result_with_format_df[result_with_format_df['Promo_Segment'] == 'Promo']


# ### For validation

# In[61]:


dfval_avg=dfa.groupby(['Promo_Segment','Period'])['Sales_All', 'Quantity_All'].mean().reset_index()
dfval_avg


# In[62]:


dfval_sum=dfa.groupby(['Promo_Segment','Period'])['Sales_All', 'Quantity_All'].sum().reset_index()
dfval_sum


# # 4. Visualization

# In[63]:


dfap = df_period.groupby(['Transaction_Date','Period']).sum(numeric_only=True).reset_index()

# Create a line chart
figa = go.Figure()

# Add a trace for the line chart with area
figa.add_trace(go.Scatter(x=dfap[dfap['Period']=='Post']['Transaction_Date'], y=dfap[dfap['Period']=='Post']['Sales_All'],
                             mode='lines',
                             line=dict(color='#7DBC13'),
                             fill='tozeroy',  # Fill area below the line for positive values
                             fillcolor='rgba(75, 120, 168, 0.2)',  # Fill color for positive values
                             name='Post'))

# Add a trace for the line chart with area
figa.add_trace(go.Scatter(x=dfap[dfap['Period']=='Pre']['Transaction_Date'], y=dfap[dfap['Period']=='Pre']['Sales_All'],
                             mode='lines',
                             line=dict(color='#9B1831'),
                             fill='tozeroy',  # Fill area below the line for positive values
                             fillcolor='rgba(75, 120, 168, 0.2)',  # Fill color for positive values
                             name='Pre'))

# Add a trace for the line chart with area
figa.add_trace(go.Scatter(x=dfap[dfap['Period']=='Out']['Transaction_Date'], y=dfap[dfap['Period']=='Out']['Sales_All'],
                             mode='markers',
                             line=dict(color='#D8D8D8'),
                             #fill='tozeroy',  # Fill area below the line for positive values
                             fillcolor='rgba(75, 120, 168, 0.2)',  # Fill color for positive values
                             name='Other'))

# Update layout
figa.update_layout(#title='Timeframe: Post-period vs Pre-period',
                     # xaxis_title='Date',
                      yaxis_title='Sales_All',
                      template='simple_white',  # Use the 'simple_white' template
                      xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black', range=[dfap['Transaction_Date'].min(), dfap['Transaction_Date'].max()]),  # Show x-axis at zero
                      yaxis=dict(gridcolor='lightgrey', tickformat='$,.2s'),  # Set the y-axis grid color
                      xaxis_showgrid=True, yaxis_showgrid=True,  # Show the gridlines
                      plot_bgcolor='white',  # Set background color to white
                      paper_bgcolor='white',  # Set paper color to white
                      font=dict(color='black'),  # Set font color to black
                      width=1200,
                      height=396,
                      )

# Show the plot
figa.show()


# In[ ]:





# In[64]:


crm_with_format_df, crm_percent_df, min_value_crm, max_value_crm = data_for_radar_and_table(result_df, 'Promo')


# In[65]:


crm_with_format_df


# In[66]:


categories = crm_percent_df['Metrics']

figcrm = go.Figure()

figcrm.add_trace(go.Scatterpolar(
      r=crm_percent_df['Pre'],
      theta=categories,
      fill='toself',
      name='Pre',
      marker_color='#9B1831'
))
figcrm.add_trace(go.Scatterpolar(
      r=crm_percent_df['Post'],
      theta=categories,
      fill='toself',
      name='Post',
      marker_color='#7DBC13'
))

figcrm.update_layout(
    title_text=f'Promo - Segment (Avg per Player per Day)',
   polar=dict(
    radialaxis=dict(
        visible=True,
        ticksuffix="%",
        tickangle=90,
        angle = 90,
        range = [min_value_crm,max_value_crm+10],
        showline=False,  # Remove radial axis line
        showticklabels=False  # Remove radial axis tick labels
        ),
   angularaxis  = dict(
       rotation = 90,
       direction = "clockwise")
 ),
  width=500,
  height=418,
  showlegend=True
)

figcrm.show()


# In[67]:


no_with_format_df, no_percent_df, min_value_no, max_value_no = data_for_radar_and_table(result_df, 'NoPromo')


# In[68]:


categories = no_percent_df['Metrics']

figno = go.Figure()

figno.add_trace(go.Scatterpolar(
      r=no_percent_df['Pre'],
      theta=categories,
      fill='toself',
      name='Pre',
      marker_color='#9B1831'
))
figno.add_trace(go.Scatterpolar(
      r=no_percent_df['Post'],
      theta=categories,
      fill='toself',
      name='Post',
      marker_color='#7DBC13'
))

figno.update_layout(
    title_text=f'NoPromo - Segment (Avg per Player per Day)',
    polar=dict(
    radialaxis=dict(
        visible=True,
        ticksuffix="%",
        tickangle=90,
        angle = 90,
        range = [min_value_no,max_value_no+10],
        showline=False,  # Remove radial axis line
        showticklabels=False  # Remove radial axis tick labels
        ),
    angularaxis  = dict(
        rotation = 90,
        direction = "clockwise")
  ),
  width=500,
  height=418,
  showlegend=True
)

figno.show()


# In[69]:


test_post_df =  dfpost_day[dfpost_day['Promo_Segment'] == 'Promo']
control_post_df =  dfpost_day[dfpost_day['Promo_Segment'] == 'NoPromo']
# Create a line chart
fig4 = go.Figure()

# Add a trace for the line chart with area
fig4.add_trace(go.Scatter(x=control_post_df['Transaction_Date'], y=control_post_df['Sales_All'],
                                 mode='lines',
                                 line=dict(color='#D8D8D8'),
                                # fill='tozeroy',  # Fill area below the line for positive values
                                # fillcolor='rgba(75, 120, 168, 0.2)',  # Fill color for positive values
                                 name='Control'))

# Add a trace for the line chart with area
fig4.add_trace(go.Scatter(x=test_post_df['Transaction_Date'], y=test_post_df['Sales_All'],
                                 mode='lines',
                                 line=dict(color='#7DBC13'),
                                # fill='tozeroy',  # Fill area below the line for positive values
                                # fillcolor='rgba(75, 120, 168, 0.2)',  # Fill color for positive values
                                 name='Test'))



# Update layout
fig4.update_layout(#title='NetWin - Post Period Test vs Control',
                         # xaxis_title='Date',
                          yaxis_title='Sales_All',
                          template='simple_white',  # Use the 'simple_white' template
                          xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black', range=[dfpost_day['Transaction_Date'].min(), dfpost_day['Transaction_Date'].max()]),  # Show x-axis at zero
                          yaxis=dict(gridcolor='lightgrey', tickformat='$,.2s'),  # Set the y-axis grid color
                          xaxis_showgrid=True, yaxis_showgrid=True,  # Show the gridlines
                          plot_bgcolor='white',  # Set background color to white
                          paper_bgcolor='white',  # Set paper color to white
                          font=dict(color='black'),  # Set font color to black
                          width=1200,
                          height=395,
                          )


# In[70]:


test_pre_df =  dfpre_day[dfpre_day['Promo_Segment'] == 'Promo']
control_pre_df =  dfpre_day[dfpre_day['Promo_Segment'] == 'NoPromo']
# Create a line chart
fig5 = go.Figure()

# Add a trace for the line chart with area
fig5.add_trace(go.Scatter(x=control_pre_df['Transaction_Date'], y=control_pre_df['Sales_All'],
                                 mode='lines',
                                 line=dict(color='#D8D8D8'),
                                # fill='tozeroy',  # Fill area below the line for positive values
                                # fillcolor='rgba(75, 120, 168, 0.2)',  # Fill color for positive values
                                 name='Control'))

# Add a trace for the line chart with area
fig5.add_trace(go.Scatter(x=test_pre_df['Transaction_Date'], y=test_pre_df['Sales_All'],
                                 mode='lines',
                                 line=dict(color='#9B1831'),
                                # fill='tozeroy',  # Fill area below the line for positive values
                                # fillcolor='rgba(75, 120, 168, 0.2)',  # Fill color for positive values
                                 name='Test'))



# Update layout
fig5.update_layout(#title='NetWin - Post Period Test vs Control',
                         # xaxis_title='Date',
                          yaxis_title='Sales_All',
                          template='simple_white',  # Use the 'simple_white' template
                          xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black', range=[dfpre_day['Transaction_Date'].min(), dfpre_day['Transaction_Date'].max()]),  # Show x-axis at zero
                          yaxis=dict(gridcolor='lightgrey', tickformat='$,.2s'),  # Set the y-axis grid color
                          xaxis_showgrid=True, yaxis_showgrid=True,  # Show the gridlines
                          plot_bgcolor='white',  # Set background color to white
                          paper_bgcolor='white',  # Set paper color to white
                          font=dict(color='black'),  # Set font color to black
                          width=1200,
                          height=395,
                          )


# # 5. Dashboard

# In[71]:


# Initial Settings

# Define the date ranges
pre_ini_date = '2019-02-01'
pre_end_date = '2019-05-30'

# Define the date ranges
post_ini_date = '2019-07-01'
post_end_date = '2019-11-30'

f1_prev_pre_ini_date = pre_ini_date
f1_prev_pre_end_date = pre_end_date
f1_prev_post_ini_date = post_ini_date
f1_prev_post_end_date = post_end_date

f2_prev_pre_ini_date = pre_ini_date
f2_prev_pre_end_date = pre_end_date
f2_prev_post_ini_date = post_ini_date
f2_prev_post_end_date = post_end_date

f3_prev_pre_ini_date = pre_ini_date
f3_prev_pre_end_date = pre_end_date
f3_prev_post_ini_date = post_ini_date
f3_prev_post_end_date = post_end_date

f4_prev_pre_ini_date = pre_ini_date
f4_prev_pre_end_date = pre_end_date
f4_prev_post_ini_date = post_ini_date
f4_prev_post_end_date = post_end_date



# In[ ]:


# Import packages
import time
from dash import Dash, dcc, html, dash_table, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_ag_grid as dag
import dash_mantine_components as dmc
import dash_html_components as html

# Initialise the App

app = Dash(__name__)
server = app.server

# Title
title = dmc.Title(children = 'Performance Analysis Framework', order = 3, style = {'font-family':'Fort-ExtraBold, sans-serif', 'text-align':'center', 'color' :'#414B56'})



getRowStyle = {
    "styleConditions": [
        {
            "condition": "params.data.Sign == 'Yes' && params.data.Var2 < 0",
            "style": {"backgroundColor": "#FFF1F3"},
        },
        {
            "condition": "params.data.Sign == 'Yes' && params.data.Var2 > 0",
            "style": {"backgroundColor": "#F5FBEA"},
        },
    ],
}
data_table_test = dag.AgGrid(
                            id='data_table_test',
                            rowData=crm_with_format_df.to_dict('records'),
                            getRowStyle=getRowStyle,
                            columnDefs=[{"autoHeight": True, "field": col} for col in crm_with_format_df.columns if col != 'Var2'],
                            defaultColDef={'sortable': True, 'filter': True, 'selectable': True, 'deletable': True},
                            className='ag-theme-alpine ag-theme-blue_compact',
                            columnSize="autoSize",
                            style={"height": 420, "width": 570},
                            dashGridOptions={"domLayout": "autoHeight"},
                            )

data_table_control = dag.AgGrid(id="data_table_control",
                                rowData=no_with_format_df.to_dict('records'),
                                getRowStyle=getRowStyle,
                                columnDefs=[{"autoHeight": True,"field": col}for col in crm_with_format_df.columns if col != 'Var2'],
                                defaultColDef={'sortable': True, 'filter': True, 'selectable': True, 'deletable': True},
                                className='ag-theme-alpine ag-theme-blue_compact',
                                columnSize="autoSize",
                                style={"height": 420, "width": 580},
                                dashGridOptions = {"domLayout": "autoHeight"},
)
playertable = dag.AgGrid(id="data_playertable",
                        rowData=df_playertable.to_dict('records'),
                        columnDefs=[{"headerName": col, "field": col}for col in df_playertable.columns                        ],
                        defaultColDef={'sortable': True, 'filter': True, 'selectable': True, 'deletable': True},
                        columnSize="autoSize",
                        style={"height": 600, "width": '100%'},
                        className='ag-theme-alpine ag-theme-blue',
                        
)



# PRE Date Picker
pre_date_range = dmc.DateRangePicker(id = 'pre_date_range',
                                    inputFormat="DD-MM-YYYY",
                                    label="PRE-date",
                                    size='xs',                                  
                                    value=[pre_ini_date, pre_end_date],
                                    minDate=dfa.Transaction_Date.min(),
                                    maxDate=dfa.Transaction_Date.max(),
                                    amountOfMonths=3,
                                    )


# POST Date Picker
post_date_range = dmc.DateRangePicker(id = 'post_date_range',
                                    inputFormat="DD-MM-YYYY",
                                    label="POST-date",
                                    value=[post_ini_date, post_end_date],
                                    minDate=dfa.Transaction_Date.min(),
                                    maxDate=dfa.Transaction_Date.max(),
                                    size='xs',
                                    amountOfMonths=3,
                                     )

# Dropdowns
metrics_dropdown = dmc.Select(id='metrics',
                              data=['Sales_All','Quantity_All','Coupon_Number_All','Transactions'],
                              value="Sales_All")

unique_values = df_segmentation['Promo_Segment'].unique()

test_dropdown = dmc.Select(id='test_segment',
                           data=[{'label': val, 'value': val} for val in unique_values],
                           label="TEST Segment",
                           value="Promo")

control_dropdown = dmc.Select(id='control_segment',
                              data=[{'label': val, 'value': val} for val in unique_values],
                              label="CONTROL Segment",
                              value="NoPromo")

post_columns = dfpost_day.select_dtypes(include=['number']).columns.tolist()
post_test_vs_control = dmc.Select(id='post_test_vs_control', 
                                  
                                  data=[{'label': col, 'value': col} for col in post_columns],
                                  value="Sales_All")

# consistent_players_dropdown = dmc.Select(id='consistent_players',
#                                          label="All-Players vs Consistent-Players",
#                                          data=[{'label': val, 'value': val} for val in ['All Players','Consistent Players']],
#                                          value='All Players')
# Figures
graph1 = dcc.Graph(id='figure1', figure=figa)
graph2 = dcc.Graph(id='figure2', figure=figcrm)
graph3 = dcc.Graph(id='figure3', figure=figno)
graph4 = dcc.Graph(id='figure4', figure=fig4)
graph5 = dcc.Graph(id='figure5', figure=fig5)

# Card content
card_content_1_1 = dmc.Card(children=[dmc.Text("Metric", 
                                            weight=500,
                                            #size="sm",
                                            #color="dimmed",
                                           ),
                                        html.Br(),
                                        metrics_dropdown,
                                        html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                        html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                        ],
                            withBorder=True,
                            shadow="sm",
                            radius="md",
                            style={"width": '100%'},
                        )

card_content_1_2 = dmc.Card(children=[dmc.Text('Timeframe: Post-period vs Pre-period', 
                                            weight=500,
                                            #size="sm",
                                            #color="dimmed",
                                           ),
                                      dmc.Center(dmc.LoadingOverlay(id="loading-1",
                                                                    loaderProps={"color": "orange"},
                                                                    children=[graph1]))],
                            withBorder=True,
                            shadow="sm",
                            radius="md",
                            style={"width": '100%'},
                        )
card_content_2_1 = dmc.Card(children=[dmc.Text("Promo-Segments", 
                                            weight=500,
                                            #size="sm",
                                            #color="dimmed",
                                           ),
                                    html.Br(),
                                    test_dropdown,
                                    html.Br(),
                                    control_dropdown,  
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                    html.Br(),html.Br(),
                                    ],
                            withBorder=True,
                            shadow="sm",
                            radius="md",
                            style={"width": '100%'},
                        )
card_content_2_2 = dmc.Card(children=[dmc.Text("TEST Segment", 
                                            weight=500,
                                            #size="sm",
                                            #color="dimmed",
                                           ),
                                      dmc.Center(dmc.LoadingOverlay(id="loading-2",
                                                                    loaderProps={"color": "orange"},
                                                                    children=[graph2])),
                                      dmc.Center(dmc.LoadingOverlay(id="loading-3",
                                                                    loaderProps={"color": "orange"},
                                                                    children=[data_table_test])),
                                      html.Br(),],
                            withBorder=True,
                            shadow="sm",
                            radius="md",
                            style={"width": '100%'},
                        )
card_content_2_3 = dmc.Card(children=[dmc.Text("CONTROL Segment", 
                                            weight=500,
                                            #size="sm",
                                            #color="dimmed",
                                           ),
                                      dmc.Center(dmc.LoadingOverlay(id="loading-4",
                                                                    loaderProps={"color": "orange"},
                                                                    children=[graph3])),
                                      dmc.Center(dmc.LoadingOverlay(id="loading-5",
                                                                    loaderProps={"color": "orange"},
                                                                    children=[data_table_control])),
                                      html.Br(),],
                            withBorder=True,
                            shadow="sm",
                            radius="md",
                            style={"width": '100%'},
                        )
card_content_3_1 = dmc.Card(children=[dmc.Text("Metric", 
                                            weight=500,
                                            #size="sm",
                                            #color="dimmed",
                                           ),
                                     html.Br(),
                                     dmc.Center(post_test_vs_control),
                                     html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                     html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                     html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                     html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                    ],
                            withBorder=True,
                            shadow="sm",
                            radius="md",
                            style={"width": '100%'},
                        )

card_content_3_2 = dmc.Card(children=[dmc.Text('POST-Period: TEST vs CONTROL', 
                                            weight=500,
                                            #size="sm",
                                            #color="dimmed",
                                           ),
                                      dmc.Center(dmc.LoadingOverlay(id="loading-6",
                                                                    loaderProps={"color": "orange"},
                                                                    children=[graph4])),
                                      dmc.Text('PRE-Period: TEST vs CONTROL', 
                                            weight=500,
                                            #size="sm",
                                            #color="dimmed",
                                           ),
                                      dmc.Center(dmc.LoadingOverlay(id="loading-7",
                                                                    loaderProps={"color": "orange"},
                                                                    children=[graph5])),],
                            withBorder=True,
                            shadow="sm",
                            radius="md",
                            style={"width": '100%'},
                        )
card_content_4_1 = dmc.Card(children=[dmc.Text("Post Player Detail", 
                                            weight=500,
                                            #size="sm",
                                            #color="dimmed",
                                           ),
                                      html.Br(),
                                      dmc.Text("Select, Sort, Filter, and Delete, directly on the table  ", 
                                            weight=500,
                                            #size="sm",
                                            #color="dimmed",
                                           ),
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),

                                    ],
                            withBorder=True,
                            shadow="sm",
                            radius="md",
                            style={"width": '100%'},
                        )

card_content_4_2 = dmc.Card(children=[dmc.Text('Post-Period: TEST vs CONTROL', 
                                            weight=500,
                                            #size="sm",
                                            #color="dimmed",
                                           ),
                                      dmc.Center(playertable),],
                            withBorder=True,
                            shadow="sm",
                            radius="md",
                            style={"width": '100%'},
                        )


# App Layout
app.layout = dmc.Container(
    [
                dmc.Loader(color="red", size="md", variant="oval"),
                dmc.Header(
                    height=100,
                    fixed=True,
                    pl=0,
                    pr=0,
                    pt=0,
                    style = {'background-color':'white'},
                    children=[dmc.Grid(children=[dmc.Col([html.Br(),dmc.Image(src="/assets/bclc-logo.png",width=120,height=50, style={'float': 'left'})],span=2),
                                                 dmc.Col(dmc.Center(dmc.Container([html.Br(),dmc.Text("Global Variables:", weight=500)])),span=1),
                                                 dmc.Col([html.Br(),pre_date_range],span=3),
                                                 dmc.Col([html.Br(),post_date_range],span=3) ,
                                                 ]),]),
              
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                dmc.Center([title], style={"text-align": "center", "margin": "auto"}),
                html.Br(),
                dmc.Grid(children=[dmc.Col(card_content_1_1, span=2),
                                   dmc.Col(card_content_1_2, span=10)]),
                dmc.Grid(children=[dmc.Col(card_content_2_1, span=2),
                                   dmc.Col(card_content_2_2, span=5),
                                   dmc.Col(card_content_2_3, span=5)]),
                dmc.Grid(children=[dmc.Col(card_content_3_1, span=2),
                                   dmc.Col(card_content_3_2, span=10)]),
                dmc.Grid(children=[dmc.Col(card_content_4_1, span=2),
                                   dmc.Col(card_content_4_2, span=10)]),
            ],
    size=2000,
    bg="#F4F5F5",
)

# @app.callback(
#     Output("loading-1", "children"),
#     Input(component_id='pre_date_range', component_property='value'),
#     Input(component_id='post_date_range', component_property='value'),
#     Input(component_id='metrics', component_property='value'),
#     Input(component_id='consistent_players', component_property='value'),
#     prevent_initial_call=True
# )
# def func(pre_range, post_range,metrics_column,consistent_players):
#     return no_update

@app.callback(
    Output(component_id='figure1', component_property='figure'),
    Output(component_id='data_playertable', component_property='rowData'),
    Input(component_id='pre_date_range', component_property='value'),
    Input(component_id='post_date_range', component_property='value'),
    Input(component_id='metrics', component_property='value'),


)
def function1(
            pre_range, post_range,metrics_column
           ):
    
    global f1_prev_pre_ini_date, f1_prev_pre_end_date, f1_prev_post_ini_date, f1_prev_post_end_date 
    global dfa, df_period, df_segmentation, player_df, dfpre_player, dfpost_player
    global day_df, dfpre_day, dfpost_day, result_df, result_with_format_df
    global df_playertable
    
    
    
    if (pre_range is not None and post_range is not None):
        
        pre_ini_date,pre_end_date = pre_range
        post_ini_date,post_end_date = post_range
    
        if (pre_ini_date is not None and pre_end_date is not None
            and post_ini_date is not None and post_end_date is not None
            and (pre_ini_date != f1_prev_pre_ini_date
            or pre_end_date != f1_prev_pre_end_date
            or post_ini_date != f1_prev_post_ini_date
            or post_end_date != f1_prev_post_end_date)):

            f1_prev_pre_ini_date = pre_ini_date
            f1_prev_pre_end_date = pre_end_date
            f1_prev_post_ini_date = post_ini_date
            f1_prev_post_end_date = post_end_date

            dfa = df0.copy()       
            df_period = pre_post_period(pre_ini_date, pre_end_date, post_ini_date, post_end_date, dfa)
            df_segmentation = promo_segmentation(df_period)
            player_df,dfpre_player,dfpost_player = hypothesis_testing_customer(df_segmentation,pre_ini_date,pre_end_date,post_ini_date,post_end_date)   
            df_playertable = player_table_data(dfpre_player,dfpost_player)

        else:
            pre_ini_date = f1_prev_pre_ini_date
            pre_end_date = f1_prev_pre_end_date
            post_ini_date = f1_prev_post_ini_date
            post_end_date = f1_prev_post_end_date 
            
       
    dfap = df_period.groupby(['Transaction_Date','Period']).sum(numeric_only=True).reset_index()
        
    fig1 = go.Figure()

    # Add a trace for the line chart with area
    fig1.add_trace(go.Scatter(x=dfap[dfap['Period']=='Post']['Transaction_Date'], y=dfap[dfap['Period']=='Post'][metrics_column],
                                 mode='lines',
                                 line=dict(color='#7DBC13'),
                                 fill='tozeroy',  # Fill area below the line for positive values
                                 fillcolor='rgba(75, 120, 168, 0.2)',  # Fill color for positive values
                                 name='Post'))

    # Add a trace for the line chart with area
    fig1.add_trace(go.Scatter(x=dfap[dfap['Period']=='Pre']['Transaction_Date'], y=dfap[dfap['Period']=='Pre'][metrics_column],
                                 mode='lines',
                                 line=dict(color='#9B1831'),
                                 fill='tozeroy',  # Fill area below the line for positive values
                                 fillcolor='rgba(75, 120, 168, 0.2)',  # Fill color for positive values
                                 name='Pre'))

    # Add a trace for the line chart with area
    fig1.add_trace(go.Scatter(x=dfap[dfap['Period']=='Out']['Transaction_Date'], y=dfap[dfap['Period']=='Out'][metrics_column],
                                 mode='markers',
                                 line=dict(color='#D8D8D8'),
                                 #fill='tozeroy',  # Fill area below the line for positive values
                                 fillcolor='rgba(75, 120, 168, 0.2)',  # Fill color for positive values
                                 name='Other'))

    # Update layout
    fig1.update_layout(#title='Timeframe: Post-period vs Pre-period',
                         # xaxis_title='Date',
                          yaxis_title=f'{metrics_column}',
                          template='simple_white',  # Use the 'simple_white' template
                          xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black', range=[dfap['Transaction_Date'].min(), dfap['Transaction_Date'].max()]),  # Show x-axis at zero
                          yaxis=dict(gridcolor='lightgrey', tickformat='$,.2s'),  # Set the y-axis grid color
                          xaxis_showgrid=True, yaxis_showgrid=True,  # Show the gridlines
                          plot_bgcolor='white',  # Set background color to white
                          paper_bgcolor='white',  # Set paper color to white
                          font=dict(color='black'),  # Set font color to black
                          width=1200,
                          height=396,
                          showlegend=True
                         )    
    
       
      
    return fig1, df_playertable.to_dict('records')



@app.callback(
    Output(component_id='figure2', component_property='figure'),
    Output(component_id='data_table_test', component_property='rowData'),
    Input(component_id='pre_date_range', component_property='value'),
    Input(component_id='post_date_range', component_property='value'),
    Input(component_id='test_segment', component_property='value'),
    Input(component_id='control_segment', component_property='value'),
    Input(component_id='post_test_vs_control', component_property='value'),


)
def function2(
            pre_range, post_range, test_segment,control_segment,column
           ):
    
    global f2_prev_pre_ini_date, f2_prev_pre_end_date, f2_prev_post_ini_date, f2_prev_post_end_date 
    global dfa, df_period, df_segmentation, player_df, dfpre_player, dfpost_player
    global day_df, dfpre_day, dfpost_day, result_df, result_with_format_df
    global df_playertable
    
    if (pre_range is not None and post_range is not None):
        
        pre_ini_date,pre_end_date = pre_range
        post_ini_date,post_end_date = post_range
            
        if (
            pre_ini_date is not None and pre_end_date is not None
            and post_ini_date is not None and post_end_date is not None
            and (
                pre_ini_date != f2_prev_pre_ini_date
                or pre_end_date != f2_prev_pre_end_date
                or post_ini_date != f2_prev_post_ini_date
                or post_end_date != f2_prev_post_end_date
                )
            ): 
            f2_prev_pre_ini_date = pre_ini_date
            f2_prev_pre_end_date = pre_end_date
            f2_prev_post_ini_date = post_ini_date
            f2_prev_post_end_date = post_end_date
            
            df_period = pre_post_period(pre_ini_date, pre_end_date, post_ini_date, post_end_date, dfa)
            df_segmentation = promo_segmentation(df_period)   
            player_df,dfpre_player,dfpost_player = hypothesis_testing_customer(df_segmentation,pre_ini_date,pre_end_date,post_ini_date,post_end_date)    
            day_df, dfpre_day, dfpost_day = hypothesis_testing_days(df_segmentation,pre_ini_date,pre_end_date,post_ini_date,post_end_date)
            result_df, result_with_format_df = results(player_df,day_df)
        
        else:
            pre_ini_date = f2_prev_pre_ini_date
            pre_end_date = f2_prev_pre_end_date
            post_ini_date = f2_prev_post_ini_date
            post_end_date = f2_prev_post_end_date 
        
        
    data_test, percent_df, min_value, max_value = data_for_radar_and_table(result_df, test_segment)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatterpolar(
          r=percent_df['Pre'],
          theta=categories,
          fill='toself',
          name='Pre',
          marker_color='#9B1831'
    ))
    fig2.add_trace(go.Scatterpolar(
          r=percent_df['Post'],
          theta=categories,
          fill='toself',
          name='Post',
          marker_color='#7DBC13'
    ))

    fig2.update_layout(
        title_text=f'{test_segment} - Segment (Avg per Player per Day)',
        polar=dict(
        radialaxis=dict(
            visible=True,
            ticksuffix="%",
            tickangle=90,
            angle = 90,
            range = [min_value,max_value+10],
            showline=False,  # Remove radial axis line
            showticklabels=False  # Remove radial axis tick labels
            ),
        angularaxis  = dict(
            rotation = 90,
            direction = "clockwise")
      ),
      width=500,
      height=418,
      showlegend=True
    )


    return fig2, data_test.to_dict('records')


        
@app.callback(
    Output(component_id='figure3', component_property='figure'),
    Output(component_id='data_table_control', component_property='rowData'),
    Input(component_id='pre_date_range', component_property='value'),
    Input(component_id='post_date_range', component_property='value'),
    Input(component_id='test_segment', component_property='value'),
    Input(component_id='control_segment', component_property='value'),
    Input(component_id='post_test_vs_control', component_property='value'),

)
def function3(
            pre_range, post_range, test_segment,control_segment,column
           ):
    
    global f3_prev_pre_ini_date, f3_prev_pre_end_date, f3_prev_post_ini_date, f3_prev_post_end_date 
    global dfa, df_period, df_segmentation, player_df, dfpre_player, dfpost_player
    global day_df, dfpre_day, dfpost_day, result_df, result_with_format_df
    global df_playertable
    
    if (pre_range is not None and post_range is not None):
        
        pre_ini_date,pre_end_date = pre_range
        post_ini_date,post_end_date = post_range
            
        if (
            pre_ini_date is not None and pre_end_date is not None
            and post_ini_date is not None and post_end_date is not None
            and (
                pre_ini_date != f3_prev_pre_ini_date
                or pre_end_date != f3_prev_pre_end_date
                or post_ini_date != f3_prev_post_ini_date
                or post_end_date != f3_prev_post_end_date
                )
            ): 
            f3_prev_pre_ini_date = pre_ini_date
            f3_prev_pre_end_date = pre_end_date
            f3_prev_post_ini_date = post_ini_date
            f3_prev_post_end_date = post_end_date
            

            df_period = pre_post_period(pre_ini_date, pre_end_date, post_ini_date, post_end_date, dfa)
            df_segmentation = promo_segmentation(df_period)   
            player_df,dfpre_player,dfpost_player = hypothesis_testing_customer(df_segmentation,pre_ini_date,pre_end_date,post_ini_date,post_end_date)    
            day_df, dfpre_day, dfpost_day = hypothesis_testing_days(df_segmentation,pre_ini_date,pre_end_date,post_ini_date,post_end_date)
            result_df, result_with_format_df = results(player_df,day_df)
        
        else:
            pre_ini_date = f3_prev_pre_ini_date
            pre_end_date = f3_prev_pre_end_date
            post_ini_date = f3_prev_post_ini_date
            post_end_date = f3_prev_post_end_date 
        
        
    data_control, percent_df, min_value, max_value = data_for_radar_and_table(result_df, control_segment)

    fig3 = go.Figure()

    fig3.add_trace(go.Scatterpolar(
          r=percent_df['Pre'],
          theta=categories,
          fill='toself',
          name='Pre',
          marker_color='#9B1831'
    ))
    fig3.add_trace(go.Scatterpolar(
          r=percent_df['Post'],
          theta=categories,
          fill='toself',
          name='Post',
          marker_color='#7DBC13'
    ))

    fig3.update_layout(
        title_text=f'{control_segment} - Segment (Avg per Player per Day)',
        polar=dict(
        radialaxis=dict(
            visible=True,
            ticksuffix="%",
            tickangle=90,
            angle = 90,
            range = [min_value,max_value+10],
            showline=False,  # Remove radial axis line
            showticklabels=False  # Remove radial axis tick labels
            ),
        angularaxis  = dict(
            rotation = 90,
            direction = "clockwise")
      ),
      width=500,
      height=418,
      showlegend=True
    )


    return fig3, data_control.to_dict('records')
        
        
@app.callback(
    Output(component_id='figure4', component_property='figure'),
    Output(component_id='figure5', component_property='figure'),
    Input(component_id='pre_date_range', component_property='value'),
    Input(component_id='post_date_range', component_property='value'),
    Input(component_id='test_segment', component_property='value'),
    Input(component_id='control_segment', component_property='value'),
    Input(component_id='post_test_vs_control', component_property='value'),


)
def function4(
            pre_range, post_range, test_segment,control_segment,column
           ):
    
    global f4_prev_pre_ini_date, f4_prev_pre_end_date, f4_prev_post_ini_date, f4_prev_post_end_date 
    global dfa, df_period, df_segmentation, player_df, dfpre_player, dfpost_player
    global day_df, dfpre_day, dfpost_day, result_df, result_with_format_df
    global df_playertable
    
    if (pre_range is not None and post_range is not None):
        
        pre_ini_date,pre_end_date = pre_range
        post_ini_date,post_end_date = post_range
  
        if (
            pre_ini_date is not None and pre_end_date is not None
            and post_ini_date is not None and post_end_date is not None
            and (
                pre_ini_date != f4_prev_pre_ini_date
                or pre_end_date != f4_prev_pre_end_date
                or post_ini_date != f4_prev_post_ini_date
                or post_end_date != f4_prev_post_end_date
                )
            ): 

            f4_prev_pre_ini_date = pre_ini_date
            f4_prev_pre_end_date = pre_end_date
            f4_prev_post_ini_date = post_ini_date
            f4_prev_post_end_date = post_end_date
            

            df_period = pre_post_period(pre_ini_date, pre_end_date, post_ini_date, post_end_date, dfa)
            df_segmentation = promo_segmentation(df_period)   
            player_df,dfpre_player,dfpost_player = hypothesis_testing_customer(df_segmentation,pre_ini_date,pre_end_date,post_ini_date,post_end_date)    
            day_df, dfpre_day, dfpost_day = hypothesis_testing_days(df_segmentation,pre_ini_date,pre_end_date,post_ini_date,post_end_date)
            result_df, result_with_format_df = results(player_df,day_df)
        
        else:
            pre_ini_date = f4_prev_pre_ini_date
            pre_end_date = f4_prev_pre_end_date
            post_ini_date = f4_prev_post_ini_date
            post_end_date = f4_prev_post_end_date 
        
    
       
    test_post_df =  dfpost_day[dfpost_day['Promo_Segment'] == test_segment]
    control_post_df =  dfpost_day[dfpost_day['Promo_Segment'] == control_segment]
    # Create a line chart
    fig4 = go.Figure()
    
    # Add a trace for the line chart with area
    fig4.add_trace(go.Scatter(x=control_post_df['Transaction_Date'], y=control_post_df[column],
                                 mode='lines',
                                 line=dict(color='#D8D8D8'),
                               #  fill='tozeroy',  # Fill area below the line for positive values
                               #  fillcolor='rgba(75, 120, 168, 0.2)',  # Fill color for positive values
                                 name='Control'))
    
    # Add a trace for the line chart with area
    fig4.add_trace(go.Scatter(x=test_post_df['Transaction_Date'], y=test_post_df[column],
                                 mode='lines',
                                 line=dict(color='#7DBC13'),
                               #  fill='tozeroy',  # Fill area below the line for positive values
                               #  fillcolor='rgba(75, 120, 168, 0.2)',  # Fill color for positive values
                                 name='Test'))

    # Update layout
    fig4.update_layout(#title=f'{column} - Post Period Test vs Control',
                         # xaxis_title='Date',
                          yaxis_title=f'{column}',
                          template='simple_white',  # Use the 'simple_white' template
                          xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black', range=[dfpost_day['Transaction_Date'].min(), dfpost_day['Transaction_Date'].max()]),  # Show x-axis at zero
                          yaxis=dict(gridcolor='lightgrey', tickformat=',.2s'),  # Set the y-axis grid color
                          xaxis_showgrid=True, yaxis_showgrid=True,  # Show the gridlines
                          plot_bgcolor='white',  # Set background color to white
                          paper_bgcolor='white',  # Set paper color to white
                          font=dict(color='black'),  # Set font color to black
                          width=1200,
                          height=395,
                          )
    
    test_pre_df =  dfpre_day[dfpre_day['Promo_Segment'] == test_segment]
    control_pre_df =  dfpre_day[dfpre_day['Promo_Segment'] == control_segment]
    # Create a line chart
    fig5 = go.Figure()
    
    # Add a trace for the line chart with area
    fig5.add_trace(go.Scatter(x=control_pre_df['Transaction_Date'], y=control_pre_df[column],
                                 mode='lines',
                                 line=dict(color='#D8D8D8'),
                               #  fill='tozeroy',  # Fill area below the line for positive values
                               #  fillcolor='rgba(75, 120, 168, 0.2)',  # Fill color for positive values
                                 name='Control'))
    
    # Add a trace for the line chart with area
    fig5.add_trace(go.Scatter(x=test_pre_df['Transaction_Date'], y=test_pre_df[column],
                                 mode='lines',
                                 line=dict(color='#9B1831'),
                               #  fill='tozeroy',  # Fill area below the line for positive values
                               #  fillcolor='rgba(75, 120, 168, 0.2)',  # Fill color for positive values
                                 name='Test'))

    # Update layout
    fig5.update_layout(#title=f'{column} - Post Period Test vs Control',
                         # xaxis_title='Date',
                          yaxis_title=f'{column}',
                          template='simple_white',  # Use the 'simple_white' template
                          xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black', range=[dfpre_day['Transaction_Date'].min(), dfpre_day['Transaction_Date'].max()]),  # Show x-axis at zero
                          yaxis=dict(gridcolor='lightgrey', tickformat=',.2s'),  # Set the y-axis grid color
                          xaxis_showgrid=True, yaxis_showgrid=True,  # Show the gridlines
                          plot_bgcolor='white',  # Set background color to white
                          paper_bgcolor='white',  # Set paper color to white
                          font=dict(color='black'),  # Set font color to black
                          width=1200,
                          height=395,
                          )
    
    
    
    
    
    return fig4, fig5

# Run the App
if __name__ == "__main__":
    app.run_server(debug=False)
#    app.run_server(jupyter_mode="external")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




