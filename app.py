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


df_initiatives_file = pd.read_csv('Initiatives.csv', index_col=0).reset_index()
df_initiatives_file.shape


# In[3]:


df_initiatives_file.sample(5)


# In[4]:


df_initiatives_file.info()


# In[5]:


df = pd.read_csv('Online_Sales.csv', index_col=0).reset_index()

df.shape


# In[6]:


df.info()


# In[7]:


df


# In[8]:


df.isna().sum()


# In[9]:


df0=df.copy()


# ## iii) Feature Engineering (Adding columns)

# In[10]:


date_format='%m/%d/%Y'
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'],format=date_format)


# In[11]:


df


# In[12]:


df['Sales'] = df['Quantity'] * df['Avg_Price']


# In[13]:


df['Year'] = df['Transaction_Date'].dt.year


# In[14]:


df['Month'] = df['Transaction_Date'].dt.month


# In[15]:


df['Coupons'] = np.where(df['Coupon_Status'] == 'Used', 1, 0)


# In[16]:


df_product_category = df['Product_Category'].value_counts()
df_product_category


# In[17]:


# Create a new column 'Product_Category_Grouped' to combine categories
df['Product_Category_Grouped'] = df['Product_Category'].apply(lambda x: x if x in ['Apparel', 'Nest-USA'] else 'Other')

# Create a pivot table to transform the data
pivot_df = pd.pivot_table(df, 
                          index=['CustomerID', 'Transaction_ID', 'Transaction_Date','Month','Year'],
                          values=['Quantity', 'Avg_Price', 'Sales', 'Coupons'],
                          columns='Product_Category_Grouped',
                          aggfunc={
                              'Quantity': 'sum',
                              'Avg_Price': 'mean',
                              'Sales': 'sum',
                              'Coupons': 'sum'
                          },
                          fill_value=0)

# Flatten the multi-level columns
pivot_df.columns = ['_'.join(map(str, col)).strip('_') for col in pivot_df.columns.values]

# Reset index to make it flat
pivot_df.reset_index(inplace=True)

pivot_df


# In[18]:


pivot_df['Quantity_All'] = pivot_df['Quantity_Apparel'] + pivot_df['Quantity_Nest-USA'] + pivot_df['Quantity_Other']
pivot_df['Sales_All'] = pivot_df['Sales_Apparel'] + pivot_df['Sales_Nest-USA'] + pivot_df['Sales_Other']
pivot_df['Coupons_All'] = pivot_df['Coupons_Apparel'] + pivot_df['Coupons_Nest-USA'] + pivot_df['Coupons_Other']
pivot_df['Price_All'] = pivot_df['Sales_All'] / pivot_df['Quantity_All']


# In[19]:


pivot_df.info()


# In[20]:


agg_functions = {
    'Transaction_ID': 'nunique',
    'Quantity_All': 'sum',
    'Price_All': 'mean',
    'Sales_All': 'sum',
    'Coupons_All': 'sum',
    'Quantity_Apparel': 'sum',
    'Sales_Apparel': 'sum',
    'Coupons_Apparel': 'sum',
    'Quantity_Nest-USA': 'sum',
    'Sales_Nest-USA': 'sum',
    'Coupons_Nest-USA': 'sum',
    'Quantity_Other': 'sum',
    'Sales_Other': 'sum',
    'Coupons_Other': 'sum',
}


# In[21]:


agg_columns = ['Transaction_ID', 'Quantity_All','Price_All', 'Sales_All', 'Coupons_All', 'Quantity_Apparel', 'Sales_Apparel', 'Coupons_Apparel','Quantity_Nest-USA', 'Sales_Nest-USA', 'Coupons_Nest-USA','Quantity_Other', 'Sales_Other', 'Coupons_Other']


# In[22]:


daily_df = pivot_df.groupby(['CustomerID','Transaction_Date','Month','Year']).agg(agg_functions).reset_index()


# In[23]:


daily_df.rename(columns={
    'Transaction_ID': 'Transactions',
}, inplace=True)


# In[24]:


daily_df


# # 1. Data Exploration

# In[25]:


dfexp0=daily_df.copy()


# In[26]:


dfexp1=dfexp0.groupby(['Year'])['Sales_All'].sum().reset_index()
dfexp1


# In[27]:


dfexp2=dfexp0.groupby(['Month'])['Sales_All'].sum().reset_index()
dfexp2


# In[28]:


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


# In[29]:


dfexp3=dfexp0.groupby(['Transaction_Date','Month'])['Sales_All'].sum().reset_index()
dfexp3


# In[30]:


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

# In[31]:


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
    print('period')
    return dfa


# ### Promo Segmentation

# In[32]:


def promo_segmentation(dfa):
    dfa['Promo_Segment'] = 'Other'  # Default to 'Other'
    
    #Grouping for Promo_segmentation
    grouped_df = pd.pivot_table(data=dfa,
                                index='CustomerID',
                                columns='Period',
                                values=['Coupons_All'],
                                aggfunc={'Coupons_All':np.sum})
    grouped_df=grouped_df.fillna(0)    
    grouped_df.columns = ['{}_{}'.format(level2, level1) for level1, level2 in grouped_df.columns]
    grouped_df= grouped_df.reset_index()
    grouped_df
    
    #Promo_Segmentation

    # Segment 1
    dfa.loc[(dfa['CustomerID'].isin(grouped_df[(grouped_df['Pre_Coupons_All'] != 0) & (grouped_df['Post_Coupons_All'] != 0)]['CustomerID'])), 'Promo_Segment'] = 'Promo'

    # Segment 2
    dfa.loc[(dfa['CustomerID'].isin(grouped_df[(grouped_df['Pre_Coupons_All'] != 0) & (grouped_df['Post_Coupons_All'] == 0)]['CustomerID'])), 'Promo_Segment'] = 'PromoPre'

    # Segment 3
    dfa.loc[(dfa['CustomerID'].isin(grouped_df[(grouped_df['Pre_Coupons_All'] == 0) & (grouped_df['Post_Coupons_All'] != 0)]['CustomerID'])), 'Promo_Segment'] = 'PromoPost'

    # Segment 4
    dfa.loc[(dfa['CustomerID'].isin(grouped_df[(grouped_df['Pre_Coupons_All'] == 0) & (grouped_df['Post_Coupons_All'] == 0)]['CustomerID'])), 'Promo_Segment'] = 'NoPromo'
    
    print('segmentation')
    return dfa


# ### Initiatives Groups

# In[33]:


def initiatives(dfa,initiatives):
    dfa = pd.merge(dfa, initiatives,on='CustomerID',how='left')
    dfa = dfa.fillna('NoInitiative')
    print('initiative')
    return dfa


# ### Hypothesis testing

# In[34]:


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


# In[35]:


def features_by_player(dfa,pre_ini_date,pre_end_date,post_ini_date,post_end_date,PS,IG):

    # Define the date ranges
    pre_ini_date = pd.to_datetime(pre_ini_date)
    pre_end_date = pd.to_datetime(pre_end_date)

    # Define the date ranges
    post_ini_date = pd.to_datetime(post_ini_date)
    post_end_date = pd.to_datetime(post_end_date)
    
    
    # First, let's calculate the mean aggregated columns
    agg_columns = ['Transactions', 'Quantity_All','Price_All', 'Sales_All', 'Coupons_All', 'Quantity_Apparel', 'Sales_Apparel', 'Coupons_Apparel','Quantity_Nest-USA', 'Sales_Nest-USA', 'Coupons_Nest-USA','Quantity_Other', 'Sales_Other', 'Coupons_Other']
    
    if (PS != 'All') & (IG == 'All'):  
        dfpre_player =  dfa[(dfa['Period'] == 'Pre')&(dfa['Promo_Segment'] == PS)].groupby(['CustomerID'])[agg_columns].mean().reset_index()
        dfpost_player =  dfa[(dfa['Period'] == 'Post')&(dfa['Promo_Segment'] == PS)].groupby(['CustomerID'])[agg_columns].mean().reset_index()

    
    elif (PS == 'All') & (IG != 'All'):  
        dfpre_player =  dfa[(dfa['Period'] == 'Pre')&(dfa['Initiative'] == IG)].groupby(['CustomerID'])[agg_columns].mean().reset_index()
        dfpost_player =  dfa[(dfa['Period'] == 'Post')&(dfa['Initiative'] == IG)].groupby(['CustomerID'])[agg_columns].mean().reset_index()

    
    elif (PS != 'All') & (IG != 'All'):  
        dfpre_player =  dfa[(dfa['Period'] == 'Pre')&(dfa['Promo_Segment'] == PS)&(dfa['Initiative'] == IG)].groupby(['CustomerID'])[agg_columns].mean().reset_index()
        dfpost_player =  dfa[(dfa['Period'] == 'Post')&(dfa['Promo_Segment'] == PS)&(dfa['Initiative'] == IG)].groupby(['CustomerID'])[agg_columns].mean().reset_index()
    
    
    elif (PS == 'All') & (IG == 'All'):  
        dfpre_player =  dfa[(dfa['Period'] == 'Pre')].groupby(['CustomerID'])[agg_columns].mean().reset_index()
        dfpost_player =  dfa[(dfa['Period'] == 'Post')].groupby(['CustomerID'])[agg_columns].mean().reset_index()
    
    # Filter data in a granular level for the Pre and Post periods
    if (PS != 'All') & (IG == 'All'):
        pre_data = dfa[(dfa['Period'] == 'Pre')&(dfa['Promo_Segment'] == PS)]
        
    
    elif (PS == 'All') & (IG != 'All'):
        pre_data = dfa[(dfa['Period'] == 'Pre')&(dfa['Initiative'] == IG)]
        
    
    elif (PS != 'All') & (IG != 'All'):
        pre_data = dfa[(dfa['Period'] == 'Pre')&(dfa['Promo_Segment'] == PS)&(dfa['Initiative'] == IG)]
        
    
    elif (PS == 'All') & (IG == 'All'):
        pre_data = dfa[(dfa['Period'] == 'Pre')]
        
    ### Feature Engineering: Frequency and Recency, and New Product ###
    
    ### Pre
    
    # Calculate the difference in days
    pre_days_length = (pre_end_date - pre_ini_date).days+1
    
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
    # Calculate the difference in days
    post_days_length = (post_end_date - post_ini_date).days+1
    
    # Filter data for the Pre and Post periods
    if (PS != 'All') & (IG == 'All'):
        post_data = dfa[(dfa['Period'] == 'Post')&(dfa['Promo_Segment'] == PS)]
        
    
    elif (PS == 'All') & (IG != 'All'):
        post_data = dfa[(dfa['Period'] == 'Post')&(dfa['Initiative'] == IG)]
        
    
    elif (PS != 'All') & (IG != 'All'):
        post_data = dfa[(dfa['Period'] == 'Post')&(dfa['Promo_Segment'] == PS)&(dfa['Initiative'] == IG)]
        
    
    elif (PS == 'All') & (IG == 'All'):
        post_data = dfa[(dfa['Period'] == 'Post')]

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
    
    print('feature_player')
    return dfpre_player, dfpost_player


# In[36]:


def hypothesis_testing_player(dfpre_player, dfpost_player):
    # I will be using One-Sample t_Test so we can use a different number of players for Pre and Post period
    # Ho: Promo does nothing vs Ha: Promo does something
    ################################### Per Player ##########################################

    # Filter data for the Pre and Post periods
    pre_data = dfpre_player
    post_data = dfpost_player

    # First, let's calculate the mean aggregated columns
    agg_columns = ['Transactions', 'Quantity_All', 'Sales_All','Price_All', 'Coupons_All', 'Quantity_Apparel', 'Sales_Apparel', 'Coupons_Apparel','Quantity_Nest-USA', 'Sales_Nest-USA', 'Coupons_Nest-USA','Quantity_Other', 'Sales_Other', 'Coupons_Other', 'Frequency%', 'Recency']

    # Calculate weighted average for every metric except Recency which is average
    weighted_avg_pre = pd.Series({
        col: (pre_data[col] * pre_data['Frequency%']).sum() / pre_data['Frequency%'].sum()
        if col in agg_columns and col != 'Recency' and col != 'Frequency%' else pre_data[col].mean()
        for col in agg_columns
    })
    pre_metrics = pd.DataFrame(weighted_avg_pre).T
    pre_metrics['Customers_Period'] = len(pre_data)

    # Calculate weighted average for every metric except Recency which is average
    weighted_avg_post = pd.Series({
        col: (post_data[col] * post_data['Frequency%']).sum() / post_data['Frequency%'].sum()
        if col in agg_columns and col != 'Recency' and col != 'Frequency%' else post_data[col].mean()
        for col in agg_columns
    })
    post_metrics = pd.DataFrame(weighted_avg_post).T
    post_metrics['Customers_Period'] = len(post_data)

    # Perform one-sample t-test for each metric
    alpha = 0.05
    results = { 'Metrics': [], 'Pre': [], 'Post': [], 'Var': [], 'Var%': [], 'Sign': []}

    columns = ['Transactions', 'Quantity_All','Price_All', 'Sales_All', 'Coupons_All', 'Quantity_Apparel', 'Sales_Apparel', 'Coupons_Apparel','Quantity_Nest-USA', 'Sales_Nest-USA', 'Coupons_Nest-USA','Quantity_Other', 'Sales_Other', 'Coupons_Other', 'Frequency%', 'Recency','Customers_Period']

    for metric in columns:
        if metric == 'Customers_Period':
             #Calculate the difference
            difference = post_metrics.loc[:,metric].values[0] - pre_metrics.loc[:, metric].values[0]

            # Store results
            results['Metrics'].append(f'{metric}')
            results['Pre'].append(pre_metrics.loc[:, metric].values[0])
            results['Post'].append(post_metrics.loc[:, metric].values[0])
            results['Var'].append(difference)
            results['Sign'].append('NA')

            continue

        if metric == 'Frequency%' or metric == 'Recency':
            post_values = post_data[metric]
            pre_weighted_avg = pre_metrics.loc[:, metric].values[0]
            t_stat, p_value = stats.ttest_1samp(post_values, pre_weighted_avg)
        else:
            # Extract data for the segment
            post_values = post_data[metric]
            weights = post_data[:]['Frequency%']

            # Calculate the weighted average for the metric from pre_metrics
            pre_weighted_avg = pre_metrics.loc[:, metric].values[0]

            # Perform one-sample t-test using the weighted average
            t_stat, p_value = weighted_ttest_1samp(post_values, pre_weighted_avg, weights)


        # Calculate the difference
        difference = post_metrics.loc[:, metric].values[0] - pre_metrics.loc[:, metric].values[0]
        difference_percent = difference/pre_metrics.loc[:, metric].values[0]
        # Store results
        results['Metrics'].append(f'{metric}')
        results['Pre'].append(pre_metrics.loc[:, metric].values[0])
        results['Post'].append(post_metrics.loc[:, metric].values[0])
        results['Var'].append(difference)
        results['Sign'].append('Yes' if p_value < alpha else 'No')


    length = len(results['Metrics'])        
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
    
    # Update the value in the cell where Var% is 'Frequency%' and Metrics is in the 'Frequency%' row
    #player_df.loc[player_df['Metrics'] == 'Frequency%', 'Var%'] = 'NA'
    print('hipo_player')
    return player_df


# In[37]:


# Function to calculate days count
def calculate_days_count(group):
    return (group['Transaction_Date'].max() - group['Transaction_Date'].min()).days+1


# In[38]:


def features_by_day(dfa,pre_ini_date,pre_end_date,post_ini_date,post_end_date,PS,IG):
    # Define the date ranges
    pre_ini_date = pd.to_datetime(pre_ini_date)
    pre_end_date = pd.to_datetime(pre_end_date)

    # Define the date ranges
    post_ini_date = pd.to_datetime(post_ini_date)
    post_end_date = pd.to_datetime(post_end_date)
    
    ### Feature Engineering: Frequency and Recency, and New Product ###
    
    agg_columns = ['Transactions', 'Quantity_All','Price_All', 'Sales_All', 'Coupons_All', 'Quantity_Apparel', 'Sales_Apparel', 'Coupons_Apparel','Quantity_Nest-USA', 'Sales_Nest-USA', 'Coupons_Nest-USA','Quantity_Other', 'Sales_Other', 'Coupons_Other']
    metrics_columns = ['Transactions', 'Quantity_All', 'Sales_All', 'Coupons_All']
    # First, let's calculate the mean aggregated columns
         
    if (PS != 'All') & (IG == 'All'):

        dfpre_day =  dfa[(dfa['Period'] == 'Pre')&(dfa['Promo_Segment'] == PS)].groupby(['Transaction_Date'])[agg_columns].mean().reset_index()
        dfpost_day =  dfa[(dfa['Period'] == 'Post')&(dfa['Promo_Segment'] == PS)].groupby(['Transaction_Date'])[agg_columns].mean().reset_index()
        
        active_players_pre = dfa[(dfa['Period'] == 'Pre')&(dfa['Promo_Segment'] == PS)].groupby(['Transaction_Date'])['CustomerID'].count().reset_index(name='ActiveCustomer ')
        active_players_post = dfa[(dfa['Period'] == 'Post')&(dfa['Promo_Segment'] == PS)].groupby(['Transaction_Date'])['CustomerID'].count().reset_index(name='ActiveCustomer')
        
        Metrics_day_Pre = dfa[(dfa['Period'] == 'Pre')&(dfa['Promo_Segment'] == PS)].groupby(['Transaction_Date'])[metrics_columns].sum().reset_index()
        Metrics_day_Post = dfa[(dfa['Period'] == 'Post')&(dfa['Promo_Segment'] == PS)].groupby(['Transaction_Date'])[metrics_columns].sum().reset_index()
    
    
    elif (PS == 'All') & (IG != 'All'):
        
        dfpre_day =  dfa[(dfa['Period'] == 'Pre')&(dfa['Initiative'] == IG)].groupby(['Transaction_Date'])[agg_columns].mean().reset_index()
        dfpost_day =  dfa[(dfa['Period'] == 'Post')&(dfa['Initiative'] == IG)].groupby(['Transaction_Date'])[agg_columns].mean().reset_index()
        
        active_players_pre = dfa[(dfa['Period'] == 'Pre')&(dfa['Initiative'] == IG)].groupby(['Transaction_Date'])['CustomerID'].count().reset_index(name='ActiveCustomer')
        active_players_post = dfa[(dfa['Period'] == 'Post')&(dfa['Initiative'] == IG)].groupby(['Transaction_Date'])['CustomerID'].count().reset_index(name='ActiveCustomer')
        
        Metrics_day_Pre = dfa[(dfa['Period'] == 'Pre')&(dfa['Initiative'] == IG)].groupby(['Transaction_Date'])[metrics_columns].sum().reset_index()
        Metrics_day_Post = dfa[(dfa['Period'] == 'Post')&(dfa['Initiative'] == IG)].groupby(['Transaction_Date'])[metrics_columns].sum().reset_index()
        
        
    elif (PS != 'All') & (IG != 'All'):
    
        dfpre_day =  dfa[(dfa['Period'] == 'Pre')&(dfa['Promo_Segment'] == PS)&(dfa['Initiative'] == IG)].groupby(['Transaction_Date'])[agg_columns].mean().reset_index()
        dfpost_day =  dfa[(dfa['Period'] == 'Post')&(dfa['Promo_Segment'] == PS)&(dfa['Initiative'] == IG)].groupby(['Transaction_Date'])[agg_columns].mean().reset_index()
 
        # Calculate count of active players
        active_players_pre = dfa[(dfa['Period'] == 'Pre')&(dfa['Promo_Segment'] == PS)&(dfa['Initiative'] == IG)].groupby(['Transaction_Date'])['CustomerID'].count().reset_index(name='ActiveCustomer')
        active_players_post = dfa[(dfa['Period'] == 'Post')&(dfa['Promo_Segment'] == PS)&(dfa['Initiative'] == IG)].groupby(['Transaction_Date'])['CustomerID'].count().reset_index(name='ActiveCustomer')
      
        Metrics_day_Pre = dfa[(dfa['Period'] == 'Pre')&(dfa['Promo_Segment'] == PS)&(dfa['Initiative'] == IG)].groupby(['Transaction_Date'])[metrics_columns].sum().reset_index()
        Metrics_day_Post = dfa[(dfa['Period'] == 'Post')&(dfa['Promo_Segment'] == PS)&(dfa['Initiative'] == IG)].groupby(['Transaction_Date'])[metrics_columns].sum().reset_index()
        
   
    elif (PS == 'All') & (IG == 'All'):
        
        dfpre_day =  dfa[(dfa['Period'] == 'Pre')].groupby(['Transaction_Date'])[agg_columns].mean().reset_index()
        dfpost_day =  dfa[(dfa['Period'] == 'Post')].groupby(['Transaction_Date'])[agg_columns].mean().reset_index()

        # Calculate count of active players
        active_players_pre = dfa[(dfa['Period'] == 'Pre')].groupby(['Transaction_Date'])['CustomerID'].count().reset_index(name='ActiveCustomer')
        active_players_post = dfa[(dfa['Period'] == 'Post')].groupby(['Transaction_Date'])['CustomerID'].count().reset_index(name='ActiveCustomer')
        
        Metrics_day_Pre = dfa[(dfa['Period'] == 'Pre')].groupby(['Transaction_Date'])[metrics_columns].sum().reset_index()
        Metrics_day_Post = dfa[(dfa['Period'] == 'Post')].groupby(['Transaction_Date'])[metrics_columns].sum().reset_index()
        
    # Calculate Total ActivePlayers
    total_active_players_pre = active_players_pre['ActiveCustomer'].sum()
    total_active_players_post = active_players_post['ActiveCustomer'].sum()
    
    # Step 2: Merge with the aggregated DataFrames
    dfpre_day = pd.merge(dfpre_day, active_players_pre, on=['Transaction_Date'], how='left')
    dfpre_day['TotalActiveCustomer'] = total_active_players_pre
    dfpre_day['ActiveCustomer%'] = dfpre_day['ActiveCustomer']/dfpre_day['TotalActiveCustomer']
    dfpre_day = dfpre_day.drop('TotalActiveCustomer', axis=1)
    dfpost_day = pd.merge(dfpost_day, active_players_post, on=['Transaction_Date'], how='left')
    dfpost_day['TotalActiveCustomer'] = total_active_players_post
    dfpost_day['ActiveCustomer%'] = dfpost_day['ActiveCustomer']/dfpost_day['TotalActiveCustomer']
    dfpost_day = dfpost_day.drop('TotalActiveCustomer', axis=1)
    
    # Add suffix '_day' to all columns
    Metrics_day_Pre.columns = [f'{col}_day' if col != 'Transaction_Date' else col for col in Metrics_day_Post.columns]
    Metrics_day_Post.columns = [f'{col}_day' if col != 'Transaction_Date' else col for col in Metrics_day_Post.columns]
    
    dfpre_day = pd.merge(dfpre_day, Metrics_day_Pre, on=['Transaction_Date'], how='left')
    dfpost_day = pd.merge(dfpost_day, Metrics_day_Post, on=['Transaction_Date'], how='left')
    
    
    ########################################################## RECENCY CALC ################################
    # Step 3: Calculate the recency per day per player
    
    if (PS != 'All') & (IG == 'All'):
        recency_raw_df= dfa[(dfa['Promo_Segment'] == PS)].copy()
    elif (PS == 'All') & (IG != 'All'):
        recency_raw_df= dfa[(dfa['Initiative'] == IG)].copy()
    elif (PS != 'All') & (IG != 'All'):
        recency_raw_df= dfa[(dfa['Promo_Segment'] == PS)&(dfa['Initiative'] == IG)].copy()
    else:
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
        player_dates = pd.DataFrame({'CustomerID': [row.CustomerID] * len(date_range), 'Transaction_Date': date_range})
        post_date_range_list.append(player_dates)

    post_date_range_df = pd.concat(post_date_range_list, ignore_index=True)

    for row in min_max_datetime_pre.itertuples(index=False):
        date_range = pd.date_range(row.MinDatetime, row.MaxDatetime, name='Transaction_Date')
        player_dates = pd.DataFrame({'CustomerID': [row.CustomerID] * len(date_range), 'Transaction_Date': date_range})
        pre_date_range_list.append(player_dates)

    pre_date_range_df = pd.concat(pre_date_range_list, ignore_index=True)
    
   
    #Merge with other necessary data
    # Merge with other necessary data
    recency_pre_df2 = recency_pre_df.copy()
    recency_pre_df2.loc[:, 'TransactionDate'] = recency_pre_df2['Transaction_Date']

    recency_post_df2 = recency_post_df.copy()
    recency_post_df2.loc[:, 'TransactionDate'] = recency_post_df2['Transaction_Date']

    # Select columns for the left dataframe
    columns_to_merge = ['CustomerID', 'Transaction_Date', 'TransactionDate']
    subset_recency_pre_df = recency_pre_df2[columns_to_merge]
    subset_recency_post_df = recency_post_df2[columns_to_merge]

    pre_date_range_merged_df = pd.merge(pre_date_range_df, subset_recency_pre_df, on=['CustomerID','Transaction_Date'], how='left')
    post_date_range_merged_df = pd.merge(post_date_range_df, subset_recency_post_df, on=['CustomerID','Transaction_Date'], how='left')

    # Fill forward the values
    pre_date_range_merged_df = pre_date_range_merged_df.sort_values(by=['CustomerID', 'Transaction_Date'])
    pre_date_range_merged_df = pre_date_range_merged_df.fillna(method='ffill')
    post_date_range_merged_df = post_date_range_merged_df.sort_values(by=['CustomerID', 'Transaction_Date'])
    post_date_range_merged_df = post_date_range_merged_df.fillna(method='ffill')
    
    pre_date_range_merged_df['Recency'] = (pre_date_range_merged_df['Transaction_Date'] - pre_date_range_merged_df['TransactionDate']).dt.days
    pre_date_range_merged_df['Recency'] = pre_date_range_merged_df['Recency']*-1
    post_date_range_merged_df['Recency'] = (post_date_range_merged_df['Transaction_Date'] - post_date_range_merged_df['TransactionDate']).dt.days
    post_date_range_merged_df['Recency'] = post_date_range_merged_df['Recency']*-1
    #Grouped by Day, Period, Promo_Segment
    df_recency_pre =  pre_date_range_merged_df.groupby(['Transaction_Date'])['Recency'].mean().reset_index()
    df_recency_post =  post_date_range_merged_df.groupby(['Transaction_Date'])['Recency'].mean().reset_index()
    
    #Merging with dfpre and dfpost
    dfpre_day = pd.merge(dfpre_day, df_recency_pre, on=['Transaction_Date'], how='left')
    dfpost_day = pd.merge(dfpost_day, df_recency_post, on=['Transaction_Date'], how='left')
    
    print('feature_day')
    return dfpre_day, dfpost_day


# In[39]:


def hypothesis_testing_days(dfpre_day, dfpost_day):
    # I will be using One-Sample t_Test so we can use a different number of players for Pre and Post period
    # Ho: Promo does nothing vs Ha: Promo does something
    ################################### Per Day ##########################################
    
    # Filter data for the Pre and Post periods
    pre_data = dfpre_day
    post_data = dfpost_day

    # First, let's calculate the mean aggregated columns
    agg_columns = ['Transactions', 'Quantity_All','Price_All', 'Sales_All', 'Coupons_All', 'Quantity_Apparel', 'Sales_Apparel', 'Coupons_Apparel','Quantity_Nest-USA', 'Sales_Nest-USA', 'Coupons_Nest-USA','Quantity_Other', 'Sales_Other', 'Coupons_Other', 'ActiveCustomer', 'Recency']
    
    # Calculate weighted average for every metric except Recency which is average
    weighted_avg_pre = pd.Series({
        col: (pre_data[col] * pre_data['ActiveCustomer%']).sum() / pre_data['ActiveCustomer%'].sum()
        if col in agg_columns and col != 'ActiveCustomer' and col != 'ActiveCustomer%' and col != 'Recency' else pre_data[col].mean()
        for col in agg_columns
    })
    pre_metrics = pd.DataFrame(weighted_avg_pre).T
    pre_metrics['DaysCount_Period'] = len(pre_data)
    pre_metrics['Transactions_Period'] = pre_data['Transactions_day'].sum()
    pre_metrics['Quantity_All_Period'] = pre_data['Quantity_All_day'].sum()
    pre_metrics['Sales_All_Period'] = pre_data['Sales_All_day'].sum()
    pre_metrics['Coupons_All_Period'] = pre_data['Coupons_All_day'].sum()
     
    # Calculate weighted average for every metric except Recency which is average
    weighted_avg_post = pd.Series({
        col: (post_data[col] * post_data['ActiveCustomer%']).sum() / post_data['ActiveCustomer%'].sum()
        if col in agg_columns and col != 'ActiveCustomer' and col != 'ActiveCustomer%' and col != 'Recency' else post_data[col].mean()
        for col in agg_columns
    })
    post_metrics = pd.DataFrame(weighted_avg_post).T
    post_metrics['DaysCount_Period'] = len(post_data)
    post_metrics['Transactions_Period'] = post_data['Transactions_day'].sum()
    post_metrics['Quantity_All_Period'] = post_data['Quantity_All_day'].sum()
    post_metrics['Sales_All_Period'] = post_data['Sales_All_day'].sum()
    post_metrics['Coupons_All_Period'] = post_data['Coupons_All_day'].sum()
  
    # Perform one-sample t-test for each metric
    alpha = 0.05
    results = {'Metrics': [], 'Pre': [], 'Post': [], 'Var': [], 'Var%': [], 'Sign': []}

    columns = ['DaysCount_Period', 'Transactions_Period', 'Quantity_All_Period', 'Sales_All_Period', 'Coupons_All_Period','Transactions', 'Quantity_All','Price_All', 'Sales_All', 'Coupons_All', 'Quantity_Apparel', 'Sales_Apparel', 'Coupons_Apparel','Quantity_Nest-USA', 'Sales_Nest-USA', 'Coupons_Nest-USA','Quantity_Other', 'Sales_Other', 'Coupons_Other', 'ActiveCustomer', 'Recency']

    for metric in columns:
        
        if metric == 'ActiveCustomer' or metric == 'Recency':
            post_values = post_data[metric]
            pre_weighted_avg = pre_metrics.loc[:, metric].values[0]
            t_stat, p_value = stats.ttest_1samp(post_values, pre_weighted_avg)
            results['Sign'].append('Yes' if p_value < alpha else 'No')

        elif metric == 'DaysCount_Period' or metric == 'Transactions_Period' or metric == 'Quantity_All_Period' or metric == 'Sales_All_Period' or metric == 'Coupons_All_Period':
            results['Sign'].append('NA')

        else:
            # Extract data for the segment
            post_values = post_data[metric]
            weights = post_data['ActiveCustomer%']

            # Calculate the weighted average for the metric from pre_metrics
            pre_weighted_avg = pre_metrics.loc[:, metric].values[0]

            # Perform one-sample t-test using the weighted average
            t_stat, p_value = weighted_ttest_1samp(post_values, pre_weighted_avg, weights)

            results['Sign'].append('Yes' if p_value < alpha else 'No')


        # Calculate the difference
        difference = post_metrics.loc[:, metric].values[0] - pre_metrics.loc[:, metric].values[0]
        difference_percent = difference/pre_metrics.loc[:, metric].values[0]

        # Store results
        results['Metrics'].append(f'{metric}')
        results['Pre'].append(pre_metrics.loc[:, metric].values[0])
        results['Post'].append(post_metrics.loc[:, metric].values[0])
        results['Var'].append(difference)


    length = len(results['Metrics'])        
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
    print('hipo_day')
    return day_df


# ### Adding Playing and Day metrics together

# In[40]:


def results(player_df,day_df):
    # Extracting the rows from player_df based on the specified condition
    selected_rows = player_df[player_df['Metrics'].isin(['Frequency%', 'Customers_Period'])]

    # Concatenating the two dataframes vertically
    result_df = pd.concat([day_df, selected_rows], ignore_index=True)

    # If you want to sort the resulting dataframe based on a specific column (e.g., 'Metrics')
    result_df = result_df.sort_values(by='Metrics')

    # Resetting the index of the resulting dataframe
    result_df.reset_index(drop=True, inplace=True)  
    
    # Extracting the rows from player_df based on the specified condition
    selected_rows = player_df[player_df['Metrics'].isin(['Frequency%', 'Customers_Period'])]

    # Concatenating the two dataframes vertically
    result_df = pd.concat([day_df, selected_rows], ignore_index=True)

    # Defining the custom sorting order for the 'Metrics' column
    custom_metric_order = ['DaysCount_Period', 'Transactions_Period', 'Quantity_All_Period', 'Sales_All_Period', 'Coupons_All_Period','Customers_Period',
                           'ActiveCustomer','Transactions','Price_All', 'Quantity_All', 'Quantity_Apparel','Quantity_Nest-USA','Quantity_Other',
                           'Sales_All','Sales_Apparel','Sales_Nest-USA','Sales_Other',
                           'Coupons_All','Coupons_Apparel','Coupons_Nest-USA','Coupons_Other',
                           'Recency'   
    ]

    # Sorting the resulting dataframe by 'Promo_Segment' and then by the custom order of 'Metrics'
    result_df = result_df.sort_values(by=['Metrics'], key=lambda x: x.map({metric: i for i, metric in enumerate(custom_metric_order)}))

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
    is_general = result_with_format_df['Metrics'].isin(['DaysCount_Period', 'Transactions_Period', 'Quantity_All_Period','Coupons_All_Period','Transactions', 'Quantity_All', 'Coupons_All', 'Quantity_Apparel', 'Coupons_Apparel','Quantity_Nest-USA', 'Coupons_Nest-USA','Quantity_Other','Coupons_Other', 'ActiveCustomer', 'Recency', 'Customers_Period', 'Days_Count' ])
    is_currency = result_with_format_df['Metrics'].isin(['Price_All','Sales_All_Period','Sales_All','Sales_Apparel','Sales_Nest-USA','Sales_Other'])
    is_percent = result_with_format_df['Metrics'].isin(['Frequency%'])

    # Apply formatting based on the mask
    result_with_format_df.loc[is_currency, ['Pre', 'Post','Var']] = result_with_format_df.loc[is_currency, ['Pre', 'Post','Var']].applymap(format_currency)
    result_with_format_df.loc[is_general, ['Pre', 'Post','Var']] = result_with_format_df.loc[is_general, ['Pre', 'Post','Var']].applymap(format_general)
    result_with_format_df.loc[is_percent, ['Pre', 'Post','Var']] = result_with_format_df.loc[is_percent, ['Pre', 'Post','Var']].applymap(format_percent)
    print('result')
    return result_df, result_with_format_df


# In[41]:


def data_for_radar_and_table(result_df) :
    data_df = result_df.copy()
    
    data_with_format_df = data_df.copy()

    # Define a formatting function for currency
    format_currency = lambda x: "${:,.0f}".format(x) 

    # Define a formatting function for general
    format_general = lambda x: "{:,.0f}".format(x)

    # Define a formatting function for %
    format_percent = lambda x: "{:.2%}".format(x) 

    # Create a mask for the 'Metrics' column
    is_general = result_with_format_df['Metrics'].isin(['DaysCount_Period', 'Transactions_Period', 'Quantity_All_Period','Coupons_All_Period','Transactions', 'Quantity_All', 'Coupons_All', 'Quantity_Apparel', 'Coupons_Apparel','Quantity_Nest-USA', 'Coupons_Nest-USA','Quantity_Other','Coupons_Other', 'ActiveCustomer', 'Recency', 'Customers_Period', 'Days_Count' ])
    is_currency = result_with_format_df['Metrics'].isin(['Price_All','Sales_All_Period','Sales_All','Sales_Apparel','Sales_Nest-USA','Sales_Other'])
    is_percent = result_with_format_df['Metrics'].isin(['Frequency%'])

    # Apply formatting based on the mask
    data_with_format_df.loc[is_currency, ['Pre', 'Post','Var']] = data_with_format_df.loc[is_currency, ['Pre', 'Post','Var']].applymap(format_currency)
    data_with_format_df.loc[is_general, ['Pre', 'Post','Var']] = data_with_format_df.loc[is_general, ['Pre', 'Post','Var']].applymap(format_general)
    data_with_format_df.loc[is_percent, ['Pre', 'Post','Var']] = data_with_format_df.loc[is_percent, ['Pre', 'Post','Var']].applymap(format_percent)

    # Use the drop method to remove the specified column
    data_with_format_df['Var2'] = data_df['Var']
    
    data_percent_df = data_df.copy()
    data_percent_df = data_percent_df.drop('Var', axis=1)
    data_percent_df = data_percent_df.drop('Var%', axis=1)
    data_percent_df = data_percent_df.drop('Sign', axis=1)
    keep = (data_percent_df['Metrics'] != 'DaysCount_Period') 
    data_percent_df = data_percent_df[keep]
    data_percent_df = data_percent_df.set_index('Metrics')
        
    data_percent_df['Total'] = data_percent_df.sum(axis=1)  # Calculate the total for each row

   
    # Check if 'Metrics' column is equal to 'Recency' or 'WithdrawalAmt'
    is_recency_or_withdrawal = (data_percent_df.index  == 'Recency') 

    # Apply different logic for 'Recency' and 'WithdrawalAmt'
    data_percent_df.loc[is_recency_or_withdrawal, :] = 100 + (data_percent_df.loc[is_recency_or_withdrawal, :]
                                                             .div(np.abs(data_percent_df.loc[is_recency_or_withdrawal, 'Total']), axis=0)
                                                             .mul(100)
                                                             .round(2))

    # Apply the original logic for other metrics
    data_percent_df.loc[~is_recency_or_withdrawal, :] = (data_percent_df.loc[~is_recency_or_withdrawal, :]
                                                         .div(data_percent_df.loc[~is_recency_or_withdrawal, 'Total'], axis=0)
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
    print('radar')
    return data_with_format_df, data_percent_df, min_value, max_value


# In[42]:


def player_table_data (dfa,dfpre_player,dfpost_player,PS,IG):
    # Merge dataframes on the 'Player' and 'CRM' columns using outer join
    merged_df = pd.merge(dfpre_player, dfpost_player, on=['CustomerID'], how='outer', suffixes=('_pre', '_post'))

    # Calculate the differences for the specified columns
    merged_df['Promo_Segment'] = PS
    merged_df['Initiative'] = IG
    merged_df['Sales_All_D'] = merged_df['Sales_All_post'] - merged_df['Sales_All_pre']
    merged_df['Quantity_All_D'] = merged_df['Quantity_All_post'] - merged_df['Quantity_All_pre']
    merged_df['Frequency%_D'] = merged_df['Frequency%_post'] - merged_df['Frequency%_pre']
    merged_df['Recency_D'] = merged_df['Recency_post'] - merged_df['Recency_pre']
    merged_df['Coupons_All_D'] = merged_df['Coupons_All_post'] - merged_df['Coupons_All_pre']
    merged_df['Transactions_D'] = merged_df['Transactions_post'] - merged_df['Transactions_pre']


    # Format columns with 2 decimals
    numeric_columns = ['Transactions_post', 'Quantity_All_post', 'Coupons_All_post', 'Recency_post', 'Sales_All_post',
                      'Sales_All_D','Quantity_All_D','Recency_D','Coupons_All_D','Transactions_D']
    merged_df[numeric_columns] = merged_df[numeric_columns].round(2)

    # Format 'Frequency%' and 'Frequency%_D' as percentages
    merged_df[['Frequency%_post', 'Frequency%_D']] = merged_df[['Frequency%_post', 'Frequency%_D']].apply(lambda x: x * 100).round(2)
    
    # Create the final dataframe with the desired columns
    df_playertable = merged_df[[
        'Promo_Segment', 'Initiative','CustomerID', 
        'Sales_All_post', 'Sales_All_D',
        'Quantity_All_post', 'Quantity_All_D',
        'Frequency%_post', 'Frequency%_D',
        'Recency_post', 'Recency_D',
        'Coupons_All_post', 'Coupons_All_D',
        'Transactions_post', 'Transactions_D',
    ]]
    return df_playertable


# # Analysis Process

# In[43]:


dfa=daily_df.copy()


# In[44]:


dfa.shape


# In[45]:


# Define the date ranges
pre_ini_date = '2019-01-29'
pre_end_date = '2019-04-30'

# Define the date ranges
post_ini_date = '2019-05-01'
post_end_date = '2019-07-31'


# Define Test and Control parameters
Test_PS = 'All'
#Test_IG = 'All'
Test_IG = 'BuyOneGetOne'
Control_PS = 'NoPromo'
Control_IG = 'NoInitiative'


# In[46]:


df_period = pre_post_period(pre_ini_date, pre_end_date, post_ini_date, post_end_date, dfa)


# In[47]:


df_period.shape


# In[48]:


df_segmentation = promo_segmentation(df_period)


# In[49]:


df_segmentation.shape


# In[50]:


df_initiatives = initiatives(df_segmentation,df_initiatives_file)


# In[51]:


df_initiatives.shape


# In[52]:


df_initiatives


# In[53]:


dfpre_player_test, dfpost_player_test = features_by_player(df_initiatives,pre_ini_date,pre_end_date,post_ini_date,post_end_date,Test_PS,Test_IG)


# In[54]:


player_df_test = hypothesis_testing_player(dfpre_player_test, dfpost_player_test)


# In[55]:


dfpre_day_test, dfpost_day_test = features_by_day(df_initiatives,pre_ini_date,pre_end_date,post_ini_date,post_end_date,Test_PS,Test_IG)


# In[56]:


day_df_test = hypothesis_testing_days(dfpre_day_test, dfpost_day_test)


# In[57]:


day_df_test


# In[58]:


player_df_test


# In[59]:


result_df_test, result_with_format_df = results(player_df_test,day_df_test)


# In[60]:


df_playertable = player_table_data(dfa,dfpre_player_test,dfpost_player_test,Test_PS,Test_IG)


# In[61]:


df_playertable


# ### For validation

# In[62]:


dfval_avg=dfa.groupby(['Promo_Segment','Period'])['Transactions', 'Quantity_All', 'Sales_All'].mean().reset_index()
dfval_avg


# In[63]:


dfval_sum=dfa.groupby(['Promo_Segment','Period'])['Transactions', 'Quantity_All', 'Sales_All'].sum().reset_index()
dfval_sum


# # 4. Visualization

# In[64]:


def time_line_chart(df_period,metric):
    df = df_period.groupby(['Transaction_Date','Period']).sum(numeric_only=True).reset_index()
    # Create a line chart
    fig = go.Figure()

    # Add a trace for the line chart with area
    fig.add_trace(go.Scatter(x=df[df['Period'] == 'Post']['Transaction_Date'], y=df[df['Period'] == 'Post'][metric],
                             mode='lines',
                             line=dict(color='#7DBC13'),
                             fill='tozeroy',
                             fillcolor='rgba(75, 120, 168, 0.2)',
                             name='Post'))

    # Add a trace for the line chart with area
    fig.add_trace(go.Scatter(x=df[df['Period'] == 'Pre']['Transaction_Date'], y=df[df['Period'] == 'Pre'][metric],
                             mode='lines',
                             line=dict(color='#9B1831'),
                             fill='tozeroy',
                             fillcolor='rgba(75, 120, 168, 0.2)',
                             name='Pre'))

    # Add a trace for the line chart with area
    fig.add_trace(go.Scatter(x=df[df['Period'] == 'Out']['Transaction_Date'], y=df[df['Period'] == 'Out'][metric],
                             mode='markers',
                             line=dict(color='#D8D8D8'),
                             fillcolor='rgba(75, 120, 168, 0.2)',
                             name='Other'))

    # Update layout
    fig.update_layout(
        yaxis_title=f'{metric}',
        template='simple_white',
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black', range=[df['Transaction_Date'].min(), df['Transaction_Date'].max()]),
        yaxis=dict(gridcolor='lightgrey', tickformat='$,.2s'),
        xaxis_showgrid=True, yaxis_showgrid=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        width=1200,
        height=396,
    )

    return fig


# In[65]:


metric_time_line_chart = 'Sales_All'
figa = time_line_chart(df_period,metric_time_line_chart)
figa.show()


# In[66]:


test_format_df, test_percent_df, test_min_value, test_max_value = data_for_radar_and_table(result_df_test)


# In[67]:


test_format_df


# In[95]:


def radar_chart(df, min_value, max_value, promo_segment, initiative_group, title):

    categories = df['Metrics']
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
          r=df['Pre'],
          theta=categories,
          fill='toself',
          name='Pre',
          marker_color='#9B1831'
    ))
    fig.add_trace(go.Scatterpolar(
          r=df['Post'],
          theta=categories,
          fill='toself',
          name='Post',
          marker_color='#7DBC13'
    ))

    fig.update_layout(
        title_text=f'{title} | Promo: {promo_segment} | Initiative: {initiative_group}',
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
      height=405,
      showlegend=True,
      legend=dict(x=1,y=0,traceorder='normal',orientation='h'),
      font=dict(size=10)
    )
    
    return fig


# In[96]:


figcrm=radar_chart(test_percent_df,test_min_value,test_max_value,Test_PS,Test_IG,'TEST')
figcrm.show()


# In[70]:


dfpre_player_control, dfpost_player_control = features_by_player(df_initiatives,pre_ini_date,pre_end_date,post_ini_date,post_end_date,Control_PS,Control_IG)
player_df_control = hypothesis_testing_player(dfpre_player_control, dfpost_player_control)

dfpre_day_control, dfpost_day_control = features_by_day(df_initiatives,pre_ini_date,pre_end_date,post_ini_date,post_end_date,Control_PS,Control_IG)
day_df_control = hypothesis_testing_days(dfpre_day_control, dfpost_day_control)

result_df_control, result_with_format_df = results(player_df_control,day_df_control)
control_format_df, control_percent_df, control_min_value, control_max_value = data_for_radar_and_table(result_df_control)


# In[71]:


result_with_format_df


# In[97]:


figno=radar_chart(control_percent_df,control_min_value,control_max_value,Control_PS,Control_IG,'CONTROL')
figno.show()


# In[101]:


def test_vs_control_chart(dfpre_day_test,dfpre_day_control,dfpost_day_test,dfpost_day_control,roll,metric):
    test_post_df = dfpost_day_test.loc[:,['Transaction_Date',metric]]
    control_post_df = dfpost_day_control.loc[:,['Transaction_Date',metric]]

    test_pre_df = dfpre_day_test.loc[:,['Transaction_Date',metric]]
    control_pre_df = dfpre_day_control.loc[:,['Transaction_Date',metric]]

    # Calculate the 14-day moving average
    test_post_df[f'{metric}_MA'] = test_post_df[metric].rolling(window=roll).mean()
    control_post_df[f'{metric}_MA'] = control_post_df[metric].rolling(window=roll).mean()

    # Calculate the 14-day moving average
    test_pre_df[f'{metric}_MA'] = test_pre_df[metric].rolling(window=roll).mean()
    control_pre_df[f'{metric}_MA'] = control_pre_df[metric].rolling(window=roll).mean()

    # Create a subplot with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])


    # Add a trace for the 14-day moving average on the primary y-axis
    fig.add_trace(go.Scatter(x=control_post_df['Transaction_Date'], y=control_post_df[f'{metric}_MA'],
                              mode='lines',
                              line=dict(color='#D8D8D8'),
                              name='Control'),
                  secondary_y=True)


    # Add a trace for the 14-day moving average on the secondary y-axis
    fig.add_trace(go.Scatter(x=test_post_df['Transaction_Date'], y=test_post_df[f'{metric}_MA'],
                              mode='lines',
                              line=dict(color='#7DBC13'),
                              name='Test'),
                   )

    # Add a trace for the 14-day moving average on the primary y-axis
    fig.add_trace(go.Scatter(x=control_pre_df['Transaction_Date'], y=control_pre_df[f'{metric}_MA'],
                              mode='lines',
                              line=dict(color='#D8D8D8'),
                              name='Control'),
                  secondary_y=True)


    # Add a trace for the 14-day moving average on the secondary y-axis
    fig.add_trace(go.Scatter(x=test_pre_df['Transaction_Date'], y=test_pre_df[f'{metric}_MA'],
                              mode='lines',
                              line=dict(color='#9B1831'),
                              name='Test'),
                   )

    # Update layout
    fig.update_layout(
        title=f'{metric} (7-days Rollin Average)',
        # xaxis_title='Date',
        yaxis_title=f'Test {metric}',
        yaxis2=dict(title='Control NetWin', overlaying='y', side='right', tickformat=',.2s'),
        template='simple_white',  # Use the 'simple_white' template
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black', range=[dfpre_day_test['Transaction_Date'].min(), dfpost_day_test['Transaction_Date'].max()]),  # Show x-axis at zero
        yaxis=dict(gridcolor='lightgrey', tickformat=',.2s'),  # Set the y-axis grid color
        xaxis_showgrid=True, yaxis_showgrid=True,  # Show the gridlines
        plot_bgcolor='white',  # Set background color to white
        paper_bgcolor='white',  # Set paper color to white
        font=dict(color='black'),  # Set font color to black
        width=1200,
        height=380,
    )
    return fig
    # Show the figure
    


# In[102]:


roll = 14
metric_test_vs_control = 'Sales_All'
fig4 = test_vs_control_chart(dfpre_day_test,dfpre_day_control,dfpost_day_test,dfpost_day_control,roll,metric_test_vs_control)
fig4.show()


# In[103]:


metric_test_vs_control_2='Transactions'
fig5 = test_vs_control_chart(dfpre_day_test,dfpre_day_control,dfpost_day_test,dfpost_day_control,roll,metric_test_vs_control_2)
fig5.show()


# In[76]:


def synthethic_control_day(dfpre_day_test,dfpre_day_control,dfpost_day_control):
     # Extract the last 35 rows from dfpre_day_test and dfpre_day_control
    lastrow_pre_test = dfpre_day_test.tail(35)
    lastrow_pre_control = dfpre_day_control.tail(35)
    
    #drop the values we will not use
    lastrow_pre_test.drop(['Transaction_Date','Transactions_day', 'Quantity_All_day', 'Coupons_All_day','Sales_All_day','ActiveCustomer%','Recency'], axis=1, inplace=True)
    lastrow_pre_control.drop(['Transaction_Date','Transactions_day', 'Quantity_All_day', 'Coupons_All_day','Sales_All_day','ActiveCustomer%','Recency'], axis=1, inplace=True)
    
    # Calculate the average for each value in the last 35 rows
    lastrow_pre_test_avg = lastrow_pre_test.mean()
    lastrow_pre_control_avg = lastrow_pre_control.mean()
    
    #make the test/control proportion
    proportion = lastrow_pre_test_avg/lastrow_pre_control_avg
    proportion.replace([np.inf, -np.inf], 0, inplace=True)
    proportion.fillna(0, inplace=True)
    
    dfpre_day_control_synthethic = dfpre_day_test.copy()
    dfpost_day_control_synthethic = dfpost_day_control.copy()
    
    # Multiplying the DataFrame columns with the corresponding values in the Series
    dfpost_day_control_synthethic[['Transactions', 'Sales_All', 'Sales_Apparel', 'Sales_Nest-USA','Sales_Other','Quantity_All','Quantity_Apparel','Quantity_Nest-USA','Quantity_Other','Coupons_All','Coupons_Apparel','Coupons_Nest-USA','Coupons_Other','ActivePlayers']] = dfpost_day_control_synthethic[['Transactions', 'Sales_All', 'Sales_Apparel', 'Sales_Nest-USA','Sales_Other','Quantity_All',  'Quantity_Apparel','Quantity_Nest-USA','Quantity_Other','Coupons_All','Coupons_Apparel','Coupons_Nest-USA','Coupons_Other','ActivePlayers']].multiply(proportion)
    
    # Multiply specific columns by 'ActivePlayers' and assign the results to new columns
    dfpost_day_control_synthethic['Transactions_day'] = dfpost_day_control_synthethic['Transactions'] * dfpost_day_control_synthethic['ActiveCustomer']
    dfpost_day_control_synthethic['Sales_All_day'] = dfpost_day_control_synthethic['Sales_All'] * dfpost_day_control_synthethic['ActiveCustomer']
    dfpost_day_control_synthethic['Quantity_All_day'] = dfpost_day_control_synthethic['Quantity_All'] * dfpost_day_control_synthethic['ActiveCustomer']
    dfpost_day_control_synthethic['Coupons_All_day'] = dfpost_day_control_synthethic['Coupons_All'] * dfpost_day_control_synthethic['ActiveCustomer']

    print('SyntheticControl-Day')
    
    return dfpre_day_control_synthethic,dfpost_day_control_synthethic


# In[77]:


def real_result(result_df_test,result_df_control):
    # Copy Metrics and Pre columns from result_df_test to result_df_real
    result_df_real = result_df_test[['Metrics', 'Pre']].copy()

    # Create columns Post, Var, and Var% in result_df_real
    result_df_real['Post'] = 0.0
    result_df_real['Var'] = 0.0
    result_df_real['Var%'] = 0.0

    # Calculate var%_test and var%_control
    var_percent_test = result_df_test.set_index('Metrics')['Post'] / result_df_test.set_index('Metrics')['Pre'] - 1
    var_percent_control = result_df_control.set_index('Metrics')['Post'] / result_df_control.set_index('Metrics')['Pre'] - 1

    # Update result_df_real based on conditions
    for metric in result_df_real['Metrics']:
        pre_test = result_df_test.loc[result_df_test['Metrics'] == metric, 'Pre'].values[0]
        post_test = result_df_test.loc[result_df_test['Metrics'] == metric, 'Post'].values[0]
        pre_control = result_df_control.loc[result_df_control['Metrics'] == metric, 'Pre'].values[0]
        post_control = result_df_control.loc[result_df_control['Metrics'] == metric, 'Post'].values[0]

        if post_control == 0 or pre_control == 0:
            result_df_real.loc[result_df_real['Metrics'] == metric, 'Var%'] = result_df_test.loc[result_df_test['Metrics'] == metric, 'Var%'].values[0]
            result_df_real.loc[result_df_real['Metrics'] == metric, 'Var'] = result_df_test.loc[result_df_test['Metrics'] == metric, 'Var'].values[0]
            result_df_real.loc[result_df_real['Metrics'] == metric, 'Post'] = result_df_test.loc[result_df_test['Metrics'] == metric, 'Post'].values[0]
        else:
            result_df_real.loc[result_df_real['Metrics'] == metric, 'Var%'] = (var_percent_test[metric] - var_percent_control[metric])
            result_df_real.loc[result_df_real['Metrics'] == metric, 'Var'] = result_df_real.loc[result_df_real['Metrics'] == metric, 'Pre'] * result_df_real.loc[result_df_real['Metrics'] == metric, 'Var%']
            result_df_real.loc[result_df_real['Metrics'] == metric, 'Post'] = result_df_real.loc[result_df_real['Metrics'] == metric, 'Pre'] + result_df_real.loc[result_df_real['Metrics'] == metric, 'Var']
            result_df_real.loc[result_df_real['Metrics'] == metric, 'Var%'] = (result_df_real.loc[result_df_real['Metrics'] == metric, 'Var%'] * 100).map('{:.2f}%'.format)

    # Copy Sign column from result_df_test to result_df_real
    result_df_real[['Sign']] = result_df_test[['Sign']].copy()
    
    print('RealResult')
    return result_df_real


# In[78]:


def synthethic_control_player(dfpre_player_test,dfpre_player_control,dfpost_player_control,pre_end_date,post_ini_date,dfa,df_initiatives_file,Control_PS,Control_IG,Test_PS,Test_IG):
    
    pre_end_date = pd.to_datetime(pre_end_date)
    post_ini_date = pd.to_datetime(post_ini_date)
    
    # Calculate 35 days before pre_end_date
    pre_ini_date_35 = pre_end_date - pd.Timedelta(days=34)
    print(pre_ini_date_35)
    # Calculate 35 days after post_ini_date
    post_end_date_35 = post_ini_date + pd.Timedelta(days=34)
    print(post_end_date_35)
    df_period_35 = pre_post_period(pre_ini_date_35, pre_end_date, post_ini_date, post_end_date_35, dfa)
    df_segmentation_35 = promo_segmentation(df_period_35)
    df_initiatives_35 = initiatives(df_segmentation_35,df_initiatives_file)
    dfpre_player_control_35, dfpost_player_control_35 = features_by_player(df_initiatives_35,pre_ini_date_35,pre_end_date,post_ini_date,post_end_date_35,consistent_players,Control_PS,Control_IG)
    dfpre_player_test_35, dfpost_player_test_35 = features_by_player(df_initiatives_35,pre_ini_date_35,pre_end_date,post_ini_date,post_end_date_35,consistent_players,Test_PS,Test_IG)

    fraction = len(dfpre_player_test_35)/len(dfpre_player_control_35)
    print(len(dfpre_player_test_35))
    print(len(dfpre_player_control_35))
    print(fraction)
    dfpost_player_control_synthethic =  dfpost_player_control.sample(frac=fraction, replace=True, random_state=45)
    
    dfpre_player_control_synthethic = dfpre_player_test.copy()
    print('SyntheticControl-Player')
    
    return dfpre_player_control_synthethic,dfpost_player_control_synthethic


# In[79]:


real_result_df = real_result(result_df_test,result_df_control)


# In[80]:


real_format_df, real_percent_df, real_min_value, real_max_value = data_for_radar_and_table(real_result_df)


# In[81]:


figreal=radar_chart(real_percent_df,real_min_value,real_max_value,Test_PS,Test_IG,'TEST-REAL')


# In[82]:


figreal.show()


# # 5. Dashboard

# In[83]:


# Initial Settings

# Define the date ranges
pre_ini_date = '2019-01-29'
pre_end_date = '2019-04-30'

# Define the date ranges
post_ini_date = '2019-05-01'
post_end_date = '2019-07-31'

f1_prev_pre_ini_date = pre_ini_date
f1_prev_pre_end_date = pre_end_date
f1_prev_post_ini_date = post_ini_date
f1_prev_post_end_date = post_end_date

f2_pre_ini_date = pre_ini_date
f2_pre_end_date = pre_end_date
f2_post_ini_date = post_ini_date
f2_post_end_date = post_end_date

f3_pre_ini_date = pre_ini_date
f3_pre_end_date = pre_end_date
f3_post_ini_date = post_ini_date
f3_post_end_date = post_end_date


# In[84]:


f1_prev_pre_ini_date


# In[85]:


f1_prev_post_end_date


# In[105]:


# Import packages
import time
import dash
from dash import Dash, dcc, html, dash_table, no_update
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
                            rowData=test_format_df.to_dict('records'),
                            getRowStyle=getRowStyle,
                            columnDefs=[{"autoHeight": True, "field": col} for col in test_format_df.columns if col != 'Var2'],
                            defaultColDef={'sortable': True, 'filter': True, 'selectable': True, 'deletable': True},
                            className='ag-theme-alpine ag-theme-blue_compact',
                            columnSize="autoSize",
                            style={"height": 420, "width": 700},
                            #dashGridOptions={"domLayout": "autoHeight"},
                            )

data_table_control = dag.AgGrid(id="data_table_control",
                                rowData=control_format_df.to_dict('records'),
                                getRowStyle=getRowStyle,
                                columnDefs=[{"autoHeight": True,"field": col}for col in control_format_df.columns if col != 'Var2'],
                                defaultColDef={'sortable': True, 'filter': True, 'selectable': True, 'deletable': True},
                                className='ag-theme-alpine ag-theme-blue_compact',
                                columnSize="autoSize",
                                style={"height": 420, "width": 720},
                                #dashGridOptions = {"domLayout": "autoHeight"},
                            )
playertable = dag.AgGrid(id="data_playertable",
                        rowData=df_playertable.to_dict('records'),
                        columnDefs=[{"headerName": col, "field": col}for col in df_playertable.columns                        ],
                        defaultColDef={'sortable': True, 'filter': True, 'selectable': True, 'deletable': True},
                        columnSize="autoSize",
                        style={"height": 605, "width": '100%'},
                        className='ag-theme-alpine ag-theme-blue',
                        )

data_table_result = dag.AgGrid(id="data_table_result",
                                rowData=real_format_df.to_dict('records'),
                                getRowStyle=getRowStyle,
                                columnDefs=[{"autoHeight": True,"field": col}for col in real_format_df.columns if col != 'Var2'],
                                defaultColDef={'sortable': True, 'filter': True, 'selectable': True, 'deletable': True},
                                className='ag-theme-alpine ag-theme-blue_compact',
                                columnSize="autoSize",
                                style={"height": 510, "width": 720},
                                #dashGridOptions = {"domLayout": "autoHeight"},
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
                              data=['Sales_All','Quantity_All','Coupons_All','Transactions'],
                              value="Sales_All")

#promo_unique_values = df_initiatives['Promo_Segment'].unique()
#promo_unique_values = np.insert(promo_unique_values, 0, 'All')
initiative_unique_values = df_initiatives['Initiative'].unique()
initiative_unique_values = np.insert(initiative_unique_values, 0, 'All')

test_promo_dropdown = dmc.Select(id='test_promo_segment',
                           data=['All','NoPromo','PromoPre','PromoPost'],
                           label="TEST Promo-Segment",
                           value="All")

test_initiative_dropdown = dmc.Select(id='test_initiative_group',
                           data=[{'label': val, 'value': val} for val in initiative_unique_values],
                           label="TEST Initiative-Group",
                           value="BuyOneGetOne")

control_promo_dropdown = dmc.Select(id='control_promo_segment',
                              data=['All','NoPromo','PromoPre','PromoPost'],
                              label="CONTROL Promo-Segment",
                              value="NoPromo")

control_initiative_dropdown = dmc.Select(id='control_initiative_group',
                              data=[{'label': val, 'value': val} for val in initiative_unique_values],
                              label="CONTROL Initiative-Group",
                              value="NoInitiative")
synthetic_control_checkbox = dmc.Checkbox(id="synthetic_control", label="Synthetic Control Group", mb=10)

post_columns = dfpost_day_test.select_dtypes(include=['number']).columns.tolist()
post_test_vs_control1 = dmc.Select(id='post_test_vs_control1', 
                                  label="Metric Selection",
                                  data=[{'label': col, 'value': col} for col in post_columns],
                                  value="Sales_All")
roll1 = dmc.Select(id='roll1', 
                   label="Rollin Average",
                   data=[{'label': '1 day', 'value': 1},
                        {'label': '7 days', 'value': 7},
                        {'label': '14 days', 'value': 14}],
                   value=7)
post_test_vs_control2 = dmc.Select(id='post_test_vs_control2', 
                                  label="Metric Selection",
                                  data=[{'label': col, 'value': col} for col in post_columns],
                                  value="Transactions")
roll2 = dmc.Select(id='roll2',
                   label="Rollin Average",
                   data=[{'label': '1 day', 'value': 1},
                        {'label': '7 days', 'value': 7},
                        {'label': '14 days', 'value': 14}],
                   value=7)

# Figures
graph1 = dcc.Graph(id='figure1', figure=figa)
graph2 = dcc.Graph(id='figure2', figure=figcrm)
graph3 = dcc.Graph(id='figure3', figure=figno)
graph4 = dcc.Graph(id='figure4', figure=fig4)
graph5 = dcc.Graph(id='figure5', figure=fig5)
graph7 = dcc.Graph(id='figure7', figure=figreal)

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
                                        html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
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
                                    test_promo_dropdown,
                                    html.Br(),
                                    test_initiative_dropdown,
                                    html.Br(),
                                    html.Br(),
                                    dmc.Text("Initiative-Groups", 
                                            weight=500,
                                            #size="sm",
                                            #color="dimmed",
                                           ),
                                    html.Br(),
                                    control_promo_dropdown,  
                                    html.Br(),
                                    control_initiative_dropdown,  
                                    html.Br(),
                                    synthetic_control_checkbox,
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                    ],
                            withBorder=True,
                            shadow="sm",
                            radius="md",
                            style={"width": '100%'},
                        )
card_content_2_2 = dmc.Card(children=[
                                      dmc.Center(dmc.LoadingOverlay(id="loading-2",
                                                                    loaderProps={"color": "orange"},
                                                                    children=[graph2])),
                                      dmc.Center(dmc.LoadingOverlay(id="loading-3",
                                                                    loaderProps={"color": "orange"},
                                                                    children=[data_table_test])),
                                      ],
                            withBorder=True,
                            shadow="sm",
                            radius="md",
                            style={"width": '100%'},
                        )
card_content_2_3 = dmc.Card(children=[
                                      dmc.Center(dmc.LoadingOverlay(id="loading-4",
                                                                    loaderProps={"color": "orange"},
                                                                    children=[graph3])),
                                      dmc.Center(dmc.LoadingOverlay(id="loading-5",
                                                                    loaderProps={"color": "orange"},
                                                                    children=[data_table_control])),
                                      ],
                            withBorder=True,
                            shadow="sm",
                            radius="md",
                            style={"width": '100%'},
                        )
card_content_3_1 = dmc.Card(children=[dmc.Text("Graph 1", 
                                            weight=500,
                                            #size="sm",
                                            #color="dimmed",
                                           ),
                                     html.Br(),
                                     dmc.Center(post_test_vs_control1),
                                     html.Br(),
                                     dmc.Center(roll1),
                                     html.Br(),
                                     html.Br(),
                                     dmc.Text("Graph 2", 
                                            weight=500,
                                            #size="sm",
                                            #color="dimmed",
                                           ),
                                     html.Br(),
                                     dmc.Center(post_test_vs_control2),
                                     html.Br(),
                                     dmc.Center(roll2),
                                     html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                     html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                     html.Br(),html.Br(),html.Br(),html.Br(),
                                    ],
                            withBorder=True,
                            shadow="sm",
                            radius="md",
                            style={"width": '100%'},
                        )

card_content_3_2 = dmc.Card(children=[dmc.Center(dmc.LoadingOverlay(id="loading-6",
                                                                    loaderProps={"color": "orange"},
                                                                    children=[graph4])),
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
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                    html.Br(),html.Br(),
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

card_content_5_1 = dmc.Card(children=[dmc.Text("REAL CHANGE", 
                                            weight=500,
                                            #size="sm",
                                            #color="dimmed",
                                           ),
                                      html.Br(),
                                    
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),  
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                    html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                    ],
                            withBorder=True,
                            shadow="sm",
                            radius="md",
                            style={"width": '100%'},
                        )

card_content_5_2 = dmc.Card(children=[
                                      dmc.Center(dmc.LoadingOverlay(id="loading-8",
                                                                    loaderProps={"color": "orange"},
                                                                    children=[graph7])),
                                      dmc.Center(dmc.LoadingOverlay(id="loading-9",
                                                                    loaderProps={"color": "orange"},
                                                                    children=[data_table_result])),
                                      ],
                            withBorder=True,
                            shadow="sm",
                            radius="md",
                            style={"width": '100%'},
                        )


# App Layout
app.layout = dmc.Container(
    [
                html.Div(id='date_change_output', style={'display': 'none'}),
                html.Div(id='test_change_output', style={'display': 'none'}),
                html.Div(id='control_change_output', style={'display': 'none'}),
                dmc.Loader(color="red", size="md", variant="oval"),
                dmc.Header(
                    height=100,
                    fixed=True,
                    pl=0,
                    pr=0,
                    pt=0,
                    style = {'background-color':'white'},
                    children=[dmc.Grid(children=[dmc.Col([html.Br(),dmc.Image(src="/assets/logo.png",width=80,height=50, style={'float': 'left'})],span=2),
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
                dmc.Grid(children=[dmc.Col(card_content_5_1, span=2),
                                   dmc.Col(card_content_5_2, span=10)]),
            ],
    size=2000,
    bg="#F4F5F5",
)
df_initiatives_f1 = df_initiatives.copy()
df_period_f1 = df_period.copy()
date_change_f1 = 'callba2and3cktrigger'
metric_time_line_chart_f1 = metric_time_line_chart

@app.callback(
    Output(component_id='figure1', component_property='figure'),
    Output(component_id='date_change_output', component_property='children'),
    Input(component_id='pre_date_range', component_property='value'),
    Input(component_id='post_date_range', component_property='value'),
    Input(component_id='metrics', component_property='value'),
    prevent_initial_call=True,
)
def function1(
            pre_range, post_range, metric
           ):
    
    global df_initiatives_f1, df_period_f1, date_change_f1, metric_time_line_chart_f1
    global f1_prev_pre_ini_date, f1_prev_pre_end_date, f1_prev_post_ini_date, f1_prev_post_end_date
    global pre_ini_date, pre_end_date, post_ini_date, post_end_date
    
    if (pre_range is not None and post_range is not None):
        
        pre_ini_date_f1,pre_end_date_f1 = pre_range
        post_ini_date_f1,post_end_date_f1 = post_range
    
        if (pre_ini_date_f1 is not None and pre_end_date_f1 is not None
            and post_ini_date_f1 is not None and post_end_date_f1 is not None
            and (pre_ini_date_f1 != f1_prev_pre_ini_date
            or pre_end_date_f1 != f1_prev_pre_end_date
            or post_ini_date_f1 != f1_prev_post_ini_date
            or post_end_date_f1 != f1_prev_post_end_date)):

            f1_prev_pre_ini_date = pre_ini_date_f1
            f1_prev_pre_end_date = pre_end_date_f1
            f1_prev_post_ini_date = post_ini_date_f1
            f1_prev_post_end_date = post_end_date_f1

            df_period_f1 = pre_post_period(pre_ini_date_f1, pre_end_date_f1, post_ini_date_f1, post_end_date_f1, dfa)
            df_segmentation_f1 = promo_segmentation(df_period_f1)
            df_initiatives_f1 = initiatives(df_segmentation_f1,df_initiatives_file)
            
            fig1 = time_line_chart(df_period_f1,metric_time_line_chart_f1)
            return fig1, date_change_f1 
        
    metric_time_line_chart_f1 = metric  
    fig1 = time_line_chart(df_period_f1,metric_time_line_chart_f1)
    return fig1, dash.no_update        


dfpre_player_test_f2 = dfpre_player_test.copy()
dfpost_player_test_f2 = dfpost_player_test.copy()
dfpre_day_test_f2 = dfpre_day_test.copy()
dfpost_day_test_f2 = dfpost_day_test.copy()
result_df_test_f2 = result_df_test.copy()
test_change_f2 = 'triggerscallback4and5and6'
test_ps_f2 = Test_PS
test_ig_f2 = Test_IG


@app.callback(
    Output(component_id='figure2', component_property='figure'),
    Output(component_id='data_table_test', component_property='rowData'),
    Output(component_id='test_change_output', component_property='children'),
    Input(component_id='test_promo_segment', component_property='value'),
    Input(component_id='test_initiative_group', component_property='value'),
    Input(component_id='date_change_output', component_property='children'),
    prevent_initial_call=True

)
def function2(
            test_ps, test_ig, date_change
           ):
    print('F2')
    global dfpre_player_test_f2, dfpost_player_test_f2
    global dfpre_day_test_f2, dfpost_day_test_f2
    global test_ps_f2, test_ig_f2
    global result_df_test_f2

    global f1_prev_pre_ini_date, f1_prev_pre_end_date, f1_prev_post_ini_date, f1_prev_post_end_date
    global f2_pre_ini_date,f2_pre_end_date,f2_post_ini_date,f2_post_end_date
    
    global test_change_f2

    df_initiatives_f2 = df_initiatives_f1.copy()
   
    test_ps_f2 = test_ps
    test_ig_f2 = test_ig
    
    f2_pre_ini_date = f1_prev_pre_ini_date 
    f2_pre_end_date = f1_prev_pre_end_date 
    f2_post_ini_date = f1_prev_post_ini_date 
    f2_post_end_date = f1_prev_post_end_date 
    
    dfpre_player_test_f2, dfpost_player_test_f2 = features_by_player(df_initiatives_f2,f2_pre_ini_date,f2_pre_end_date,f2_post_ini_date,f2_post_end_date,test_ps,test_ig)
    player_df_test_f2 = hypothesis_testing_player(dfpre_player_test_f2, dfpost_player_test_f2)

    dfpre_day_test_f2, dfpost_day_test_f2 = features_by_day(df_initiatives_f2,f2_pre_ini_date,f2_pre_end_date,f2_post_ini_date,f2_post_end_date,test_ps,test_ig)
    day_df_test_f2 = hypothesis_testing_days(dfpre_day_test_f2, dfpost_day_test_f2)

    result_df_test_f2, result_with_format_df_f2 = results(player_df_test_f2,day_df_test_f2)
        
    test_format_df_f2, test_percent_df_f2, test_min_value_f2, test_max_value_f2 = data_for_radar_and_table(result_df_test_f2)
    
    fig2 = radar_chart(test_percent_df_f2,test_min_value_f2,test_max_value_f2,test_ps,test_ig,'TEST')
    
    return fig2, test_format_df_f2.to_dict('records'), test_change_f2

dfpre_player_control_f3 = dfpre_player_control.copy()
dfpost_player_control_f3 = dfpost_player_control.copy()
dfpre_day_control_f3 = dfpre_day_control.copy()
dfpost_day_control_f3 = dfpost_day_control.copy()
player_df_control_f3 = player_df_control.copy()
result_df_control_f3 = result_df_control.copy()
control_change_f3 = 'triggerscallback4and5'

@app.callback(
    Output(component_id='figure3', component_property='figure'),
    Output(component_id='data_table_control', component_property='rowData'),
    Output(component_id='control_change_output', component_property='children'),
    Input(component_id='control_promo_segment', component_property='value'),
    Input(component_id='control_initiative_group', component_property='value'),
    Input(component_id='date_change_output', component_property='children'),
    Input(component_id='synthetic_control', component_property="checked"),
    prevent_initial_call=True

)
def function3(
            control_ps, control_ig, date_change, checked
           ):
    
    print('F3')
    global dfpre_player_control_f3, dfpost_player_control_f3
    global dfpre_day_control_f3, dfpost_day_control_f3
    global dfpre_day_test_f2, player_df_control_f3 
    global dfpre_player_test_f2
    global result_df_control_f3,dfa,df_initiatives_file,test_ps_f2,test_ig_f2

    global f1_prev_pre_ini_date, f1_prev_pre_end_date, f1_prev_post_ini_date, f1_prev_post_end_date
    global f3_pre_ini_date,f3_pre_end_date,f3_post_ini_date,f3_post_end_date
    
    global control_change_f3
    
    if checked == True:
        dfpre_day_control_synthethic, dfpost_day_control_synthethic = synthethic_control_day(dfpre_day_test_f2,dfpre_day_control_f3,dfpost_day_control_f3)
        dfpre_day_control_f3 = dfpre_day_control_synthethic.copy()
        dfpost_day_control_f3 = dfpost_day_control_synthethic.copy()
        
        dfpre_player_control_synthethic,dfpost_player_control_synthethic = synthethic_control_player(dfpre_player_test_f2,dfpre_player_control_f3,dfpost_player_control_f3,f3_pre_end_date,f3_post_ini_date,dfa,df_initiatives_file,consistent_players,control_ps,control_ig,test_ps_f2,test_ig_f2)
        dfpre_player_control_f3 = dfpre_player_control_synthethic.copy()
        dfpost_player_control_f3 = dfpost_player_control_synthethic.copy()
    
    else:
        df_initiatives_f3 = df_initiatives_f1.copy()

        f3_pre_ini_date = f1_prev_pre_ini_date 
        f3_pre_end_date = f1_prev_pre_end_date 
        f3_post_ini_date = f1_prev_post_ini_date 
        f3_post_end_date = f1_prev_post_end_date 

        dfpre_player_control_f3, dfpost_player_control_f3 = features_by_player(df_initiatives_f3,f3_pre_ini_date,f3_pre_end_date,f3_post_ini_date,f3_post_end_date,control_ps,control_ig)
   
        dfpre_day_control_f3, dfpost_day_control_f3 = features_by_day(df_initiatives_f3,f3_pre_ini_date,f3_pre_end_date,f3_post_ini_date,f3_post_end_date,control_ps,control_ig)
    
    player_df_control_f3 = hypothesis_testing_player(dfpre_player_control_f3, dfpost_player_control_f3)
    
    day_df_control_f3 = hypothesis_testing_days(dfpre_day_control_f3, dfpost_day_control_f3)
    
    result_df_control_f3, result_with_format_df_f3 = results(player_df_control_f3,day_df_control_f3)
    
    control_format_df_f3, control_percent_df_f3, control_min_value_f3, control_max_value_f3 = data_for_radar_and_table(result_df_control_f3)
    
    fig3 = radar_chart(control_percent_df_f3,control_min_value_f3,control_max_value_f3,control_ps,control_ig,'CONTROL')
    
    return fig3, control_format_df_f3.to_dict('records'), control_change_f3
        
        
@app.callback(
    Output(component_id='figure4', component_property='figure'),
    Input(component_id='post_test_vs_control1', component_property='value'),
    Input(component_id='roll1', component_property='value'),
    Input(component_id='test_change_output', component_property='children'),
    Input(component_id='control_change_output', component_property='children'),
    prevent_initial_call=True
)
def function4(
            column_1, roll_1, test_change, control_change
           ):
    print('F4')
    global dfpre_day_test_f2, dfpost_day_test_f2
    global dfpre_day_control_f3, dfpost_day_control_f3

    fig4 = test_vs_control_chart(dfpre_day_test_f2,dfpre_day_control_f3,dfpost_day_test_f2,dfpost_day_control_f3,roll_1,column_1)

    return fig4

@app.callback(
    Output(component_id='figure5', component_property='figure'),
    Input(component_id='post_test_vs_control2', component_property='value'),
    Input(component_id='roll2', component_property='value'),
    Input(component_id='test_change_output', component_property='children'),
    Input(component_id='control_change_output', component_property='children'),
    prevent_initial_call=True

)
def function5(
            column_2,roll_2, test_change, control_change
           ):
    
    print('F5')
    global dfpre_day_test_f2, dfpost_day_test_f2
    global dfpre_day_control_f3, dfpost_day_control_f3
    
    fig5 = test_vs_control_chart(dfpre_day_test_f2,dfpre_day_control_f3,dfpost_day_test_f2,dfpost_day_control_f3,roll_2,column_2)
    
    return fig5

@app.callback(
    Output(component_id='data_playertable', component_property='rowData'),
    Input(component_id='test_change_output', component_property='children'),
    prevent_initial_call=True

)
def function6(
            test_change
           ):
    print('F6')
    global dfa,dfpre_player_test_f2, dfpost_player_test_f2
    global test_ig_f2, test_ps_f2

    df_playertable = player_table_data(dfa,dfpre_player_test_f2,dfpost_player_test_f2,test_ps_f2,test_ig_f2)
   
    return df_playertable.to_dict('records')
def function7(
            test_change,control_change
           ):
    print('F7')
    global result_df_test_f2, result_df_control_f3
    global test_ig_f2, test_ps_f2

    real_result_df = real_result(result_df_test_f2,result_df_control_f3)
    real_format_df, real_percent_df, real_min_value, real_max_value = data_for_radar_and_table(real_result_df)
    fig7=radar_chart(real_percent_df,real_min_value,real_max_value,test_ps_f2,test_ig_f2,'TEST-REAL')
    return fig7, real_format_df.to_dict('records')
# Run the App
if __name__ == "__main__":
#    app.run_server(debug=False)
    app.run_server(jupyter_mode="external")


# In[ ]:





# In[ ]:




