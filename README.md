Performance Measurement

website: https://performance-v09.onrender.com/

Notebooks structure:


i) Libraries used:

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

# Import packages
import time
import dash
from dash import Dash, dcc, html, dash_table, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_ag_grid as dag
import dash_mantine_components as dmc
import dash_html_components as html


ii) Data Wrangling

import Initiatives.csv files with the list of initiatives/marketing campaigns.
import Online_Sales.csv with the transactional data.

Feature Engineering Online_Sales

 i)Sales = Quantity * Avg_Price
 ii)Year = Transaction_Date.year
 iii)Month = Transaction_Date.month
 iv)Coupons = Coupon_Status == Used, 1, 0
	 
Created a dataframe pivot_df by day by customer by transactions pivoting Online_Sales,
and created metrics for the 2 most important product categories apparel, Nest-USA, and the rest in Other.

Feature Engineering pivot_df

 i)Quatity_All = Quantity_Apparel + Quantity_Net-USA + Quantity_Other
 the same for:
 ii)Sales_All
 iii)Coupons_All
 iv)Price_All = Sales_All / Quantity_All

created a daily_df by customer by day.

	 aggregating:
	 i) Transaction_ID: nunique
	 ii) Quantity_All: sum
	 iii) Price_All: mean
	 iv) Sales_All: sum
	 v) Coupons_All: sum
	 vi) Quantity_Apparel: sum
	 vii) Sales_Apparel: sum
	 viii) Cupons_Apparel: sum
	 ix) Quantity_Nest-USA: sum
	 x) Sales_Nest-USA: sum
	 xi) Coupons_Nest-USA: sum
	 xii) Quantity_Other: sum
	 xiii) Sales_Other: sum
	 xiv) Coupons_Other: sum

1) Data Exploration

Validated total SALES amt.
Visualized total SALES amt per year.
Visualized total SALES amt per day.


2) Functions Definitions

all functions will be adding information to the daily_df dataframe or transforming it.

2.1) pre_post_period: Create a column with "pre" and "post" depending on the date timeframes selected. We will need this function output for the promo_segmentation function.
		Inputs: pre_ini_date, pre_end_date, post_ini_date, post_end_date, df
		Output: df

2.2) promo_segmentation: creates a column with the status of any coupon used for every pre and post period. We will need this function output for the initiatives function.
		Inputs: df
		Output: df

2.3) initiatives: creates a column with the initiatives status for each player.We will need this function output for the features_by_player and features_by_day functions.
		Inputs: df, df_initiatives
		Outputs: df

2.4) weighted_ttest_1samp: calculates a hypothesis test with weighted average parameters. We will use this function inside the hypothesis testing functions.
		Inputs: list, value, list
		Outputs: value, value

2.5) features_by_player: aggregates the df by player and splits it into pre and post df, also feature engineer frequency and recency. 
We will need this function output for the hypothesis_testing_player function.
		Inputs: df, pre_ini_date,pre_end_date,post_ini_date,post_end_date, promo_segment,initiative_group
		Outputs: df with pre data by player, df with post data by player

2.6) hypothesis_testing_player: Aggregate the dfpre_player and dfpost_player and creates the Customers_Period metric. 
The result is a very summarized df with the comparisson of the pre and post period metrics.
		Inputs: dfpre_player, dfpost_player
		Outputs: player_df

2.7) features_by_day: aggregates the df by day and splits it into pre and post df, also feature engineer ActiveCustomer, Transactions_day, Quantity_All_day, Sales_All_day, Coupons_All_day, frequency and recency. 
We will need this function output for the hypothesis_testing_player function.
		Inputs: df, pre_ini_date,pre_end_date,post_ini_date,post_end_date, promo_segment,initiative_group
		Outputs: df with pre data by player, df with post data by player

2.8) hypothesis_testing_day: Aggregate the dfpre_day and dfpost_day and creates the DaysCount_Period, Transactions_Period, Quantity_All_Period, Sales_All_Period, Coupons_All_Period metric. 
The result is a very summarized df with the comparisson of the pre and post period metrics.
		Inputs: dfpre_day, dfpost_day
		Outputs: day_df

2.9) results: Create the final table with the pre post comparisson combinning 2 metrics from the player_df (frequency and Customers_Period) to all the metrics in the day_df.
		Inputs: player_df, day_df
		Outputs: result_df

2.10) data_for_radar_and_table: from the result_df, it create one table with % proportions based on the pre post total for the radar map and a min and max value for the radat map. In addition to nother table with the same data that result_df already has but wit format.
		Inputs: result_df
		Outputs: data_with_format_df, data_percent_df, min_value, max_value

2.11) player_table_data: creates a table with the post_test data per player with added columns that have the difference between pre and post.
		Inputs: dfa, df_pre_player, dfpost_player, promo_segment, initiative_group
		Outputs: df_playertable

2.12) time_line_chart: it creates the first graph that shows all the data in the databse and highlights the desired pre and post periods.
		Inputs: df_period, metrics
		Outputs: fig

2.13) radar_chart: 	it creates the radar maps.
		Inputs: percent_df, min_value, max_value, promo_Segment, initiative_group, title
		result: fig

2.14) test_vs_control_chart: it creates the line graphs with the pre post test vs control.
		Inputs: dfpre_day_test, dfpre_day_control, dfpost_day_test, dfpost_day_control, roll, metric.
		Result: fig

2.15) synthetic_control_day: it creates a synthetic control group by making the pre_control = pre_test and post_control first 35 days to be proportional to pre_test last 35 days.
		Inputs: dfpre_day_test, dfpre_day_control, dfpost_day_control
		Result: dfpre_day_control_synthetic, dfpost_day_control_synthetic

2.16) synthetic_control_player: it creates a synthetic control group by making the pre_control = pre_test and post_control same proportional sample of players than post_test.
		Inputs: dfpre_player_test, dfpre_player_control, dfpost_player_control, per_end_date, post_ini_date, dfa, df_initiatives_file, control_promo_segment, control_initiative_group, test_promo_segment, test_initiative_group
		Result: dfpre_player_control_synthetic, dfpost_player_control_synthetic

2.17) real_results: calculates the real change of the pre_test_values by using the difference between the test changes and control changes.
		Inputs: result_df_test, result_df_control
		Result: result_df_real


3) DASHBOARD

It uses dash.mantine.components, dash_ag_grid, and dash_html_components.
and it has 7 callbacks for each different group of graphs and tables.

3.1) Callback 1 is for the time-frame graph.
3.2) Callback 2 is for the test-radar map.
3.3) Callback 3 is for the control_radar map.
3.4) Callback 4 is for the first test_vs_control line chart
3.5) Callback 5 is for the second test_vs_control line chart
3.6) Callback 6 is for the player_table
3.7) Callback 7 is for the REAL_RESULT radar map.





