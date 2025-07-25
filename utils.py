
import pandas as pd
import numpy as np

from IPython.display import display
import ipywidgets as widgets
from ipywidgets import interact
from ipywidgets import interact_manual
import numpy as np
import pandas as pd
import seaborn as sns

from itertools import combinations
import utils
import json


#------------------------------------------------------------------------------------------


#ctr
def click_through_rate(clicks,impressions):
    '''
    click_through_rate(clicks,impressions)
    returns a percentage
    '''
    if impressions==0: return 0.0
    return clicks/impressions*100
#cr
def conversion_rate(approved_conversion,total_conversions):
    '''
    conversion_rate(approved_conversion,total_conversions)
    returns a ratio
    '''
    if total_conversions==0: return 0.0
    return approved_conversion/total_conversions
#click to lead rate
def click_to_lead_rate(clicks,leads):
    '''
    click_to_lead_rate(clicks,leads) where leads are 'Total_Conversion'
    returns a ratio
    '''
    if clicks==0: return 0.0
    return leads/clicks
#cpc
def cost_per_click(spent,clicks):
    '''
    cost_per_click(spent,clicks)
    returns cost per 1 click
    '''
    if clicks==0: return 0.0
    return spent/clicks
# cpc
def cost_per_conversion(spent,approved_conversion):
    '''
    cost_per_conversion(spent,approved_conversion)
    returns cost per 1 conversion
    '''
    if approved_conversion==0: return 0.0
    return spent/approved_conversion
# cv
def conversion_value(approved_conversions,avg_profit_each):
    '''
    conversion_value(approved_conversions,avg_profit_each):
    returns total profit from conversions
    '''
    return approved_conversions*avg_profit_each
# ROAS
def return_on_ad_spend(conversion_value,spent):
    '''
    return_on_ad_spend(conversion_value,spent)
    returns conversion_value / spend
    '''
    if spent==0: return 0
    return conversion_value/spent
#cpm
def cost_per_mille(spent,impressions):
    '''
    cost_per_mille(spent,impressions)
    returns cost per thousand impressions
    '''
    if impressions<=0: return 0.0
    return spent/impressions*1000
#Revenue Per Mille (RPM)
def rev_per_mille(Conversion_Value,Impressions):
    '''
    Where if conversion value == 0: if impressions>500: return 0.001, else 0.01
    rev_per_mille(Conversion_Value,Impressions)
    returns the revenue per thousand impressions
    total_revenue/impressions/1000
    '''
    if Impressions<=0: return 0.0
    return Conversion_Value/Impressions*1000

#----------------------------------------------------------------------------------------------------------------------------

def add_metrics(data):
    data=data.replace([np.inf,-np.inf],np.nan).dropna(axis=0)
    data.drop_duplicates(keep='first',inplace=True)
    for _ in data.columns[:6]:
        if data[_].nunique()>500:
            data=data.drop([_],axis=1)
    data['xyz_campaign_id'] = data['xyz_campaign_id'].replace((1178, 936, 916),('campaign_c','campaign_b','campaign_a'))
    data['ClickThroughRate']=data.apply(lambda x: click_through_rate(x['Clicks'],x['Impressions']),axis=1)
    data['ConversionRate']=data.apply(lambda x: conversion_rate(x['Approved_Conversion'],x['Total_Conversion']),axis=1)
    data['CostPerClick']=data.apply(lambda x: cost_per_click(x['Spent'],x['Clicks']),axis=1)
    data['CostPerConversion']=data.apply(lambda x: cost_per_conversion(x['Spent'],x['Approved_Conversion']),axis=1)
    data['Conversion_Value']=data.apply(lambda x: conversion_value(x['Approved_Conversion'],100),axis=1)
    data['ROAS']=data.apply(lambda x: return_on_ad_spend(x['Conversion_Value'],x['Spent']),axis=1)
    data['CostPerMille']=data.apply(lambda x: cost_per_mille(x['Spent'],x['Impressions']),axis=1)
    data['RevenuePerMille']=data.apply(lambda x: rev_per_mille(x['Conversion_Value'],x['Impressions']),axis=1)
    data['ClickToLeadRate']=data.apply(lambda x: click_to_lead_rate(x['Clicks'],x['Total_Conversion']),axis=1)

    # a hot cold column for whether or not a conversion was free
    data['isFree']=data.apply(lambda x: 1 if x['Spent']<=0 and x['Approved_Conversion'] >=1 else 0,axis=1)

    return data

#-----------------------------------------------------------------------------------------------------------------------------
def add_metrics2(data):
   
    
    data['ClickThroughRate']=data.apply(lambda x: click_through_rate(x['Clicks'],x['Impressions']),axis=1)
    data['ConversionRate']=data.apply(lambda x: conversion_rate(x['Approved_Conversion'],x['Total_Conversion']),axis=1)
    data['CostPerClick']=data.apply(lambda x: cost_per_click(x['Spent'],x['Clicks']),axis=1)
    data['CostPerConversion']=data.apply(lambda x: cost_per_conversion(x['Spent'],x['Approved_Conversion']),axis=1)
    data['Conversion_Value']=data.apply(lambda x: conversion_value(x['Approved_Conversion'],100),axis=1)
    data['ROAS']=data.apply(lambda x: return_on_ad_spend(x['Conversion_Value'],x['Spent']),axis=1)
    data['CostPerMille']=data.apply(lambda x: cost_per_mille(x['Spent'],x['Impressions']),axis=1)
    data['RevenuePerMille']=data.apply(lambda x: rev_per_mille(x['Conversion_Value'],x['Impressions']),axis=1)
    data['ClickToLeadRate']=data.apply(lambda x: click_to_lead_rate(x['Clicks'],x['Total_Conversion']),axis=1)

    # a hot cold column for whether or not a conversion was free
    data['isFree']=data.apply(lambda x: 1 if x['Spent']<=0 and x['Approved_Conversion'] >=1 else 0,axis=1)

    return data
#-----------------------------------------------------------------------------------------------------------------------------
def give_numeric_lables(data):
    data['age'].replace((3.0,4.5,3.5,4.0),('30-34','45-49','35-39','40-44'),inplace=True)
    data['gender'].replace((2,1),('M','F'),inplace=True)
    data=data.drop(columns='xyz_campaign_id')
    return data
#-----------------------------------------------------------------------------------------------------------------------------

def campaign_overview(data):
    scope_plot_data=data[['xyz_campaign_id','Approved_Conversion','Spent','Impressions']].groupby(['xyz_campaign_id'], #,'ROAS','RevenuePerMille'
    as_index=False).agg('mean').rename(columns={'xyz_campaign_id':'Campaign',
    'Approved_Conversion':'Average Number Of Conversions','Spent':'Average Expense',
    'Impressions': 'Average Number Of Impressions'})#,'ROAS':'Average ROAS','RevenuePerMille':'Average Revenue per Mille'})
    display(scope_plot_data.style.set_table_styles([ 
    #{"selector": "th",       "props": [("background-color", "#2F1B08"), ("color", "white"),("border", "3px solid black")]},  # Headers
    {"selector": ".col_heading","props":[("background-color", "#0E3AA8"), ("color", "white"),("border", "3px solid black")]},  # Headers
    {"selector": ".row_heading", "props": [("background-color", "#0E3AA8"), ("color", "white"),("border", "3px solid black")]},  # Index (row labels)
    {"selector": "td", "props": [("background-color", "#4D7BED"), ("color", "white"),("border", "3px solid black")]} #table cells
    ]))
    
    scope_plot_data_totals=data[['xyz_campaign_id','Approved_Conversion','Spent','Impressions']].groupby(['xyz_campaign_id'],
    as_index=False).agg('sum').rename(columns={'xyz_campaign_id':'Campaign',
    'Approved_Conversion':'Total Number Of Conversions','Spent':'Total Expense',
    'Impressions': 'Total Number Of Impressions'})
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig=plt.figure(figsize=(10,2.5))
    plt.title('Totals\n',fontsize=18)
    plt.subplot(1,3,1)
    sns.barplot(hue=scope_plot_data_totals['Campaign'],y=scope_plot_data_totals['Total Number Of Conversions'])
    plt.title('Conversions')
    plt.xlabel('')
    plt.ylabel('')
    
    plt.subplot(1,3,2)
    sns.barplot(hue=scope_plot_data_totals['Campaign'],y=scope_plot_data_totals['Total Expense'])
    plt.title('Expense')
    plt.xlabel('')
    plt.ylabel('')
    
    plt.subplot(1,3,3)
    sns.barplot(hue=scope_plot_data_totals['Campaign'],y=scope_plot_data_totals['Total Number Of Impressions'])
    plt.title('Impressions')
    plt.xlabel('')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.show()
    

    

    

#-----------------------------------------------------------------------------------------------------------------------------

def data_makeup(data):
    camp_c=data.loc[data['xyz_campaign_id']=='campaign_c'];print(f"campaign_c makes up:\n      {camp_c.shape[0]/data.shape[0]:,.2%} of the data\n      {camp_c['Spent'].sum()/data['Spent'].sum():.2%} of ad spend\n      {camp_c['Approved_Conversion'].sum()/data['Approved_Conversion'].sum():.2%} of Approved Conversions")
    print('---');camp_b=data.loc[data['xyz_campaign_id']=='campaign_b'];print(f"campaign_b makes up:\n      {camp_b['Approved_Conversion'].sum()/data['Approved_Conversion'].sum():.2%} of Approved Conversions."); print('---')
    print('total data rows: ',data.shape[0],'\ncamp_c data rows: ',camp_c.shape[0])

#-------------------------------------------------------------------------------------------------------------------------

def paid_and_not(data):
    no_spend_no_impressions=data.loc[(data['Spent']<=0)&(data['Impressions']<=0)].shape[0];
    yes_spend_no_impressions=data.loc[(data['Spent']>0)&(data['Impressions']<=0)].shape[0];
    no_spend_yes_impressions=data.loc[(data['Spent']<=0)&(data['Impressions']>0)].shape[0];
    grouped_yes_no_spend=data.loc[data['Spent']<=0].groupby('xyz_campaign_id',as_index=False)[['Approved_Conversion','Impressions']].sum().rename(columns={'Approved_Conversion':'Free_Approved_Conversions','Impressions':'Free_Impressions'});
    grouped_yes_spend=data.loc[data['Spent']>0].groupby('xyz_campaign_id',as_index=False)[['Approved_Conversion','Impressions']].sum();
    grouped_yes_no_spend['Paid_Approved_Conversions']=grouped_yes_spend['Approved_Conversion']
    grouped_yes_no_spend['Paid_Impressions']=grouped_yes_spend['Impressions'];
    display(grouped_yes_no_spend[[
        'xyz_campaign_id',	'Free_Approved_Conversions','Paid_Approved_Conversions',	'Free_Impressions',		'Paid_Impressions']].style.set_table_styles([ 
    #{"selector": "th",       "props": [("background-color", "#2F1B08"), ("color", "white"),("border", "3px solid black")]},  # Headers
    {"selector": ".col_heading","props":[("background-color", "#0E3AA8"), ("color", "white"),("border", "3px solid black")]},  # Headers
    {"selector": ".row_heading", "props": [("background-color", "#0E3AA8"), ("color", "white"),("border", "3px solid black")]},  # Index (row labels)
    {"selector": "td", "props": [("background-color", "#4D7BED"), ("color", "white"),("border", "3px solid black")]} #table cells
    ]))

    data['Free']=data['Spent'].apply(lambda x: 'Free' if x<=0 else 'Paid');
    free_impressions=data.loc[(data['Spent']==0)]['Impressions'].sum();
    total_impressions=data['Impressions'].sum();
    pie_data={'Free': free_impressions,'Paid':total_impressions-free_impressions};

    paid_free=data.groupby(['xyz_campaign_id','Free'],as_index=False)['Approved_Conversion'].sum();
    a=paid_free.loc[paid_free['xyz_campaign_id']=='campaign_a'];
    b=paid_free.loc[paid_free['xyz_campaign_id']=='campaign_b'];
    c=paid_free.loc[paid_free['xyz_campaign_id']=='campaign_c'];
    pie_df=pd.DataFrame(pie_data,[0]);
    labels=list(pie_df.columns);
    sizes=list(pie_df.iloc[0,:]);
    spent=data.groupby('xyz_campaign_id',as_index=False)['Spent'].sum();
    sp_labels=list(f'\n{camp}\n   {spen:,.0f}' if camp == 'campaign_a' else f'{camp}\n   {spen:,.0f}' for camp,spen in zip(spent['xyz_campaign_id'],spent['Spent']));
    sp_sizes=spent['Spent'];
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(18,7));

    plt.subplot(2,3,1);
    plt.pie(sizes,labels=labels,explode=[0,.2]);
    plt.title(f'Of {total_impressions:,.0f} impressions, {round(free_impressions/total_impressions*100,2)}% are free.');

    plt.subplot(2,3,2);
    plt.pie(sp_sizes,labels=sp_labels,explode=[.5,.5,.5]);
    plt.title('Ad Spend Across Campaigns');

    plt.subplot(2,3,3);
    sns.barplot(data=paid_free,x='xyz_campaign_id',y='Approved_Conversion',hue='Free');
    plt.title('Conversions Across Campaigns');
    plt.xlabel('');

    plt.subplot(2,3,4);
    sns.barplot(data=a,x='xyz_campaign_id',y='Approved_Conversion',hue='Free');
    plt.xlabel('');

    plt.subplot(2,3,5);
    sns.barplot(data=b,x='xyz_campaign_id',y='Approved_Conversion',hue='Free');
    plt.xlabel('');

    plt.subplot(2,3,6);
    sns.barplot(data=c,x='xyz_campaign_id',y='Approved_Conversion',hue='Free');
    plt.xlabel('');

    plt.tight_layout();
    plt.show();

#-------------------------------------------------------------------------------------------------------------------------------------------

def metric_evaluation_by_campaign(data):
    display(data.groupby(['xyz_campaign_id'])[['ClickThroughRate',
        'ConversionRate', 'CostPerClick', 'CostPerConversion',
        'Conversion_Value','ROAS','RevenuePerMille']].agg('mean').reset_index().rename(columns={'xyz_campaign_id':'Campaign',
        'ClickThroughRate': 'Average CTR','CostPerClick':'Average Cost/Click',
        'ConversionRate': 'Average ConversionRate','ROAS': 'Average ROAS','RevenuePerMille':'Average RevenuePerMille'}).style.set_table_styles([ 
    #{"selector": "th",       "props": [("background-color", "#2F1B08"), ("color", "white"),("border", "3px solid black")]},  # Headers
    {"selector": ".col_heading","props":[("background-color", "#0E3AA8"), ("color", "white"),("border", "3px solid black")]},  # Headers
    {"selector": ".row_heading", "props": [("background-color", "#0E3AA8"), ("color", "white"),("border", "3px solid black")]},  # Index (row labels)
    {"selector": "td", "props": [("background-color", "#4D7BED"), ("color", "white"),("border", "3px solid black")]} #table cells
        ]))

#-------------------------------------------------------------------------------------------------------------------------------------------

# impact on ROAS

def roas_impact(data):
    # age and gender partitioned with xyz_campaign_id

    #only include paid
    df=data.loc[data['Free']=='Paid']

    age_df=df.groupby(['age','xyz_campaign_id'],as_index=False)[['Conversion_Value','Spent']].sum()
    age_df['ROAS']=age_df['Conversion_Value']/age_df['Spent']
    age_df=age_df.replace([np.inf,-np.inf],np.nan).dropna(axis=0)
    age_df.sort_values(by=['age','xyz_campaign_id'],ascending=True,inplace=True)

    gender_df=df.groupby(['gender','xyz_campaign_id'],as_index=False)[['Conversion_Value','Spent']].sum()
    gender_df['ROAS']=gender_df['Conversion_Value']/gender_df['Spent']
    gender_df=gender_df.replace([np.inf,-np.inf],np.nan).dropna(axis=0)
    gender_df.sort_values(by=['gender','xyz_campaign_id'],ascending=True,inplace=True)

    # only include paid
    df=data.loc[data['Free']=='Paid']

    interest_df=df.groupby(['interest',],as_index=False)[['Conversion_Value','Spent']].sum()
    interest_df['ROAS']=interest_df['Conversion_Value']/interest_df['Spent']
    interest_df=interest_df.replace([np.inf,-np.inf],np.nan).dropna(axis=0)
    interest_df.sort_values(by=['interest'],ascending=True,inplace=True)
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['figure.figsize']=(14,4)

    plt.subplot(1,2,1)
    sns.barplot(hue=gender_df['gender'],y=gender_df['ROAS'],x=gender_df['xyz_campaign_id'],palette='colorblind')
    plt.xlabel('gender')

    plt.subplot(1,2,2)
    sns.barplot(hue=age_df['age'],y=age_df['ROAS'],x=age_df['xyz_campaign_id'],palette='colorblind')
    plt.xlabel('age')

    plt.suptitle(' Paid Ads\nImpact of Gender and Age on ROAS')
    plt.show()

    plt.rcParams['figure.figsize']=(18,5)
    plt.title('Paid Ads\nImpact on ROAS by Different Interest Groups Listed on the Facebook Account of Users',fontsize = 20)

    sns.barplot(x=interest_df['interest'],y=interest_df['ROAS'],palette='muted',)
    plt.xlabel('Interest Groups')
    plt.show()


#-------------------------------------------------------------------------------------------------------------------------------------------

def interactive_interest_roas(data):
    df=data.loc[data['Free']=='Paid']
    interest_df=df.groupby(['interest',],as_index=False)[['Conversion_Value','Spent']].sum()
    interest_df['ROAS']=interest_df['Conversion_Value']/interest_df['Spent'] if ( not np.isna(interest_df['Spent']) and interest_df['Spent']>0) else 0
    interest_df=interest_df.replace([np.inf,-np.inf],np.nan).dropna(axis=0)
    interest_df=interest_df.sort_values(by=['ROAS'],ascending=False)
    interest_df=interest_df=interest_df.reset_index(drop=True)
    @interact
    def interact_roas_interest(top_n=list(i for i in range(5,41))):        
        
        display(interest_df[['interest','ROAS']].head(top_n).style.set_table_styles([ 
            #{"selector": "th",       "props": [("background-color", "#2F1B08"), ("color", "white"),("border", "3px solid black")]},  # Headers
            {"selector": ".col_heading","props":[("background-color", "#0E3AA8"), ("color", "white"),("border", "3px solid black")]},  # Headers
            {"selector": ".row_heading", "props": [("background-color", "#0E3AA8"), ("color", "white"),("border", "3px solid black")]},  # Index (row labels)
            {"selector": "td", "props": [("background-color", "#4D7BED"), ("color", "white"),("border", "3px solid black")]} #table cells
            ]))
      
#-------------------------------------------------------------------------------------------------------------------------------------------
def cost_per_mill_by_campaign(data):
    paid=data.loc[data['Free']=='Paid']
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['figure.figsize']=(8,2.5)
    plt.gcf().set_facecolor("#416AD2")
    sns.barplot(data=paid,hue='xyz_campaign_id',y='CostPerMille')
    plt.title('Paid Impressions\nCost Per Mille Across Campaigns')
    plt.xlabel(' ')
    plt.gca().set_facecolor("#7E9FF4")
    plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------

def per_mille_review(data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['figure.figsize']=(12,4)
    plt.title('Paid and Free\nAcross Age and Gender\n.\n')
    plt.gca().set_facecolor("#416AD2")
    plt.subplot(1,2,1)
    plt.gca().set_facecolor("#7E9FF4")
    sns.barplot(x=data['age'],y=data['CostPerMille'],hue= data['gender'],palette='colorblind')
    plt.xlabel('Age Groups')
    plt.ylabel('CostPerMille')
    plt.title('Brand Awareness Related to Cost')
    plt.subplot(1,2,2)
    plt.gca().set_facecolor("#7E9FF4")
    plt.title('Brand Awareness Related to Revenue')
    sns.barplot(x=data['age'],y=data['RevenuePerMille'], hue= data['gender'], palette='colorblind')
    plt.xlabel('Age Groups')
    plt.ylabel('RevenuePerMille')
    plt.gcf().set_facecolor("#416AD2")

    plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------

#GET DEMOGRAPHIC DATAFRAMES
import itertools
def create_demographic_dfs(data):
    def get_combos(cols):
        dem_combos=[list(combo) for r in range(1,len(cols)+1) for combo in itertools.combinations(cols,r)]
        return dem_combos
    cols= ['age', 'gender', 'interest']
    dem_combos=get_combos(cols)

    def get_agg_frames(data,dem_combos,metric,dataframes='dataframes'):
        '''
        get_agg_frames(dem_combos,metric,print_syntax=False,dataframes='dataframes'):
        combinations is a list of input column strings
        metric is the column to be aggregated
        print syntax will print a string for copy and pasting dataframes
        dataframes is the variable name passed in syntax 

        '''
        title='sub_demographic'
        d1=dataframes
        dataframes={}
        for frame in dem_combos:
            x=''
            for i in frame:
                x+=i+'_'
            df=x+title
            df=data.groupby(frame,as_index=False)[metric].sum()
            size_df=data.groupby(frame,as_index=False).size().rename(columns={'size':'NumObservations'})
            df=pd.merge(df,size_df,on=frame)
            df['ClickThroughRate']=df.apply(lambda x: click_through_rate(x['Clicks'],x['Impressions']),axis=1)
            df['ConversionRate']=df.apply(lambda x: conversion_rate(x['Approved_Conversion'],x['Total_Conversion']),axis=1)
            df['CostPerClick']=df.apply(lambda x: cost_per_click(x['Spent'],x['Clicks']),axis=1)
            df['CostPerConversion']=df.apply(lambda x: cost_per_conversion(x['Spent'],x['Approved_Conversion']),axis=1)
            df['ConversionValue']=df.apply(lambda x: conversion_value(x['Approved_Conversion'],100),axis=1)
            df['ROAS']=df.apply(lambda x: return_on_ad_spend(x['Conversion_Value'],x['Spent']),axis=1)
            df['CostPerMille']=df.apply(lambda x: cost_per_mille(x['Spent'],x['Impressions']),axis=1)
            df['RevenuePerMille']=df.apply(lambda x: rev_per_mille(x['Conversion_Value'],x['Impressions']),axis=1)
            df=df[frame+['NumObservations']+metric+['ClickThroughRate','ConversionRate','CostPerClick','CostPerConversion','ConversionValue','ROAS','CostPerMille','RevenuePerMille']]
            dataframes[x+title]=pd.DataFrame(df)
        return dataframes

    metric=['Spent','Impressions','Clicks','Total_Conversion','Approved_Conversion','Conversion_Value']
    dataframes=get_agg_frames(data,dem_combos,metric,dataframes='dataframes')
    return dataframes


def plot_dfs_under15_rows(dfs_dict):
    plot_dfs=[key for key,val in dfs_dict.items() if val.shape[1]>16 and val.shape[0]>8]    
    for ax in range(1,len(plot_dfs)+1):
        plot_index=int(ax-1)
        cur_plot=dfs_dict[plot_dfs[plot_index]]
        slice=cur_plot.shape[1]-15
        key = pd.concat([cur_plot.iloc[:, :slice],cur_plot[['RevenuePerMille', 'ROAS']]], axis=1)
        plot=plot_dfs[plot_index]        
        display(key.T.style.set_table_styles([ 
            #{"selector": "th",       "props": [("background-color", "#2F1B08"), ("color", "white"),("border", "3px solid black")]},  # Headers
            {"selector": ".col_heading","props":[("background-color", "#0E3AA8"), ("color", "white"),("border", "3px solid black")]},  # Headers
            {"selector": ".row_heading", "props": [("background-color", "#0E3AA8"), ("color", "white"),("border", "3px solid black")]},  # Index (row labels)
            {"selector": "td", "props": [("background-color", "#4D7BED"), ("color", "white"),("border", "3px solid black")]}, #table cells
            {"selector": "th.blank","props": [ ("background-color", "#0E3AA8"), ("color", "white"),("text-align", "center"),("border", "3px solid black")]}]))        
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig=plt.figure(figsize=(20,3))
        top_ntile=np.ceil((cur_plot.shape[0]-40)/cur_plot.shape[0]*100) if cur_plot.shape[0]>40 else 0
        import matplotlib.pyplot as plt
        import seaborn as sns
        pct=np.percentile(cur_plot['RevenuePerMille'],top_ntile)
        cur_plot1=cur_plot.loc[(cur_plot['RevenuePerMille']>pct)]
        y=cur_plot1['RevenuePerMille']
        mn=y.min()
        x=cur_plot1.index
        sns.barplot(y=y,x=x)
        plt.title(f'{plot[:-16]} - RevenuePerMille - top {100-top_ntile}% - RevenuePerMille > {round(mn,2)}'if cur_plot.shape[0]>40 else f'{plot[:-16]} - RevenuePerMille')
        plt.xlabel('')
        plt.grid()
        plt.show() 
        fig=plt.figure(figsize=(20,3))
        top_ntile2=np.ceil((cur_plot.shape[0]-40)/cur_plot.shape[0]*100) if cur_plot.shape[0]>40 else 0
     
        pct2=np.percentile(cur_plot['ROAS'],top_ntile2)
        cur_plot2=cur_plot.loc[(cur_plot['ROAS']>pct2)]
        y2=cur_plot2['ROAS']
        mn2=y2.min()
        x2=cur_plot2.index          
        sns.barplot(y=y2,x=x2)
        plt.title(f'{plot[:-16]} - ROAS - top {100-top_ntile}% - ROAS > {round(mn2,2)}' if cur_plot.shape[0]>40 else f'{plot[:-16]} - ROAS') 
        plt.xlabel('')
        plt.grid()
        plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------

border_color="#0E3AA8"
body_color="#4D7BED"
framestyles=[{"selector": ".col_heading","props":[("background-color",border_color ), ("color", "white"),("border", "3px solid black")]},  # Headers
            {"selector": ".row_heading", "props": [("background-color", border_color), ("color", "white"),("border", "3px solid black")]},  # Index (row labels)
            {"selector": "td", "props": [("background-color", body_color), ("color", "white"),("border", "3px solid black")]}, #table cells
            {"selector": "th.blank","props": [ ("background-color", border_color), ("color", "white"),("text-align", "center"),("border", "3px solid black")]}] 

#-------------------------------------------------------------------------------------------------------------------------------------------
#THIS FUNCTION REVEALS THAT THE DATA HAS BEEN TAMPERED WITH OR IS OTHERWISE NOT GENUINE
def cost_conversion(data):
    function_cn=data[['age','gender','interest','Spent','CostPerConversion']].copy()
    function_cn['(spent-cpConv)/spent']=round(abs((function_cn['Spent']-function_cn['CostPerConversion'])/function_cn['Spent']),2)
    a=function_cn.loc[(function_cn['(spent-cpConv)/spent']<0.3)&(function_cn['CostPerConversion']>0.001)]
    b=function_cn.loc[(function_cn['(spent-cpConv)/spent']>=0.3)&(function_cn['(spent-cpConv)/spent']<0.58)&(function_cn['CostPerConversion']>0.001)]
    e=c=function_cn.loc[(function_cn['(spent-cpConv)/spent']>=.58)&(function_cn['(spent-cpConv)/spent']<0.71)&(function_cn['CostPerConversion']>0.001)]
    c=function_cn.loc[(function_cn['(spent-cpConv)/spent']>=0.71)&(function_cn['CostPerConversion']>0.001)]
    d=function_cn.loc[(function_cn['CostPerConversion']<=0.001)]
    sns.scatterplot(data=a,x='Spent',y='CostPerConversion',color='r',label='1-to-1')
    sns.scatterplot(data=b,x='Spent',y='CostPerConversion',color='b',label='1-to-1/2')
    sns.scatterplot(data=c,x='Spent',y='CostPerConversion',color='y',label='1-to-1/4 & lower')
    sns.scatterplot(data=d,x='Spent',y='CostPerConversion',color='g',label='No Conversions')
    sns.scatterplot(data=e,x='Spent',y='CostPerConversion',color='teal',label='1-to-1/3')
    plt.legend()
    plt.grid()
    plt.show()
    # List of your DataFrames and categorical columns
    dfs = [a, b, c, d, e]
    cn_categories = ['age', 'gender', 'interest']
    cn_titles = ['age distribution', 'gender distribution', 'interest distribution']

    # Loop through each DataFrame
    for i, df in enumerate(dfs):
        print(f"Segment {i + 1}")
        
        # Row of pie charts
        fig = make_subplots(rows=1, cols=3, subplot_titles=cn_titles, specs=[[{'type':'domain'}]*3])
        
        for j, cat in enumerate(cn_categories):
            pie_fig = px.pie(df, names=cat, title=None)
            fig.add_trace(pie_fig.data[0], row=1, col=j+1)

        fig.update_layout(title_text=f"Data Summary for Segment {i + 1}", showlegend=True)
        fig.show()


#-------------------------------------------------------------------------------------------------------------------------------------------

def get_data():
    data=pd.read_csv('campaign_c_facebook_ads.csv')
    data.drop(columns=[ 'ClickThroughRate',
       'ConversionRate', 'CostPerClick', 'CostPerConversion',
       'Conversion_Value', 'ROAS', 'CostPerMille', 'RevenuePerMille', 'isFree',
       'Free'],inplace=True)
    data=add_metrics2(data)
    return data

#-------------------------------------------------------------------------------------------------------------------------------------------
def sales_funnel_plot(data):
    border_color="#0E3AA8"
    body_color="#4D7BED"
    display(pd.DataFrame([data[['CostPerMille','ClickThroughRate','ClickToLeadRate','ConversionRate']].mean()], columns=['CostPerMille','ClickThroughRate','ClickToLeadRate','ConversionRate']).style.set_table_styles([{"selector": ".col_heading","props":[("background-color",border_color ), ("color", "white"),("border", "3px solid black")]},  # Headers
                {"selector": ".row_heading", "props": [("background-color", border_color), ("color", "white"),("border", "3px solid black")]},  # Index (row labels)
                {"selector": "td", "props": [("background-color", body_color), ("color", "white"),("border", "3px solid black")]}, #table cells
                {"selector": "th.blank","props": [ ("background-color", border_color), ("color", "white"),("text-align", "center"),("border", "3px solid black")]}] ))
    spent=data['Spent'].sum()
    impressions=data['Impressions'].sum()/1_1000
    clicks=data['Clicks'].sum()
    leads=data['Total_Conversion'].sum()
    conversions=data['Approved_Conversion'].sum()
    funnel_data = dict(
        value=[spent, impressions, clicks, leads, conversions,conversions*100],
        stage=["Spent", "Impressions Per 1,000 (Mille)", "Clicks", "Leads", "Sales","Revenue"])
    fig = px.funnel(funnel_data, x='value', y='stage')
    fig.show()
#-------------------------------------------------------------------------------------------------------------------------------------------
def funnel_by_demographics(data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    funnel_metrics=['CostPerMille','ClickThroughRate','ClickToLeadRate','ConversionRate']
    plt.figure(figsize=(16,3.5))
    plt.title('Sales Funnel by Age Groups\n\n')
    for i,metric in enumerate(funnel_metrics):
        plt.subplot(1,4,i+1)
        sns.barplot(y=data[funnel_metrics[i]],hue=data['age'])
        plt.xlabel('')
        plt.title(funnel_metrics[i])
    plt.tight_layout()
    plt.show()
    fig=plt.figure(figsize=(16,3.5))

    plt.title('Sales Funnel by Gender\n\n')
    for i,metric in enumerate(funnel_metrics):
        ax=fig.add_subplot(1,4,i+1)
        sns.barplot(y=data[funnel_metrics[i]],hue=data['gender'])
        ax.set_xlabel('')
        ax.set_title(funnel_metrics[i])
    plt.tight_layout()
    plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------
def funnel_plot_lin_reg(data,features_targets=[('Spent', 'Approved_Conversion')]):
    sales_funnel = features_targets
    num_plots = len(sales_funnel)
    fig = make_subplots(rows=1, cols=num_plots)

    for i, (x_col, y_col) in enumerate(sales_funnel):
        scatter = px.scatter(
            data, x=x_col, y=y_col, opacity=0.65,
            trendline='ols', trendline_color_override='darkblue'
        )
        for trace in scatter.data:
            fig.add_trace(trace, row=1, col=i+1)
        fig.update_xaxes(title_text=x_col, row=1, col=i+1)
        fig.update_yaxes(title_text=y_col, row=1, col=i+1)

    fig.update_layout(
        height=600, width=600 * num_plots
    )
    fig.show()


#-------------------------------------------------------------------------------------------------------------------------------------------

def optimal_three_bins(arr, min_k=5, max_k=None, metric='mean_diff'):
    if max_k is None:
        max_k = len(arr) // 2  # default upper limit

    arr = np.sort(arr)
    n = len(arr)
    best_score = -np.inf
    best_split = None

    # Try all valid splits where category sizes fall within [min_k, max_k]
    for i in range(min_k, min(max_k + 1, n - 2 * min_k + 1)):
        for j in range(i + min_k, min(i + max_k + 1, n - min_k + 1)):
            A, B, C = arr[:i], arr[i:j], arr[j:]

            if len(C) < min_k or len(C) > max_k:
                continue

            # Example metric: maximize total pairwise mean distance
            mu = [np.mean(bin_) for bin_ in [A, B, C]]
            score = np.abs(mu[0] - mu[1]) + np.abs(mu[1] - mu[2]) + np.abs(mu[0] - mu[2])

            if score > best_score:
                best_score = score
                best_split = (A, B, C)
   
    low=best_split[0]
    middle=best_split[1]
    high=best_split[2]
    lookup_dict={}
    for i in low:lookup_dict[str(round(float(i),3))]='low'
    for i in middle:lookup_dict[str(round(float(i),3))]='middle'
    for i in high:lookup_dict[str(round(float(i),3))]='high'

    return lookup_dict

#---------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from itertools import combinations

def optimal_n_bins(arr, num_bins=6, min_k=5, max_k=None, metric='mean_diff'):
    if max_k is None:
        max_k = len(arr) // 2

    arr = np.sort(arr)
    n = len(arr)
    best_score = -np.inf
    best_split = None

    # Recursive split candidate generator
    def generate_splits(start, depth, path):
        if depth == num_bins - 1:
            remaining = arr[start:]
            if min_k <= len(remaining) <= max_k:
                yield path + [remaining]
            return
        for end in range(start + min_k, min(n, start + max_k + 1)):
            segment = arr[start:end]
            if len(segment) >= min_k:
                yield from generate_splits(end, depth + 1, path + [segment])

    for split in generate_splits(0, 0, []):
        # Calculate pairwise mean difference
        mu = [np.mean(bin_) for bin_ in split]
        score = sum(np.abs(mu[i] - mu[j]) for i, j in combinations(range(num_bins), 2))

        if score > best_score:
            best_score = score
            best_split = split

    # Build lookup dictionary
    lookup_dict = {}
    labels = [f'{i+1}' for i in range(num_bins)]
    for label, bin_ in zip(labels, best_split):
        for val in bin_:
            lookup_dict[str(round(float(val), 3))] = label

    return lookup_dict,num_bins
#-------------------------------------------------------------------------------------------------------------------------------------------
def interest_reduction(data):
    grouped_interest=data.groupby('interest',as_index=False)[['ROAS','RevenuePerMille']].median()
    best_roas,num_roas=utils.optimal_n_bins(np.array(grouped_interest['ROAS']),num_bins=6,min_k=5,max_k=grouped_interest.shape[0]-(8),metric='mean_diff')
    best_rpm,num_roas=utils.optimal_n_bins(np.array(grouped_interest['RevenuePerMille']),num_bins=6,min_k=5,max_k=grouped_interest.shape[0]-(8),metric='mean_diff')
    roas_list=[int(i+1) for i in range(num_roas)]
    rpm_list=[int(i+1) for i in range(num_roas)]
    unique_pairs = list(itertools.product(roas_list,rpm_list))
    def get_interest_cat(roas,rpm):
        key=(int(best_roas[str(round(float(roas),3))]),int(best_rpm[str(round(float(rpm),3))]))
        unique_pairs = list(itertools.product(roas_list,rpm_list))
        for i in range(len(unique_pairs)):
            if key==unique_pairs[i] or key[::-1]==unique_pairs[i]: 
                return int(i+1)
        print('func is done')   
    grouped_interest['interest_category']=grouped_interest.apply(lambda row: get_interest_cat(row['ROAS'],row['RevenuePerMille']),axis=1)
    lookup_data=dict(zip(grouped_interest['interest'],grouped_interest['interest_category']))
    data['interest_category']=data['interest'].apply(lambda x: lookup_data[x])
    grouped_int_cat=data.groupby('interest_category',as_index=False)['Total_Conversion'].median()
    grouped_int_cat=grouped_int_cat.sort_values(by='Total_Conversion').reset_index(drop=True)
    grouped_int_cat['ordinalized_interest']=grouped_int_cat.index+1
    revised_lookup=dict(zip(grouped_int_cat['interest_category'],grouped_int_cat['ordinalized_interest']))
    data['interest_category']=data['interest_category'].apply(lambda x: revised_lookup[x])
    json_data={'second_lookup':revised_lookup,'first_lookup':lookup_data}
    with open('campaign_c_interest_keys.json','w') as f:
        json.dump(json_data,f,indent=4)
    
    return data
#-------------------------------------------------------------------------------------------------------------------------------------------
def get_pred_plots(data):
    plot_data=data.copy()
    plot_data=plot_data.sort_values(by='Spent',ascending=True)
    #plot_data['pred_Approved_Conversion']=round(plot_data['pred_Approved_Conversion'])
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(20,10))
    plt.subplot(2,1,1)
    sns.scatterplot(plot_data,x='Spent',y='Total_Conversion',label='Total_Conversion',color='blue')
    sns.scatterplot(plot_data,x='Spent',y='pred_Total_Conversion',label='Predicted Leads',color='red')
    plt.title('Predicted and Actual Leads')
    plt.grid()
    plt.legend()
    plt.subplot(2,1,2)
    sns.scatterplot(plot_data,x='Spent',y='Approved_Conversion',label='Sales',color='blue')
    sns.scatterplot(plot_data,x='Spent',y='pred_Approved_Conversion',label='Predicted Sales',color='red')
    plt.title('Predicted and Actual Leads')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------------sa