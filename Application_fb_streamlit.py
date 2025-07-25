
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

from itertools import combinations
import utils
import modeling_utils
import joblib
from joblib import dump, load
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

#plt.style.use('seaborn-v0_8-colorblind')
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

# Streamlit page configuration
st.set_page_config(
    page_title="Facebook Campaign Analysis",
    page_icon="📊",
    layout="wide"
)


# Load and process data (only once)
@st.cache_data
def load_data():
    return utils.get_data()
full_data=load_data()


@st.cache_data
def grouped_metrics(df=full_data):    
    frame=['age','gender','interest']
    metric=['Clicks','Impressions','Approved_Conversion','Total_Conversion','Spent']
    df=df.groupby(frame,as_index=False)[metric].sum()
    size_df=df.groupby(frame,as_index=False).size().rename(columns={'size':'NumObservations'})
    df=pd.merge(df,size_df,on=frame)
    df['ClickThroughRate']=df.apply(lambda x: utils.click_through_rate(x['Clicks'],x['Impressions']),axis=1)
    df['ConversionRate']=df.apply(lambda x: utils.conversion_rate(x['Approved_Conversion'],x['Total_Conversion']),axis=1)
    df['CostPerClick']=df.apply(lambda x: utils.cost_per_click(x['Spent'],x['Clicks']),axis=1)
    df['CostPerConversion']=df.apply(lambda x: utils.cost_per_conversion(x['Spent'],x['Approved_Conversion']),axis=1)
    df['ConversionValue']=df.apply(lambda x: utils.conversion_value(x['Approved_Conversion'],100),axis=1)
    df['ROAS']=df.apply(lambda x: utils.return_on_ad_spend(x['ConversionValue'],x['Spent']),axis=1)
    df['CostPerMille']=df.apply(lambda x: utils.cost_per_mille(x['Spent'],x['Impressions']),axis=1)
    df['RevenuePerMille']=df.apply(lambda x: utils.rev_per_mille(x['ConversionValue'],x['Impressions']),axis=1)
    df=df[frame+['NumObservations']+metric+['ClickThroughRate','ConversionRate','CostPerClick','CostPerConversion','ConversionValue','ROAS','CostPerMille','RevenuePerMille']]
    return {'df':df}


@st.cache_data
def leads_sales_plots(demographic_data=full_data):
    #import plotly.graph_objects as go
    #from plotly.subplots import make_subplots

    lead_model_info=joblib.load('lead_prediction_model_info.joblib')
    demographic_data=modeling_utils.make_predictions_pipeline(demographic_data,lead_model_info,{})
    approved_conversion_model_info=joblib.load('approved_conversion_model_info.joblib')
    demographic_data=modeling_utils.make_predictions_pipeline(demographic_data,approved_conversion_model_info,{})

    demographic_data=demographic_data.groupby(['age','gender','interest'],as_index=False)[['Spent','Total_Conversion','pred_Total_Conversion','pred_Approved_Conversion','Approved_Conversion']].sum()
    demographic_data['ROAS']=demographic_data.apply(lambda row: utils.return_on_ad_spend(row['Approved_Conversion']*100,row['Spent']),axis=1)
    demographic_data=demographic_data.sort_values(by='ROAS',ascending=False)
    demographic_data=demographic_data.reset_index(drop=True)  
    demographic_data['demographic']=demographic_data.index
    
    X = round(demographic_data['Spent'])
    Y = demographic_data['demographic'] 
    Z = demographic_data['Total_Conversion'] 
    Z2 = round(demographic_data['pred_Total_Conversion'])
    Z3 = demographic_data['Approved_Conversion']  
    Z4 = round(demographic_data['pred_Approved_Conversion'])

    # Create subplot container with two 3D plots
    fig = make_subplots(rows=2, cols=1, specs=[[{'type': 'scene'}], [{'type': 'scene'}]])

    # Plot 1: Leads
    fig.add_trace(
        go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(color='purple', opacity=0.6),name='Actual Leads'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter3d(x=X, y=Y, z=Z2, mode='markers', marker=dict(color='yellow', opacity=0.6),name='Predicted Leads'),
        row=1, col=1
    )

    # Plot 2: Sales
    fig.add_trace(
        go.Scatter3d(x=X, y=Y, z=Z3, mode='markers', marker=dict(color='purple', opacity=0.6),name='Actual Sales'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter3d(x=X, y=Y, z=Z4, mode='markers', marker=dict(color='yellow', opacity=0.6),name='Predicted Sales'),
        row=2, col=1
    )

    # Customize axes and titles
    fig.update_layout(
        height=1200, width=600,
        title_text="Plot_One:Leads; Plot_Two: Sales",
        scene=dict(
            xaxis_title='Spent',
            yaxis_title='Demographic Combinations',
            zaxis_title='Leads'
        ),
        scene2=dict(
            xaxis_title='Spent',
            yaxis_title='Demographic Combinations',
            zaxis_title='Sales'
        )
    )

    st.plotly_chart(fig)




@st.cache_data
def hint_plot():
    key=grouped_metrics()['df']      
    key=key.sort_values(by='ROAS',ascending=False)
    key=key.reset_index(drop=True)  
    key['age']=key['age'].replace((3.0,4.5,3.5,4.0),('30-34','45-49','35-39','40-44'))
    key['gender']=key['gender'].replace((2,1),('M','F'))
    key['ROAS']=key['ROAS'].round(2)          

    top_ntile=np.ceil((key.shape[0]-30)/key.shape[0]*100) if key.shape[0]>30 else 0
    pct=np.percentile(key['ROAS'],top_ntile)
    cur_plot1=key.loc[(key['ROAS']>pct)]

    #fig=plt.figure(figsize=(16,3))
    top_ntile2=np.ceil((cur_plot1.shape[0]-30)/cur_plot1.shape[0]*100) if cur_plot1.shape[0]>30 else 0
    pct2=np.percentile(cur_plot1['ROAS'],top_ntile2)
    cur_plot1=cur_plot1.loc[(cur_plot1['ROAS']>pct2)]
    y2=cur_plot1['ROAS']
    mn2=y2.min()
           
    st.markdown(f'Demographics - ROAS - top {100-top_ntile2}% - ROAS > {round(mn2,2)}' if cur_plot1.shape[0]>30 else f'Demographic - ROAS')
    st.bar_chart(data=cur_plot1,  x=None, y='ROAS', x_label=None, 
                 y_label='ROAS', color=None, horizontal=False, 
                 stack=None, width=None, height=None, use_container_width=True)
    st.dataframe(key[['age','gender','interest','ROAS']].T)  

@st.cache_data
def disp_img():
    st.image(image='Lead_pred_across_unseen_data.png',
             caption='Lead Predictions Across Unseen Data')
    st.image(image='Approved_Conversion_pred_across_unseen_data.png',
             caption='Sales Predictions Across Unseen Data')


@st.cache_data
def funnel(data=full_data): 
    spent=data['Spent'].sum()
    impressions=data['Impressions'].sum()/1_1000
    clicks=data['Clicks'].sum()
    leads=data['Total_Conversion'].sum()
    conversions=data['Approved_Conversion'].sum()
    funnel_data = dict(
        value=[spent, impressions, clicks, leads, conversions,conversions*100],
        stage=["Spent", "Impressions Per 1,000 (Mille)", "Clicks", "Leads", "Sales","Revenue"])
    fig = px.funnel(funnel_data, x='value', y='stage')
    fig.update_traces(
        textposition='inside',
        textfont=dict(color='black'))

    st.plotly_chart(fig)

#---------------------------------------------------------------------------------------------------------
st.title("Ad Campaign Prediction Model")
st.markdown("---")
#---------------------------------------------------------------------------------------------------------
st.subheader("The Original Campaign")

funnel()
#----------------------------------------------------------------------------------------------------------------
st.header("Data Driven Model")
st.subheader("---")

col1a, col2a = st.columns([.75, .25],gap='large',vertical_alignment='top',border=True)

with col1a:
    st.markdown("Prediction Plots")
    leads_sales_plots()

with col2a:
    st.markdown("Strength of Predictions")
    disp_img()


st.header("Run Your Own Campaign")
st.subheader("🕵️ Hint Data")
hint_plot()


#---#

@st.cache_data
def load_data_process_predictions(newer_data,older_data=pd.DataFrame()):
    newer_data=newer_data.copy()
    newer_data['age']=newer_data['age'].replace(('30-34','45-49','35-39','40-44'),(3.0,4.5,3.5,4.0))
    newer_data['gender']=newer_data['gender'].replace(('M','F'),(2,1))
    # Predict number of leads
    lead_model_info = joblib.load('lead_prediction_model_info.joblib')
    newer_data = modeling_utils.make_predictions_pipeline(newer_data, lead_model_info, dict())    
    # Predict approved conversions to sale
    approved_conversion_model_info = joblib.load('approved_conversion_model_info.joblib')
    newer_data = modeling_utils.make_predictions_pipeline(newer_data, approved_conversion_model_info, dict())
    older_data=pd.concat((newer_data,older_data),ignore_index=True)
    older_data=older_data.reset_index(drop=True)

    return older_data


# Sidebar for campaign selection
st.sidebar.header("🎯 Campaign Builder")
st.sidebar.markdown("Select parameters for your campaign:")

# Campaign selection inputs
age_cat = st.sidebar.selectbox(
    label="🎂 Select an Age Group",
    options=('30-34', '35-39', '40-44', '45-49'),
    index=None,
    placeholder="Choose age group..."
)

interest_cat = st.sidebar.selectbox(
    label="🎯 Select an Interest Category",
    options=tuple([i for i in sorted(list(full_data['interest'].unique()))]),
    index=None,
    placeholder="Choose interest..."
)

gender_cat = st.sidebar.selectbox(
    label="👥 Select a Gender",
    options=('M', 'F'),
    index=None,
    placeholder="Choose gender..."
)

spend_values = tuple([i for i in range(50, 351, 50)])
spend_cat = st.sidebar.selectbox(
    label="💰 Select a Spend Amount",
    options=spend_values,
    index=None,
    placeholder="Choose spend amount..."
)



# Initialize session state for persistent data storage
if 'age_inputs' not in st.session_state:
    st.session_state.age_inputs = []
if 'interest_inputs' not in st.session_state:
    st.session_state.interest_inputs = []
if 'gender_inputs' not in st.session_state:
    st.session_state.gender_inputs = []
if 'spend_inputs' not in st.session_state:
    st.session_state.spend_inputs = []

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Campaign Configuration")
    
    # Display current selection
    if age_cat or interest_cat or gender_cat or spend_cat:
        st.write("**Current Selection:**")
        current_selection = pd.DataFrame({
            'Age': [age_cat or "Not selected"],
            'Interest': [interest_cat or "Not selected"],
            'Gender': [gender_cat or "Not selected"],
            'Spend': [spend_cat or "Not selected"]
        })
        st.table(current_selection)

with col2:
    st.subheader("⚡ Actions")
    
    # Add selection button
    add_button = st.button(
        "➕ Add Current Selection",
        type="primary",
        use_container_width=True
    )
    
    if add_button:
        if age_cat and interest_cat and gender_cat and spend_cat:
            st.session_state.age_inputs.append(age_cat)
            st.session_state.interest_inputs.append(interest_cat)
            st.session_state.gender_inputs.append(gender_cat)
            st.session_state.spend_inputs.append(spend_cat)
            st.success("✅ Selection added successfully!")
        else:
            st.error("❌ Please select all categories before adding")
    


# Display saved campaigns
if len(st.session_state.age_inputs) > 0:
    st.markdown("---")
    st.subheader(f"({len(st.session_state.age_inputs)} Saved Mini-Campaigns)")
    
    # Create DataFrame from saved selections
    campaigns_df = pd.DataFrame({
        'Campaign #': range(1, len(st.session_state.age_inputs) + 1),
        'age': st.session_state.age_inputs,
        'interest': st.session_state.interest_inputs,
        'gender': st.session_state.gender_inputs,
        'Spent': st.session_state.spend_inputs
    })
    # Load data
    plot_data = load_data_process_predictions(campaigns_df,older_data=pd.DataFrame())

    # Display overview chart
    st.subheader("Campaign Expectations")
    
    st.line_chart(plot_data[['pred_Total_Conversion', 'pred_Approved_Conversion']],
                  y_label=['Expected_Leads','Expected_Sales'],
                  x=None,
                  x_label='Campaign')####-------------> remove or start with user inputs !@#$%^&

    
    # Display the table
    st.table(campaigns_df)
    
    revenue_per_sale=100

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_spend = sum(st.session_state.spend_inputs)
        st.metric("Total Spend", f"${total_spend:,}")

    with col2:
        avg_spend = total_spend / len(st.session_state.spend_inputs)
        st.metric("Avg Spend", f"${avg_spend:.0f}")
    
    with col3:
        expected_revenue = int(plot_data['pred_Approved_Conversion'].sum()*revenue_per_sale)
        st.metric("Expected Revenue", f"${expected_revenue:,}")
    
    with col4:
        expected_avg_rev = int(plot_data['pred_Approved_Conversion'].mean()*revenue_per_sale)
        st.metric("Expected Average Revenue", f"${expected_avg_rev:.0f}")
    




 
