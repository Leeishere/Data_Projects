from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from joblib import dump,load


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#########RENAME fit() and create class to store degree
################ use elastic  
# create a class and put gridsearch args in the constructor   
# with in the constructor l1 ratio
class PolyDegreeElasticHypertuner:

    def __init__(self,
                 l1_ratio=[0],
                 degrees=[1],
                 train_test_score_sum_reward_multiplier=1,
                 train_test_score_diff_penalty_multiplier=1,
                 gridsearch_params={'l1_ratio':[0,0.16,0.33,0.47,0.65,0.82,1],
                                    'selection':['random'],
                                    'alpha':np.linspace(0,1,10)+np.linspace(1,101,10),
                                    'max_iter':[100,500]},
                cross_val=5,
                scoring_metric='r2',
                test_score_weight=0.5
                ):
        """
        degrees is list of poly degrees
        l1_ratio default 0
        sum reward and diff penalty multipliers act to balance over and under fitting. sum*x-dif*y
        test score weight should be between 0 and 1, its compliment will be used as the train weight. the larger value favors its recipient
        """
        self.degrees=degrees
        self.train_test_score_sum_reward_multiplier=train_test_score_sum_reward_multiplier
        self.train_test_score_diff_penalty_multiplier=train_test_score_diff_penalty_multiplier
        self.gridsearch_params=gridsearch_params
        self.cross_val=cross_val
        self.scoring_metrics=scoring_metric
        self.test_score_weight=test_score_weight
        self.train_score_weight=1-self.test_score_weight
        

        self.fit_data={}
        self.fit_balance_scores={}

                 
# fit replaces: get_poly_degree_hyper_params()
    def fit(self,X_data,y_data):

        self.X_cols=X_data.columns
        assert(self.test_score_weight>=1)
        
        labels=[]
        test=[]
        train=[]
        balance=[]


        for cur_degree in self.degrees:
            
            poly=PolynomialFeatures(degree=cur_degree,include_bias=True)
            X_data_poly=poly.fit_transform(X_data)
            X_poly_df=pd.DataFrame(X_data_poly, columns=poly.get_feature_names_out(list(self.X_cols)))
            X_poly_df_cols=X_poly_df.columns

            poly_X_train,poly_X_test,poly_y_train,poly_y_test= train_test_split(X_data_poly,y_data,test_size=0.4,random_state=32)
            scaler=StandardScaler()
            poly_X_train=scaler.fit_transform(poly_X_train)
            poly_X_test=scaler.transform(poly_X_test)

 
            grid=GridSearchCV(param_grid=self.gridsearch_params,estimator=ElasticNet(),cv=5,scoring='r2')
            grid.fit(poly_X_train,poly_y_train)
            best_model=grid.best_estimator_
 
            test_score=r2_score(poly_y_test,grid.best_estimator_.predict(poly_X_test))
            train_score=r2_score(poly_y_train,grid.best_estimator_.predict(poly_X_train))
            ###############
            #THIS SHOULD GET THE TRAIN SCORE AND STORE BOTH VALUES, THEN RETURN PLOT DATA AND MODEL DATA AS DICTS. 
            # IT SHOULD ALSO RECOMEND K VERSIONS OF TOP N. TOP TRAIN, TOP TEST, TOP SUMMATION, TOP WITH WEIGHTED PARAMETERS, SUCH AS CRIPPLE the larger's contributin to overall score based on the ones this pair is compared to and visa versa

            balanced_score=( (test_score*self.test_score_weight+train_score*self.train_score_weight)*self.train_test_score_sum_reward_multiplier ) - ( abs(test_score-train_score)*self.train_test_score_diff_penalty_multiplier )
        

            self.fit_data[cur_degree]={}
            self.fit_data[cur_degree]['test_score']=test_score
            self.fit_data[cur_degree]['train_score']=train_score
            self.fit_data[cur_degree]['test_train_balance_score']=balanced_score            
            self.fit_data[cur_degree]['degree']=cur_degree
            self.fit_data[cur_degree]['params']=grid.best_params_
            self.fit_data[cur_degree]['model']=best_model

            labels.append(cur_degree)
            test.append(test_score)
            train.append(train_score)
            balance.append(balanced_score)

        self.fit_plot_data=pd.DataFrame({'poly_degrees':labels,'test_scores':test,'train_scores':train,'balanced_score':balance})
        self.model_poly_degree=None
        top_score=-float('inf')
        self.model=None
        for deg in self.fit_data.keys():
            if self.fit_data[deg]['test_train_balance_score']>top_score:
                self.model=self.fit_data[deg]['model']
                top_score=self.fit_data[deg]['test_train_balance_score']
                self.model_poly_degree=self.fit_data[deg]['degree']

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def fit_data_plot(self):
        plt.figure(figsize=(8,6))
        x=self.fit_plot_data['poly_degrees']
        sns.barplot(x=x,y=self.fit_plot_data['train_scores'],label='train_scores')
        sns.barplot(x=x,y=self.fit_plot_data['test_scores'],label='test_scores')
        sns.barplot(x=x,y=self.fit_plot_data['balanced_score'],label='balanced_scores')
        plt.legend()
        plt.grid()
        plt.show()


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def poly_scale_split(X_data,y_data,degree=1):

        X_cols=X_data.columns
        poly=PolynomialFeatures(degree=degree,include_bias=True)
        X_data_fit=poly.fit_transform(X_data)
        X_poly_df=pd.DataFrame(X_data_fit, columns=poly.get_feature_names_out(list(X_cols)))
        X_poly_df_cols=X_poly_df.columns

        X_train,X_test,y_train,y_test = train_test_split(X_poly_df,y_data,test_size=.4,random_state=32)

        scaler=StandardScaler()
        poly_X_train=scaler.fit_transform(X_train)
        poly_X_test=scaler.transform(X_test)

        

        return poly_X_test,y_test,X_cols,X_poly_df_cols,scaler


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def get_performance_metrics(X_test,y_test,hyper_tuned_model=None,num_splits=2):
    split_size=np.ceil(X_test.shape[0]/num_splits)
    edges=[i for i in range(0,X_test.shape[0],int(split_size))]     
    scores_dict={'R2':[],'MAE':[],'Max_abs_residual':[],'Min_abs_residual':[]}    
    for i in range(len(edges)):
        low_row=edges[i]
        high_row=X_test.shape[0] if i+1>len(edges)-1 else edges[i+1]
        X_train=X_test[low_row:high_row,:]
        y_train=y_test[low_row:high_row]
        train_pred=hyper_tuned_model.predict(X_train)
        test_residuals=np.abs(y_train-train_pred)
        scores_dict['MAE'].append(mean_absolute_error(y_train,train_pred))
        scores_dict['R2'].append(r2_score(y_train,train_pred))
        mx_residual=np.max(test_residuals)
        mn_residual=np.min(test_residuals)
        scores_dict['Max_abs_residual'].append(mx_residual)
        scores_dict['Min_abs_residual'].append(mn_residual)
    return scores_dict


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_tables(scores_dict):
    score_table=pd.DataFrame(scores_dict)
    test_r2_var=score_table['R2'].var()
    test_MAE_var=score_table['MAE'].var()
    avg_test_r2=score_table['R2'].mean()
    avg_test_mae=score_table['MAE'].mean()  
    mx_residual=score_table['Max_abs_residual'].max()
    mn_residual=score_table['Min_abs_residual'].min()
    var_table=pd.DataFrame({'r2_var':[test_r2_var],'avg_r2':[avg_test_r2],'mae_var':[test_MAE_var],'avg_mae':[avg_test_mae],'max_abs_residual':[mx_residual],'min_abs_residual':[mn_residual]})
    return score_table,var_table 


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def get_bias_variance_plots(scores_dict):
    plots=[key for key in scores_dict.keys()]
    x=[i for i in range(len(scores_dict[plots[0]]))]
    cols=3
    rows=int(np.ceil(len(plots)/cols))
    plt.figure(figsize=(12,6))
    for i in range(len(plots)):
        #plt.subplot(rows,cols,i+1)
        sns.lineplot(x=x,y=scores_dict[plots[i]],label=plots[i])
    plt.legend()
    plt.grid()
    plt.show()


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def age_gender_rename(data):
    age_gender_dict=joblib.load('age_gender_dict.joblib')

    for key,val in age_gender_dict.items():
        data[key]=data[key].apply(lambda x: val[x])
    return data

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def make_predictions_pipeline(data,model_info_dict,cap_dict=None):
    """
    returns a new dataframe with an added prediction column
    """
    poly=PolynomialFeatures(degree=model_info_dict['degree'])
    X_features=data[model_info_dict['X_columns']]
    
    if cap_dict:
        for feature,val in cap_dict.items():
            if feature=='Spend': continue
            data.loc[data[feature]>val,feature]=val
    
    X_poly=poly.fit_transform(X_features)
    X_scaled_poly=model_info_dict['scaler'].transform(X_poly)
    y_pred=model_info_dict['model'].predict(X_scaled_poly)
    data['pred_'+model_info_dict['y_col']]=y_pred
    return data


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def leads_sales_plots(data):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    demographic_data=data.groupby(['age','gender','interest'],as_index=False)[['Spent','Total_Conversion','pred_Total_Conversion','pred_Approved_Conversion','Approved_Conversion']].sum()
    demographic_data['demographic']=demographic_data.index
    
    X = round(demographic_data['Spent'])
    Y = demographic_data['demographic'] 
    Z = demographic_data['Total_Conversion'] 
    Z2 = round(demographic_data['pred_Total_Conversion'])
    Z3 = demographic_data['Approved_Conversion']  
    Z4 = round(demographic_data['pred_Approved_Conversion'])

    # Create subplot container with two 3D plots
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]])

    # Plot 1: Leads
    fig.add_trace(
        go.Scatter3d(x=X, y=Y, z=Z, mode='markers', marker=dict(color='black', opacity=0.6),name='Actual Leads'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter3d(x=X, y=Y, z=Z2, mode='markers', marker=dict(color='yellow', opacity=0.6),name='Predicted Leads'),
        row=1, col=1
    )

    # Plot 2: Sales
    fig.add_trace(
        go.Scatter3d(x=X, y=Y, z=Z3, mode='markers', marker=dict(color='black', opacity=0.6),name='Actual Sales'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter3d(x=X, y=Y, z=Z4, mode='markers', marker=dict(color='yellow', opacity=0.6),name='Predicted Sales'),
        row=1, col=2
    )

    # Customize axes and titles
    fig.update_layout(
        height=600, width=1200,
        title_text="Plot_One:Leads; Plot_Two: Sales",
        scene=dict(
            xaxis_title='X = Spent',
            yaxis_title='Y = Demographic Combinations',
            zaxis_title='Z = Leads'
        ),
        scene2=dict(
            xaxis_title='X = Spent',
            yaxis_title='Y = Demographic Combinations',
            zaxis_title='Z = Sales'
        )
    )

    fig.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def simple_plots_leads_sales(data):
    demographic_data=data.groupby(['age','gender','interest'],as_index=False)[['Spent','Total_Conversion','pred_Total_Conversion','pred_Approved_Conversion','Approved_Conversion']].sum()
    demographic_data['demographic']=demographic_data.index
        
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax2 = fig.add_subplot(1,2,2,projection='3d')

    X = demographic_data['Spent']
    Y = demographic_data['demographic'] 
    Z = demographic_data['Total_Conversion'] 
    Z2 = demographic_data['pred_Total_Conversion']
    Z3 = demographic_data['Approved_Conversion']  
    Z4 = demographic_data['pred_Approved_Conversion']

    ax.scatter(X, Y, Z, alpha=0.6,color='black')
    ax.scatter(X,Y,Z2, alpha=0.6,color='yellow')
    ax.set_xlabel('Spent')
    ax.set_ylabel('Demographic Combinations')
    ax.set_zlabel('Leads')
    ax.set_title('Leads by Spent and Demographic')
    ax2.scatter(X, Y, Z3, alpha=0.6,color='black')
    ax2.scatter(X,Y,Z4, alpha=0.6,color='yellow')
    ax2.set_xlabel('Spent')
    ax2.set_ylabel('Demographic Combinations')
    ax2.set_zlabel('Sales')
    ax2.set_title('Sales by Spent and Demographic')
    fig.subplots_adjust(left=0.1, right=0.5, bottom=0.1, top=0.9)
    plt.show()




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#PREPROCESS TARGET
def observed_rate(data,numerator,denominator,num_classes,observed_class_name):
    multiplier=100/num_classes if num_classes>=10 else 1
    decimal=2 if num_classes>=10 else 1
    data[observed_class_name]=data[numerator].div(data[denominator])
    data[observed_class_name] = ((((data[observed_class_name].replace([np.inf, -np.inf], np.nan).fillna(0)).round(decimal)*100)//multiplier)*multiplier)/100
    return data

def classify(data,num_classes,class_name,observed_class_name):
    edges=np.linspace(0,1,num_classes+1)
    classes=[(round(edges[i],2),round(edges[i+1],2)) for i in range(len(edges)-1)]

    data[class_name]='temp'
    for cls in classes:
        mask = ( (data[observed_class_name]>=cls[0]) & ( (data[observed_class_name]<cls[1]) | ((data[observed_class_name]==cls[1])&(cls[1]==classes[-1][1]))) )
        #mask_len=data.loc[mask].shape[0]  
        #new_vals=pd.Series([cls[0]]*mask_len)
        new_vals=cls[0]
        data.loc[mask,class_name]=new_vals
    return data

def get_prob_class(data,numerator,denominator,class_name,num_classes,observed_class_name):
    data=observed_rate(data,numerator,denominator,num_classes,observed_class_name)
    data=classify(data,num_classes,class_name,observed_class_name)
    return data

def plot_cnts(df,col):
    plot_data=df[col].value_counts()
    sns.barplot(x=plot_data.index,y=plot_data.values)
    plt.show()



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def add_conversion_rate_prediction(data):
    classifier=joblib.load('3_class_conversion_predictor_model')
    for poly in [5]:
        multiplyer=data['CostPerMille']
        col=(multiplyer**poly)
        mean=col.mean()
        std=col.std()
        data[f'CostPerMille**{poly}']=(col-mean)/std
    data['3_Class_Predict']=classifier.predict(data[['age','gender','interest','CostPerMille','CostPerMille**5']])
    data['Three_Conversion_Classes']=(   ((  data['Three_Conversion_Classes']*1_000)//1).astype(int)).astype(str)

    data=data.drop(columns='CostPerMille**5')
    return data


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def class_probas(data):
    classifier=joblib.load('3_class_conversion_predictor_model')
    for poly in [5]:
        multiplyer=data['CostPerMille']
        col=(multiplyer**poly)
        mean=col.mean()
        std=col.std()
        data[f'CostPerMille**{poly}']=(col-mean)/std
    class_probas=classifier.predict_proba(data[['age','gender','interest','CostPerMille','CostPerMille**5']])
    data=data.drop(columns='CostPerMille**5')
    return class_probas



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def plot_classification(data,major_title):
    probs=class_probas(data)
    true_bins=np.zeros((data.shape[0],3))
    roc_data=data.copy()
    roc_data['Three_Conversion_Classes'].replace(('0','330','670'),('Low','Medium','High'),inplace=True)
    for i,v in enumerate(['Low','Medium','High']):
        true_bins[:,i]=(roc_data['Three_Conversion_Classes'].astype(str).to_numpy()==v).astype(int)
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr = {}
    tpr = {}
    roc_auc = {}
    classes=['Low','Medium','High']
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(true_bins[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    from IPython.display import display, HTML
    display(HTML(f"<p style='font-size:30px;'>{major_title}</p>"))
    plt.figure(figsize=(12,4))
    plt.title("",fontsize=30)

    plt.title("Low(1), Medium(2), or High(3) Odds of Conversion Given a Lead\n\n")


    plt.subplot(1,3,1)
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], label=f"{classes[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plot_data=data.copy()
    plot_data['Three_Conversion_Classes'].replace(('0','330','670'),(1,2,3),inplace=True)
    plot_data['3_Class_Predict'].replace(('0','330','670'),(1,2,3),inplace=True)
    plot_data.sort_values(by='CostPerMille',inplace=True)
    x_ticks=[1,2,3]
    y_ticks=list(np.linspace(0,.35,6))
    plt.subplot(1,3,2)
    sns.barplot(data=plot_data,y='CostPerMille',x='Three_Conversion_Classes')
    plt.xticks(x_ticks)
    plt.grid()
    plt.yticks(y_ticks)
    plt.title('Actual')
    plt.subplot(1,3,3)
    sns.barplot(data=plot_data,y='CostPerMille',x='3_Class_Predict')
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.title('Predicted')

    plt.grid()
    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def unseen_data_report():   
    plt.figure(figsize=(14,4))
    plt.title('Model Performance on Test Data\n\n')
    plt.subplot(1,2,1)
    plt.title('Lead Predictions Across Unseen Data')
    plt.imshow(plt.imread('Lead_pred_across_unseen_data.png'))
    plt.subplot(1,2,2)
    plt.title('Sales Predictions Across Unseen Data')
    plt.imshow(plt.imread('Approved_Conversion_pred_across_unseen_data.png'))


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def confusion_matrix(data):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    plot_data=data.copy()
    plot_data['3_Class_Predict'].replace(('0','330','670'),('Low','Medium','High'),inplace=True)
    plot_data['Three_Conversion_Classes'].replace(('0','330','670'),('Low','Medium','High'),inplace=True)
    cm = confusion_matrix(plot_data['Three_Conversion_Classes'], plot_data['3_Class_Predict'], labels=['Low','Medium','High'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low','Medium','High'],
                yticklabels=['Low','Medium','High'],
                cbar=False,square=True,linewidth=0.5)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def parallel_plot(data):
    import plotly.express as px
    plot_data=data.copy()    
    plot_data['3_Class_Predict'].replace(('0','330','670'),('Low','Medium','High'),inplace=True)
    plot_data['Three_Conversion_Classes'].replace(('0','330','670'),('Low','Medium','High'),inplace=True)
    df = pd.DataFrame({
        'Actual': plot_data['Three_Conversion_Classes'],
        'Predicted': plot_data['3_Class_Predict']
    })


    # Color by Actual class
    fig = px.parallel_categories(df, dimensions=['Actual', 'Predicted'], color=df['Actual'].astype('category').cat.codes,
                                color_continuous_scale=px.colors.sequential.Reds)

    fig.update_layout(title='Actual vs Predicted Categories')
    fig.show()



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_conversion_rate_preds(data):
    mod=joblib.load('scv_59_60_f1s.pkl')
    y_pred=mod.predict(data[['age','gender','interest','CostPerMille']])
    data['3_Class_Predict']=y_pred
    return data


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------





