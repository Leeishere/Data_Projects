


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# CAP MAX VALUES WITH NaN

# a function to find outliers that exceed a max threshold. 

def cap_max_w_nan(data,to_cap={'column_name':'maximum value'}):
    '''
    input the dataframe and a dictionary of columns and maximum values
    values that exceed the maximum will be replaced with np.nan
    returns a modified dataframe
    '''
    for i in data.columns:
        cap=to_cap.get(i,None)
        if cap is not None:        
            data.loc[data[i]>cap,i]=np.nan
    return data


#FILL NaN

# fill nan function that considers surrounding rows

def neighbor_fillna(most_important_features,dataframe,window_size):
    '''
    neighbors with nan will be filtered out prior to sorting. Hence, it is necessary to consider the relations of neighbor nans to each column
    This is iterative, so may not be ideal when fast performance is needed
    most_important_features is a list of features, ordered from most to least important
    window_size: averages will be taken from the center of the window, it should be >3
    returns a modified datafram
    '''

    null_counts=dataframe.isnull().sum()
    columns_w_null=null_counts.loc[null_counts.values>0].index   #a list of columns with nulls
    
    for null_col in columns_w_null:
        sort_cols=[col for col in most_important_features if col != null_col] 
        for feature in sort_cols:  
            if feature == 'Target':                                                     
                dataframe['temp']=dataframe['Target']                                       # create a temp column to store the original target because we dont want to return a modified target
                dataframe['Target']=dataframe['Target'].apply(lambda x: round(x,2))          #round target to 10 thousands place so that there is more influence from other 'feature' columns.                             
            dataframe=dataframe.loc[~dataframe[feature].isnull()]                     #filter out features with nan
        sorted_df=dataframe.sort_values(by=sort_cols).reset_index(drop=True)          # a sorted dataframe based on important features (not the column we are targeting with fill values in this iteration)
        nan_indexes=sorted_df.loc[sorted_df[null_col].isnull()].index                        # these are the indexes we sort through to fill nans
        df=sorted_df
        half_size=window_size//2
        for index in nan_indexes:
            half_size=min(half_size,df.shape[0]-index-1,index)       #calculate window sizes at top and bottom rows of the df by shrinking window, such that it remains centered
            sum_first_half=df.loc[max(0,index-half_size):index,null_col].sum()
            sum_second_half=df.loc[index+1:index+half_size+1,null_col].sum()
            two_halves_sum=sum_first_half+sum_second_half                                                                               # sum of values on both sides of target
            two_halves_len=len(df.loc[max(0,index-half_size):index,null_col])+len(df.loc[index+1:index+half_size+1,null_col])      #number of values on both sides
            fill_value=two_halves_sum/two_halves_len                                                                                    # average value in the window ===> the fill value
            df.loc[index,null_col]=fill_value                                           # fill the nan cell
            dataframe=df
    dataframe['Target']=dataframe['temp']                  # restore column name
    dataframe.drop(columns='temp',inplace=True)
    return dataframe


# CREATE LATITUDE AND LONGITUDE BINS (RESPECTIVELY)
# a function to create bins for latitude and longitude

def get_bins(data,feature,edge_size=0.01,return_bins=False,num_bins=None):
    '''
    if return bins is true it returns the bins, otherwise unique integer label for each bin
    if num bins is false, it returns a list of tuples containing the bins [(mn,mx),(mn,mx)  of len height for edge_size of a square region in degrees, else equally spaced bins
    max bin edge of bins is excluded and larger than max input is used in in end-corner edge case
    '''
    func_mn=(data[feature].min())-(2*edge_size)
    func_mx=(data[feature].max())+(2*edge_size)
    if num_bins is not None:
        func_bin_boundaries=np.linspace(func_mn,func_mx+0.001,num_bins+1)
    else:
        func_bin_boundaries=np.arange(func_mn,func_mx,edge_size)
    func_bins=[(func_bin_boundaries[index-1],func_bin_boundaries[index]) for index in range(1,len(func_bin_boundaries))]
    if return_bins==True: return func_bins
    #helper function
    def place_in_bin(func_var,bin_and_category=False):
        hf_l=0
        hf_r=len(func_bins)   -1     
        while hf_l<=hf_r:
            hf_mid=(hf_l+hf_r)//2
            if func_var>=func_bins[hf_mid][1]:
                hf_l=hf_mid+1
            elif func_var<func_bins[hf_mid][0]:
                hf_r=hf_mid-1
            else:
                if bin_and_category==False:
                    return func_bins[hf_mid]   # to return the bin
                else: return hf_mid+1           # returns the index as a number label for the category bin
                
        return None
    


#RETRIEVE REGION BASED ON LATITUDE AND LONGITUDE

# function to return a region based on latitude and longitude input
# 3 functions. the first 2 are combined inside the 3rd, hence only the 3rd what is called

def get_latitude_position(latitude, lat_long_tuple_keys):
    '''
    Binary search for the latitude bin index.
    '''
    l, r = 0, len(lat_long_tuple_keys) - 1
    while l <= r:
        mid = (l + r) // 2
        lat_min, lat_max = lat_long_tuple_keys[mid][0][0]
        if latitude >= lat_max:
            l = mid + 1
        elif latitude < lat_min:
            r = mid - 1
        else:
            return mid
    return None

def check_subset(target_latitude, target_longitude, found_subset_index, lat_long_tuple_keys):
    '''
    Binary search for the longitude bin within the found latitude bin.
    '''
    if found_subset_index is None:
        return np.nan

    # Find all bins with the same latitude range as the found index
    lat_min, lat_max = lat_long_tuple_keys[found_subset_index][0][0]
    # Find the range of indices with this latitude bin
    n = len(lat_long_tuple_keys)
    left = found_subset_index
    while left > 0 and lat_long_tuple_keys[left-1][0][0] == (lat_min, lat_max):
        left -= 1
    right = found_subset_index
    while right + 1 < n and lat_long_tuple_keys[right+1][0][0] == (lat_min, lat_max):
        right += 1

    # Binary search for longitude within this latitude bin range
    l, r = left, right
    while l <= r:
        mid = (l + r) // 2
        lon_min, lon_max = lat_long_tuple_keys[mid][0][1]
        if target_longitude >= lon_max:
            l = mid + 1
        elif target_longitude < lon_min:
            r = mid - 1
        else:
            return lat_long_tuple_keys[mid][1]
    return np.nan

def get_region(latitude, longitude, pickle_data):
    '''
    Returns the region value for a given latitude and longitude using the lookup data.
    '''
    aproximate_location = get_latitude_position(latitude, pickle_data)
    return check_subset(latitude, longitude, aproximate_location, pickle_data)



# a function to create bins

def get_bins(data,feature,edge_size=0.01,return_bins=False,num_bins=None):
    '''
    if return bins is true it returns the bins, otherwise unique integer label for each bin
    if num bins , it returns a list of tuples containing the bins [(mn,mx),(mn,mx)  of len height for edge_size of a square region in degrees, else equally spaced bins
    max bin edge of bins is excluded and larger than max input is used in end-corner edge case
    '''
    func_mn=(data[feature].min())-(2*edge_size)   # add empty bins at min and max of edges
    func_mx=(data[feature].max())+(2*edge_size)
    if num_bins is not None:
        func_bin_boundaries=np.linspace(func_mn,func_mx+0.001,num_bins+1)
    else:
        func_bin_boundaries=np.arange(func_mn,func_mx,edge_size)
    func_bins=[(func_bin_boundaries[index-1],func_bin_boundaries[index]) for index in range(1,len(func_bin_boundaries))]
    if return_bins==True: return func_bins
    #helper function
    def place_in_bin(func_var,bin_and_category=False):
        hf_l=0
        hf_r=len(func_bins)-1     
        while hf_l<=hf_r:
            hf_mid=(hf_l+hf_r)//2
            if func_var>=func_bins[hf_mid][1]:
                hf_l=hf_mid+1
            elif func_var<func_bins[hf_mid][0]:
                hf_r=hf_mid-1
            else:
                if bin_and_category==False:
                    return func_bins[hf_mid]   # to return the bin
                else: return hf_mid+1           # returns the index as a number label for the category bin, hence add 1 to exclude 0
                
        return None
    
    #add bin column
    new_col=feature+'_binned'
    cat_col=feature+'_category'
    data[new_col]=data[feature].apply(place_in_bin)
    data[cat_col]=data[feature].apply(lambda x: place_in_bin(x,True))
    if return_bins==False:
        return data
    



# a function to print scatterplots

def plot_pred(data,target,pred,plot_size=3):
    '''
    all columns except pred and observed will be plotted against pred and observed in a scatter plot
    plot_pred(data,target,pred,plot_size=3)
    '''
    maxfigwidth=20
    num_plots=len(data.columns)-2
    edgesize=np.ceil(np.sqrt(num_plots))*plot_size
    cols=int(min(np.ceil(edgesize/plot_size),np.ceil(maxfigwidth/plot_size)))
    rows=int(np.ceil(num_plots/cols))

    plots=[i for i in data.columns if i!= target and i!= pred]

    plt.figure(figsize=(cols*plot_size,rows*plot_size))
    plt.title('predicted (orange) over observed (blue)')
    for i in range(1,num_plots+1):
        plot=plots[i-1]
        plt.subplot(rows,cols,i)
        sns.scatterplot(data=data,x=plot,y=target,color='blue')
        sns.scatterplot(data=data,x=plot,y=pred,color='orange')
        plt.xlabel(f'{plot}')    
    plt.tight_layout()
    plt.show()