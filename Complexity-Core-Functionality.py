
"Importing the libraries needed to perform operations"

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import statistics 
import warnings
warnings.filterwarnings("ignore")
from product_space_gen import *
import json
import re
import networkx as nx
from itertools import count
from itertools import combinations
from itertools import product
import collections 
import pickle
import urllib.request, json
import geopandas
from tqdm import tqdm
import time
import seaborn as sns
import seaborn as sns

# Set the year for further analysis
year = 2021

#%%

# Upload the CEPPI file with import/export trade flow for a specific year
df_orig = pd.read_csv('BACI_HS92_Y%s_V202301.csv' % (year))
df_orig.rename(columns={"t":"Year", "i":"Exporter", "j":"Importer", "k":"Product_code", "v":"Value", "q":"Quantity"}, inplace = True)

# Total product export (eliminating destination diversity):
df_export = df_orig.groupby(['Year', 'Exporter', 'Product_code'], as_index=False)['Value'].sum()
df_import = df_orig.groupby(['Year', 'Importer', 'Product_code'], as_index=False)['Value'].sum()

def calc_rca(data, country_col, product_col, value_col, time_col):
    """
    Calculates: Revealed Comparative Advantage (RCA) of country-product-time combinations
    Returns: Pandas dataframe with RCAs
    """

    # Aggregate to country-product-time dataframe
    print('creating all country-product-time combinations')
    # - else matrices later on will have missing values in them, complicating calculations
    df_all = pd.DataFrame(list(product(data[time_col].unique(), data[country_col].unique(),data[product_col].unique())))
    df_all.columns = [time_col,country_col,product_col]
    
    print('merging data in')
    df_all = pd.merge(df_all, data[[country_col, product_col, value_col, time_col]], how='left', on=[time_col, country_col, product_col])
    # - add all possible products for each country with export value 0
    df_all.loc[df_all[value_col].isnull(),value_col] = 0
    
    # Calculate the properties
    print('calculating properties')
    df_all['Xcpt'] = df_all[value_col]
    df_all['Xct'] = df_all.groupby([country_col, time_col])[value_col].transform(sum)
    df_all['Xpt'] = df_all.groupby([product_col, time_col])[value_col].transform(sum)
    df_all['Xt'] = df_all.groupby([time_col])[value_col].transform('sum')
    df_all['RCAcpt'] = (df_all['Xcpt']/df_all['Xct'])/(df_all['Xpt']/df_all['Xt'])
    # set to 0 if missing, e.g. if product / country have 0 (total) exports
    df_all.loc[df_all['RCAcpt'].isnull(),'RCAcpt'] = 0
    # drop the properties 
    df_all.drop(['Xcpt','Xct','Xpt','Xt'],axis=1,inplace=True,errors='ignore')
    
    return df_all

# Calculating an export-based RCA dataframe
df_exp_rca = calc_rca(data = df_export, country_col="Exporter", product_col="Product_code", value_col="Value", time_col="Year")
# Calculating an import-based RCA dataframe
df_imp_rca = calc_rca(data = df_import, country_col="Importer", product_col="Product_code", value_col="Value", time_col="Year")

print('rca dataframe ready')
print('df_rca ready')

# Create a separate column with values corresponding to respective entries of the binary geographic matrix M_cp
df_exp_rca['Mcp'] = 0
df_exp_rca.loc[df_exp_rca['RCAcpt']>1,'Mcp'] = 1                               # proximity implies intensive import / export
dft_exp = df_exp_rca[df_exp_rca['Year'] == year].copy()
dft_exp = dft_exp[dft_exp['Mcp'] == 1]                                         # Keep only country-product combinations where Mcp == 1 (thus RCAcp > 1)

df_imp_rca['Mcp'] = 0
df_imp_rca.loc[df_imp_rca['RCAcpt']>1,'Mcp'] = 1                               # proximity implies intensive import / export
dft_imp = df_imp_rca[df_imp_rca['Year'] == year].copy()
dft_imp = dft_imp[dft_imp['Mcp'] == 1]                                         # Keep only country-product combinations where Mcp == 1 (thus RCAcp > 1)


#%%

""" Calculates: product co-occurences in countries:     
    Returns: pandas dataframe with co-occurence value for each product pair
"""

def calc_cppt(data,country_col,product_col):

    print('Initiate the proximity calculations')
    # Create combinations within country_col (i.e. countries) of entities (i.e. products)
    dft = (data.groupby(country_col)[product_col]
            .apply(lambda x: pd.DataFrame(list(combinations(x,2))))
            .reset_index(level=1, drop=True)
            .reset_index())
        
    dft.rename(columns={0:f'{product_col}_1'}, inplace=True)
    dft.rename(columns={1:f'{product_col}_2'}, inplace=True)

    # Create second half of matrix (assymmetrical):
    # -- {product_col} 1 X {product_col} 2 == {product_col} 2 X {product_col} 1
    dft2 = dft.copy()
    dft2.rename(columns={f'{product_col}_1':f'{product_col}_2t'}, inplace=True)
    dft2.rename(columns={f'{product_col}_2':f'{product_col}_1'}, inplace=True)
    dft2.rename(columns={f'{product_col}_2t':f'{product_col}_2'}, inplace=True)
    # -- add second half
    dft3 = pd.concat([dft,dft2],axis=0,sort=False)
        
    # Drop diagonal if present
    dft3 = dft3[ dft3[f'{product_col}_1'] != dft3[f'{product_col}_2'] ]
    
    print('Combinations are formed')

    # Now calculate N of times that {product_col}s occur together
    dft3['count'] = 1
    dft3 = dft3.groupby([f'{product_col}_1',f'{product_col}_2'],as_index=False)['count'].sum()
    dft3.rename(columns={f'count':f'Cpp'}, inplace=True)

    # Calculate ubiquity
    df_ub = data.groupby(product_col,as_index=False)['Mcp'].sum()
    
    print('Ubiquity is calculated')
    
    # Merge ubiqity into cpp matrix
    df_ub.rename(columns={f'{product_col}':f'{product_col}_1'}, inplace=True)
    dft3 = pd.merge(dft3,df_ub,how='left',on=f'{product_col}_1')
    df_ub.rename(columns={f'{product_col}_1':f'{product_col}_2'}, inplace=True)
    dft3 = pd.merge(dft3,df_ub,how='left',on=f'{product_col}_2')
    
    print('Calculating proximities')

    # Take minimum of conditional probabilities
    dft3['kpi'] = dft3['Cpp']/dft3['Mcp_x']
    dft3['kpj'] = dft3['Cpp']/dft3['Mcp_y']
    dft3['phi'] = dft3['kpi']
    dft3.loc[dft3['kpj']<dft3['kpi'],'phi'] = dft3['kpj']

    return dft3

df_cppt_exp = calc_cppt(dft_exp, country_col='Exporter', product_col='Product_code')
print('Exports data analysis was successful')
# Saving data into pickle to avoid repetitive calculations
df_cppt_exp.to_pickle("export_matrix_%s.pkl" % (year))

df_cppt_imp = calc_cppt(dft_imp, country_col='Importer', product_col='Product_code')
print('Imports data analysis was successful')
# Saving data into pickle to avoid repetitive calculations
df_cppt_imp.to_pickle("import_matrix_%s.pkl" % (year)) 


#%%
# Importing the data back / Comment out the above if runnint the code all-together
df_cppt_imp = pd.read_pickle("import_matrix_%s.pkl" % (year))
df_cppt_exp = pd.read_pickle("export_matrix_%s.pkl" % (year))

# Importing historic PCI data
pci_df = pd.read_csv('pci_hs6_hs92.csv')

# Calculatung parameters that are needed for further normalisation
delta_pci_year = max(pci_df['%s' % (year)]) - min(pci_df['%s' % (year)])
min_pci_year = min(pci_df['%s' % (year)])

# Importing the list of country codes
df_countries = pd.read_csv('country_codes_V202301.csv')
    
#%%
''' Import and Process Hydrogen Space '''

# Establish a lower boundary on traded amounts to avoid the distortion of the results
minimum_traded_amount = 500000 #in thousands USD

# Import the H2 product list
df_H2_space_original = pd.read_csv('H2-product-list.csv')

# Identifying trade flow value of hydrogen-related products | gross across imports/exports should be the same
H2_product_trade = df_export.loc[df_export['Product_code'].isin(df_H2_space_original['Product_code'])]
H2_product_trade_total = H2_product_trade.groupby(['Product_code'], as_index=False)['Value'].sum()

df_H2_space = pd.merge(df_H2_space_original, H2_product_trade_total, on='Product_code')
df_H2_space = df_H2_space[df_H2_space['Value'] >= minimum_traded_amount]
df_H2_space = df_H2_space.reset_index(drop = True)

# Calculating product-to-product proximities in the hydrogen space 
n = len(df_H2_space['Product_code'])
H2_proximity_matrix_export = np.zeros([n, n])
H2_proximity_matrix_import = np.zeros([n, n])

for i, h2_product_1 in tqdm(enumerate(df_H2_space['Product_code'])):
    for j, h2_product_2 in enumerate(df_H2_space['Product_code']):
        if i == j:
            H2_proximity_matrix_export[i, j] = 0
            H2_proximity_matrix_import[i, j] = 0
        else:
            index1 = (df_cppt_exp['Product_code_1'] == h2_product_1)
            index2 = (df_cppt_imp['Product_code_1'] == h2_product_1)
            H2_proximity_matrix_export[i, j] = df_cppt_exp.loc[index1 & (df_cppt_exp['Product_code_2'] == h2_product_2), 'phi'].astype(float).mean()
            H2_proximity_matrix_import[i, j] = df_cppt_imp.loc[index2 & (df_cppt_imp['Product_code_2'] == h2_product_2), 'phi'].astype(float).mean()

#%%

def distance_calc(country, rca_dataset, cppt_dataset, calc_type, weighting = False):
    
    """ Product-to-country distance calculation """
    
    country_index = int(df_countries[df_countries['country_name_abbreviation'] == country]['country_code'])
    country_export_products = set(rca_dataset[(rca_dataset['Year'] == int(year)) & (rca_dataset[calc_type] == country_index) & (rca_dataset['Mcp'] == 1)].sort_values(by=['RCAcpt'],ascending=False)['Product_code'])
    distance = []
    
    for i, h2_product in tqdm(enumerate(df_H2_space['Product_code'])):
        index = (cppt_dataset['Product_code_1'] == h2_product)
            
        no_export_products = cppt_dataset.loc[index & ~cppt_dataset['Product_code_2'].isin(country_export_products), 'phi'].astype(float)
        dummy = sum(no_export_products)
        total = cppt_dataset[index]['phi'].sum()
        
        distance.append(dummy / total)
            
    return distance

# An example of countries for which to perform distance calculations
country_list = ['Mexico', 'Thailand', 'Norway']

imp_distance_df = pd.DataFrame()
imp_distance_df['Product_code'] = df_H2_space['Product_code']
exp_distance_df = pd.DataFrame()
exp_distance_df['Product_code'] = df_H2_space['Product_code']

# Perofming country-to-H2 space calculations:
for country in tqdm(country_list):
        
    country_index = int(df_countries[df_countries['country_name_abbreviation'] == country]['country_code'])

    index1 = country + '_Import'
    imp_distance_df[index1] = distance_calc(country, df_imp_rca, df_cppt_imp, 'Importer')
    
    index2 = country + '_Export'
    exp_distance_df[index2] = distance_calc(country, df_exp_rca, df_cppt_exp, 'Exporter')
    
    dummy = df_import[(df_import['Importer'] == country_index) & (df_import['Product_code'].isin(df_H2_space['Product_code']))][['Product_code', 'Value']]
    dummy = dummy.rename(columns={"Value": "%s_Value_Import" % (country)})
    imp_distance_df = imp_distance_df.merge(dummy[['Product_code', "%s_Value_Import" % (country)]], on=['Product_code'], how='left')

    dummy = df_export[(df_export['Exporter'] == country_index) & (df_export['Product_code'].isin(df_H2_space['Product_code']))][['Product_code', 'Value']]
    dummy = dummy.rename(columns={"Value": "%s_Value_Export" % (country)})
    exp_distance_df = exp_distance_df.merge(dummy[['Product_code', "%s_Value_Export" % (country)]], on=['Product_code'], how='left')

    imp_distance_df.to_pickle("%s/distance_import_%s.pkl" % (year, country))
    exp_distance_df.to_pickle("%s/distance_export_%s.pkl" % (year, country))

#%%
# Could be adjusted for SSCI and SSCP calculations
threshold_rca = 1 

def ssci_thresholds(country_dataset, aggregate, colors):
    """ 
    Calculating Sub-Sectoral Complexity Potential; 
    Different levels of aggregation correspond to different levels of tech grouping, where 0 - the lowest, 1 - the highest 
    
    """
    
    threshold = []

    for production_type in colors_dark:
        pci = (country_dataset[country_dataset['Color'] == production_type]['PCI_%d' % (year)] - min_pci_year) / delta_pci_year
        threshold += [threshold_rca * pci.sum() / len(pci)]
 
    if aggregate == 0:
        if len(colors) == 7:
            return threshold
        if len(colors) == 6:
            return threshold[:-1]
        
    if aggregate == 0.5:
        if len(colors) == 5:
            return [(threshold[0] + threshold[1])/2, threshold[2], (threshold[3] + threshold[4])/2, (threshold[3] + threshold[5])/2, (threshold[3] + threshold[6])/2]
        if len(colors) == 4:
            return [(threshold[0] + threshold[1])/2, threshold[2], (threshold[3] + threshold[4])/2, (threshold[3] + threshold[5])/2]
    
    if aggregate == 1:
        if len(colors) == 4:
            return [(threshold[0] + threshold[1] + threshold[2])/3, (threshold[0] + threshold[1] + threshold[3] + threshold[4])/4, (threshold[0] + threshold[1] + threshold[3] + threshold[5])/4, (threshold[0] + threshold[1] + threshold[3] + threshold[6])/4]
        if len(colors) == 3:
            return [(threshold[0] + threshold[1] + threshold[2])/3, (threshold[0] + threshold[1] + threshold[3] + threshold[4])/4, (threshold[0] + threshold[1] + threshold[3] + threshold[5])/4]

def hci_calc(country_dataset, country, aggregate = 0, plot = 1):
        
    """
    
    Calculating HCI and SSCI (depending on the level of aggregation)
    
    """    
    categories = ['Plant', 'Infrastructure', 'Blue', 'Green', 'Wind Turbine', 'Solar PV', 'Nuclear']
    colors_dark = ['Brown', 'Grey', 'Navy', 'Green', 'Dodgerblue', 'Gold', 'Pink']

    country_dataset.fillna(0, inplace=True)
    
    # Preparing the lists
    total_basket = country_dataset['Value'].sum()                              
    basket_values = []
    hci_values = []
    sectoral_distance_values = []
    distance_transitory = []
    hci_transitory_list = []
    length = []
    
    rca = country_dataset['RCAcpt'].apply(lambda x: 0 if x < 1 else 1)
    weight = country_dataset['Weight']
    pci = (country_dataset['PCI_%d' % (year)] - min_pci_year) / (delta_pci_year) # normalising pci values so that they range between 0 and 1;
    hci = rca * pci                                                              # product contribution to the complexity index 
    hci_country = hci.sum()                                                      # hydrogen complexity index 
    
    for production_type in colors_dark:
        value = country_dataset[country_dataset['Color'] == production_type]['Value']
        # 5e3 value can be adjusted depending on the economy size - for visualisation purposes 
        basket_values += [value.sum() / 5e3]                                     # there is no double counting since only one of the import/export basket is considered
    
    # If the category size is too little (e.g., nuclear) remove from the visualisation | division by 10 is something to fiddle with
    if basket_values[-1] < (basket_values[0] / 10) :
        colors_dark = colors_dark[:-1]
    basket_values = []
        
    for production_type in colors_dark:
        
        value = country_dataset[country_dataset['Color'] == production_type]['Value']
        # 2.5e3 value can be adjusted depending on the economy size - for visualisation purposes 
        basket_values += [value.sum() / 2.5e3]                                    
        
        rca2 = country_dataset[country_dataset['Color'] == production_type]['RCAcpt'].apply(lambda x: x if x <= threshold_rca else threshold_rca)
        phi = country_dataset[country_dataset['Color'] == production_type]['Phi']
        pci = (country_dataset[country_dataset['Color'] == production_type]['PCI_%d' % (year)] - min_pci_year) / delta_pci_year
        weight = country_dataset[country_dataset['Color'] == production_type]['Weight']
        
        length += [len(value)]
        hci_transitory = rca2 * pci
        hci_transitory_list += [hci_transitory.sum()]
        hci_values += [hci_transitory.sum() / len(hci_transitory)]
        distance_transitory += [phi.sum()]
        sectoral_distance_values += [phi.sum() / len(phi)]    # the problem with that is that the greater the number of products the more accurate the average is; tried to add weigth
            
    if aggregate == 1:
            
        if len(colors_dark) == 7:                                              # i.e., with nuclear
                        
            combined_hci = [(hci_transitory_list[0]/length[0] + hci_transitory_list[1]/length[1] + hci_transitory_list[2]/length[2])/3, 
                                     (hci_transitory_list[0]/length[0] + hci_transitory_list[1]/length[1] + hci_transitory_list[3]/length[3] + hci_transitory_list[4]/length[4])/4, 
                                     (hci_transitory_list[0]/length[0] + hci_transitory_list[1]/length[1] + hci_transitory_list[3]/length[3] + hci_transitory_list[5]/length[5])/4, 
                                     (hci_transitory_list[0]/length[0] + hci_transitory_list[1]/length[1] + hci_transitory_list[3]/length[3] + hci_transitory_list[6]/length[6])/4] 
               
            combined_distances = [(distance_transitory[0]/length[0] + distance_transitory[1]/length[1] + distance_transitory[2]/length[2])/3, 
                                     (distance_transitory[0]/length[0] + distance_transitory[1]/length[1] + distance_transitory[3]/length[3] + distance_transitory[4]/length[4])/4, 
                                     (distance_transitory[0]/length[0] + distance_transitory[1]/length[1] + distance_transitory[3]/length[3] + distance_transitory[5]/length[5])/4, 
                                     (distance_transitory[0]/length[0] + distance_transitory[1]/length[1] + distance_transitory[3]/length[3] + distance_transitory[6]/length[6])/4]
                
            combined_values = [sum(basket_values[:3]), 
                                sum(basket_values[:2]) + basket_values[3] + basket_values[4], 
                                sum(basket_values[:2]) + basket_values[3] + basket_values[5], 
                                sum(basket_values[:2]) + basket_values[3] + basket_values[6]]
            
            colors_combined = np.array(['Navy', 'Dodgerblue', 'Gold', 'Pink'])
            categories = np.array(['Blue', 'Green Wind', 'Green Solar PV', 'Pink Nuclear'])
            
            hci_values = [hci_country] + list(combined_hci)

            
        else:                                                                  # i.e., without nuclear
                   
            combined_hci = [(hci_transitory_list[0]/length[0] + hci_transitory_list[1]/length[1] + hci_transitory_list[2]/length[2])/3, 
                                     (hci_transitory_list[0]/length[0] + hci_transitory_list[1]/length[1] + hci_transitory_list[3]/length[3] + hci_transitory_list[4]/length[4])/4, 
                                     (hci_transitory_list[0]/length[0] + hci_transitory_list[1]/length[1] + hci_transitory_list[3]/length[3] + hci_transitory_list[5]/length[5])/4] 
               
            combined_distances = [(distance_transitory[0]/length[0] + distance_transitory[1]/length[1] + distance_transitory[2]/length[2])/3, 
                                     (distance_transitory[0]/length[0] + distance_transitory[1]/length[1] + distance_transitory[3]/length[3] + distance_transitory[4]/length[4])/4, 
                                     (distance_transitory[0]/length[0] + distance_transitory[1]/length[1] + distance_transitory[3]/length[3] + distance_transitory[5]/length[5])/4] 
               
            combined_values = [sum(basket_values[:3]), 
                                sum(basket_values[:2]) + basket_values[3] + basket_values[4], 
                                sum(basket_values[:2]) + basket_values[3] + basket_values[5]]
            
            colors_combined = np.array(['Navy', 'Dodgerblue', 'Gold'])
            categories = np.array(['Blue', 'Green Wind', 'Green Solar PV'])
            
            hci_values = [hci_country] + list(combined_hci)
            
        if plot == 1:
            
            plt.figure(figsize = (4.5, 4))
            
            thresholds = ssci_thresholds(country_dataset, aggregate, colors_combined)
            print(thresholds)
            
            y_boundary = max(abs((combined_hci - np.mean(combined_hci)) / np.std(combined_hci)))
            x_boundary = max(abs((combined_distances - np.mean(combined_distances)) / np.std(combined_distances)))
            
            plt.scatter((combined_distances - np.mean(combined_distances)) / np.std(combined_distances), (combined_hci - np.mean(combined_hci)) / np.std(combined_hci), marker = 'o', s = combined_values, c = colors_combined, label = categories)
            plt.xlabel('PM Distance to the Economy', fontdict=font)
            plt.ylabel('PM Complexity Index', fontdict=font)
            plt.title('%s' % (country), fontdict=font)
            plt.xlim(-x_boundary * 1.1, x_boundary * 1.1)
            plt.ylim(-y_boundary * 1.1, y_boundary * 1.1)
            plt.axvline(x=0, color='grey', linestyle='-', alpha = 0.5)
            plt.axhline(y=0, color='grey', linestyle='-', alpha = 0.5)
            plt.show()
                            
        return hci_values, combined_distances, combined_values
            
    if aggregate == 0.5:

        if len(colors_dark) == 7:
            
            combined_hci = [hci_transitory_list[0]/length[0],hci_transitory_list[1]/length[1], hci_transitory_list[2]/length[2],
                                     (hci_transitory_list[3]/length[3] + hci_transitory_list[4]/length[4])/2, 
                                     (hci_transitory_list[3]/length[3] + hci_transitory_list[5]/length[5])/2, 
                                     (hci_transitory_list[3]/length[3] + hci_transitory_list[6]/length[6])/2]   
                
            combined_distances = [distance_transitory[0]/length[0],distance_transitory[1]/length[1], distance_transitory[2]/length[2],
                                     (distance_transitory[3]/length[3] + distance_transitory[4]/length[4])/2, 
                                     (distance_transitory[3]/length[3] + distance_transitory[5]/length[5])/2, 
                                     (distance_transitory[3]/length[3] + distance_transitory[6]/length[6])/2]   
                                  
            combined_values = [basket_values[0], basket_values[1], basket_values[2],
                                basket_values[3] + basket_values[4], 
                                basket_values[3] + basket_values[5], 
                                basket_values[3] + basket_values[6]]
            
            colors_combined = np.array(['Brown', 'Grey', 'Navy', 'Dodgerblue', 'Gold', 'Pink'])
            categories = np.array(['Plant', 'Infrastructure', 'Blue', 'Green Wind', 'Green Solar PV', 'Pink Nuclear'])
            
            hci_values = [hci_country] + list(combined_hci)

        else:
            
            combined_hci = [hci_transitory_list[0]/length[0],hci_transitory_list[1]/length[1], hci_transitory_list[2]/length[2],
                                     (hci_transitory_list[3]/length[3] + hci_transitory_list[4]/length[4])/2, 
                                     (hci_transitory_list[3]/length[3] + hci_transitory_list[5]/length[5])/2] 
                
            combined_distances = [distance_transitory[0]/length[0],distance_transitory[1]/length[1], distance_transitory[2]/length[2],
                                     (distance_transitory[3]/length[3] + distance_transitory[4]/length[4])/2, 
                                     (distance_transitory[3]/length[3] + distance_transitory[5]/length[5])/2] 


            combined_values = [basket_values[0], basket_values[1], basket_values[2],
                                basket_values[3] + basket_values[4], 
                                basket_values[3] + basket_values[5]]
            
            colors_combined = np.array(['Brown', 'Grey', 'Navy', 'Dodgerblue', 'Gold'])
            categories = np.array(['Plant', 'Infrastructure', 'Blue', 'Green Wind', 'Green Solar PV'])
            
            hci_values = [hci_country] + list(combined_hci)
            
        if plot == 1:
            
            thresholds = ssci_thresholds(country_dataset, aggregate, colors_combined)
            print(thresholds)
            
            y_boundary = max(max(abs((combined_hci - np.mean(combined_hci)) / np.std(combined_hci))), max((thresholds - np.mean(combined_hci)) / np.std(combined_hci)))
            x_boundary = max(abs((combined_distances - np.mean(combined_distances)) / np.std(combined_distances)))
            
            plt.figure(figsize = (5.5, 5))
            plt.scatter((combined_distances - np.mean(combined_distances)) / np.std(combined_distances), (combined_hci - np.mean(combined_hci)) / np.std(combined_hci), marker = 'o', s = combined_values, c = colors_combined, label = categories)
            plt.xlabel('Distance, normalised', fontdict=font)
            plt.ylabel('SSCI, normalised', fontdict=font)
            plt.title('Opportunity screening, %s' % (country), fontdict=font)
            plt.xlim(-x_boundary * 1.1, x_boundary * 1.1)
            plt.ylim(-y_boundary * 1.1, y_boundary * 1.1)
            plt.axvline(x=0, color='grey', linestyle='-', alpha = 0.5)
            plt.axhline(y=0, color='grey', linestyle='-', alpha = 0.5)
            plt.show()
            
            plt.figure(figsize = (0.5, 5))
            for i, value in enumerate(thresholds): 
                plt.axhline(y = (value - np.mean(combined_hci)) / np.std(combined_hci), color = colors_combined[i], linestyle='-')
            plt.ylim(min((combined_hci - np.mean(combined_hci)) / np.std(combined_hci)) * 1.1, max((thresholds - np.mean(combined_hci)) / np.std(combined_hci))*1.1)
            plt.title('CCP', fontdict=font)
            plt.xticks([])
            plt.yticks([])
            plt.show()
                
        return hci_values, combined_distances, combined_values
    
    else:
        
        if plot == 1:
            
            thresholds = ssci_thresholds(country_dataset, aggregate, colors_dark)
            print('Thresholds:', thresholds)


            y_boundary = max(max(abs((hci_values - np.mean(hci_values)) / np.std(hci_values))), max((thresholds - np.mean(hci_values)) / np.std(hci_values)))
            x_boundary = max(abs((sectoral_distance_values - np.mean(sectoral_distance_values)) / np.std(sectoral_distance_values)))
            
            plt.figure(figsize = (5.5, 5))
            
            d = (sectoral_distance_values - np.mean(sectoral_distance_values)) / np.std(sectoral_distance_values)
            ssci = (hci_values - np.mean(hci_values)) / np.std(hci_values)
            
            plt.scatter(d, ssci, marker = 'o', s = basket_values, c = colors_dark, label = categories)
            plt.xlabel('Distance, normalised', fontdict=font)
            plt.ylabel('SSCI, normalised', fontdict=font)
            plt.title('Opportunity screening, %s' % (country), fontdict=font)
            plt.xlim(-x_boundary * 1.1, x_boundary * 1.1)
            plt.ylim(-y_boundary * 1.1, y_boundary * 1.1)
            plt.xticks([-2, -1, 0, 1, 2])
            plt.yticks([-2, -1, 0, 1, 2])
            plt.axvline(x=0, color='grey', linestyle='-', alpha = 0.5)
            plt.axhline(y=0, color='grey', linestyle='-', alpha = 0.5)
            plt.show()
            
            plt.figure(figsize = (0.5, 5))
            for i, value in enumerate(thresholds): 
                plt.axhline(y = (value - np.mean(hci_values)) / np.std(hci_values), color = colors_dark[i], linestyle='-')
            plt.ylim(-y_boundary * 1.1, y_boundary * 1.1)
            plt.title('CCP', fontdict=font)
            plt.xticks([])
            plt.yticks([])
            plt.show()
        
        hci_values = [hci_country] + list(hci_values)
            
    return hci_values, sectoral_distance_values, basket_values


def rca_addition(country_h2_dataset, country, flow_type):

    """ Adding RCA to a hydrogen products dataframe (for convenience) """     

    country_index = int(df_countries[df_countries['country_name_abbreviation'] == country]['country_code'])
    
    if flow_type == 'Import':
        country_set = df_imp_rca[(df_imp_rca['Importer'] == country_index)]
        df_merge = country_h2_dataset.merge(country_set, on = 'Product_code')
        df_merge.drop(columns=['Year', 'Importer', 'Value', 'Mcp'], inplace=True)
        df_merge.rename(columns={'RCAcpt': 'RCAcpt_Import'}, inplace=True)
        return df_merge
        
    if flow_type == 'Export':
        country_set = df_exp_rca[(df_exp_rca['Exporter'] == country_index)]
        df_merge = country_h2_dataset.merge(country_set, on = 'Product_code')
        df_merge.drop(columns=['Year', 'Exporter', 'Value', 'Mcp'], inplace=True)
        df_merge.rename(columns={'RCAcpt': 'RCAcpt_Export'}, inplace=True)
        return df_merge
    
    return country_h2_dataset

    
#%% 

country_list = ['Germany', 'Italy', 'USA', 'Denmark', 'China', 'France', 'Japan', 'United Kingdom', 'Spain', 'Finland', 'Mexico', \
                  'Turkey', 'Rep. of Korea', 'India', 'Thailand', 'Netherlands', 'Norway', 'South Africa', 'Canada', 'Philippines', \
                  'Brazil', 'Indonesia', 'Russian Federation', 'New Zealand', 'United Arab Emirates', 'United Rep. of Tanzania', 'Argentina', 'Oman', 'Australia', \
                  'Kazakhstan', 'Kuwait', 'Iran', 'Chile', 'Saudi Arabia', 'Qatar', 'Algeria', 'Iraq', 'Azerbaijan', 'Malaysia', \
                  'Viet Nam', 'Nigeria', 'Hungary', 'Libya', 'Croatia', 'Angola', 'Tunisia', 'Egypt', 'Austria', 'Czechia', 'Sweden', 'Poland', \
                  'Portugal', 'Ireland', 'Slovenia', 'Sri Lanka', 'Malawi', 'Guatemala', 'Costa Rica', 'Bangladesh', 'Nicaragua', 'Panama', 'Yemen', \
                  'Paraguay', 'Ecuador', 'Jamaica', 'Ghana', 'Estonia', 'Switzerland', 'Romania', 'Slovakia', 'Bulgaria', 'Georgia', 'Pakistan', 'Morocco', 'Lithuania', \
                  'Israel', 'Latvia', 'Lebanon', 'Singapore', 'Ukraine', 'Bosnia Herzegovina', 'Belarus', 'Greece', 'Rep. of Moldova', 'Jordan', 'TFYR of Macedonia', \
                  'Dominican Rep.', 'Honduras', 'Cameroon', 'Albania', 'Colombia', 'Kyrgyzstan', 'Peru', 'Mauritius', 'Zimbabwe', 'Madagascar', 'Uruguay', 'Uzbekistan', \
                  'Mozambique', 'Ethiopia']
    
country_hci_list = np.zeros(len(country_list))

#%%

''' Main Body '''

categories = np.array(['Plant', 'Infrastructure', 'Blue', 'Green', 'Wind Turbine', 'Solar PV', 'Nuclear'])
colors_dark = np.array(['Brown', 'Grey', 'Navy', 'Green', 'Dodgerblue', 'Gold', 'Pink'])
colors_combined = np.array(['Navy', 'Dodgerblue', 'Gold', 'Pink'])
colors_part_combined = np.array(['Brown', 'Grey', 'Navy', 'Dodgerblue', 'Gold', 'Pink'])

# Set std_display to true, if distance needs to be expressed in standard deviation terms
std_display = True
# Set weighted to true, if weigthing is needed to discern between the ranks 
weighted = True

def product_distr(sorted_data_points_1, sorted_data_points_2, country):
    """ For visualisation of the product distances | vertical bars """
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 1))
            
    distance_import_max = max(abs(max(sorted_data_points_1) - np.mean(sorted_data_points_1)), abs(min(sorted_data_points_1) - np.mean(sorted_data_points_1)))
    distance_export_max = max(abs(max(sorted_data_points_2) - np.mean(sorted_data_points_2)), abs(min(sorted_data_points_2) - np.mean(sorted_data_points_2)))
    
    print(distance_import_max)
    
    cmap_exp = 'Reds_r'
    norm_exp = plt.Normalize(0, distance_export_max)
#    norm_exp = plt.Normalize(0, 1)

    cmap_imp = 'Blues_r'
    norm_imp = plt.Normalize(0, distance_import_max)
#    norm_imp = plt.Normalize(0, 1)

    distances_imp = sorted_data_points_1 - np.mean(sorted_data_points_1)
    colors_imp = plt.cm.get_cmap(cmap_imp)(norm_imp(abs(distances_imp)))
    
    distance_exp = sorted_data_points_2 - np.mean(sorted_data_points_2)
    colors_exp = plt.cm.get_cmap(cmap_exp)(norm_exp(abs(distance_exp)))
    
    ax.errorbar(sorted_data_points_1, np.full_like(sorted_data_points_1, 0), yerr=0.05, fmt='none', ecolor=colors_imp, capsize=0)
    ax.errorbar(sorted_data_points_2, np.full_like(sorted_data_points_2, 0), yerr=0.05, fmt='none', ecolor=colors_exp, capsize=0)
    ax.set_xlim(0, 1)
    ax.yaxis.set_visible(False)
    plt.xlabel('Product distance')
    
    plt.show()

def outputs_calc(country, hyperparameter = 1.0, plot = 1, rca_criteria = False):   
    
    imp_dist = pd.read_pickle("%s/distance_import_%s.pkl" % (year, country))
    exp_dist = pd.read_pickle("%s/distance_export_%s.pkl" % (year, country))

    if weighted == True:
        
        index1_3 = country + '_Import_relative_w/o'
        imp_dist.loc[:, index1_3] = (imp_dist[country + '_Import'] - np.mean(imp_dist[country + '_Import'])) / np.std(imp_dist[country + '_Import'])
        index1_2 = country + '_Import_relative'
        imp_dist.loc[:, index1_2] = imp_dist[country + '_Import_relative_w/o'] * (hyperparameter ** (2 - df_H2_space['Weight']))

        index2_3 = country + '_Export_relative_w/o'
        exp_dist.loc[:, index2_3] = (exp_dist[country + '_Export'] - np.mean(exp_dist[country + '_Export'])) / np.std(exp_dist[country + '_Export'])
        index2_2 = country + '_Export_relative'
        exp_dist.loc[:, index2_2] = exp_dist[country + '_Export_relative_w/o'] * (hyperparameter ** (2 - df_H2_space['Weight']))

        # Distances for import and export product spaces (for a specific country)
        data_points_1 = np.array(imp_dist['%s_Import' % (country)])
        data_points_2 = np.array(exp_dist['%s_Export' % (country)])
        
    else:
        
        index1_3 = country + '_Import_relative'
        imp_dist.loc[:, index1_3] = (imp_dist[country + '_Import'] - np.mean(imp_dist[country + '_Import'])) / np.std(imp_dist[country + '_Import'])

        index2_3 = country + '_Export_relative'
        exp_dist.loc[:, index2_3] = (exp_dist[country + '_Export'] - np.mean(exp_dist[country + '_Export'])) / np.std(exp_dist[country + '_Export'])

        data_points_1 = np.array(imp_dist['%s_Import' % (country)])
        data_points_2 = np.array(exp_dist['%s_Export' % (country)])
        
    # Calculate the delta between the import-export distances | to be used for visualisation purposes
    x_delta = abs(data_points_2 - data_points_1)
    
    # Sort the categories based on x_delta in ascending order
    sorted_indices = np.argsort(x_delta)
    sorted_data_points_1 = [data_points_1[i] for i in sorted_indices]
    sorted_data_points_2 = [data_points_2[i] for i in sorted_indices]
    sorted_products = [imp_dist['Product_code'][i] for i in sorted_indices]
    sorted_x_delta = [x_delta[i] for i in sorted_indices]

    # Plotting verticle bar in a heatmap format
    if plot == 1: 
        product_distr(sorted_data_points_1, sorted_data_points_2, country)
    
    # Plotting import/export distances against every product
    if plot == 1: 
        plt.figure(figsize=(6, 18))
        
        for i, product in enumerate(sorted_indices):
            product_color = df_H2_space[df_H2_space['Product_code'] == imp_dist.loc[sorted_indices[i], 'Product_code']]['Color'].values[0]
            plt.scatter(sorted_data_points_1[i], i, marker = '^', color=product_color)
            plt.scatter(sorted_data_points_2[i], i, marker = 's', color=product_color)
        
        plt.axvline(x = np.mean(sorted_data_points_1), color='grey', linestyle='--', alpha = 0.5)
        plt.axvline(x = np.mean(sorted_data_points_2), color='blue', linestyle='--', alpha = 0.5)
    
        plt.yticks(range(len(sorted_indices)), sorted_products)
        plt.ylabel('H2 Product space')
        plt.xlabel('Product distance to the economy')
        plt.title('Import / Export distance to %s' % (country))
        plt.xlim(0.35, 1)
        plt.show()
    
        ''' Histograms for Display '''
        plt.hist(data_points_1, bins = 10, color = 'red', alpha = 0.5)
        plt.axvline(x = np.mean(sorted_data_points_1), color='grey', linestyle='--', alpha = 0.75)
    
        plt.hist(data_points_2, bins = 7, color = 'blue', alpha = 0.5)
        plt.axvline(x = np.mean(sorted_data_points_2), color='blue', linestyle='--', alpha = 0.75)
    
        print('Import average - %.2f ; export - %.2f' % (np.mean(sorted_data_points_1), np.mean(sorted_data_points_2)))
        plt.title('Distribution of products across the trade spaces')
        plt.xlabel('Distance to the economy')
        plt.ylabel('Number of products')
        plt.show()
    
    # Add rca to the dataframes 
    imp_dist = rca_addition(imp_dist, country, 'Import')
    exp_dist = rca_addition(exp_dist, country, 'Export')
        
    # Plot the product space in a product network representation
    if plot == 1:
        product_space_plot(H2_proximity_matrix_export, exp_dist['RCAcpt_Export'], 'Export', df_H2_space, rca = 0, seed = 91) # selection of a seed that pictures the space best
        product_space_plot(H2_proximity_matrix_import, imp_dist['RCAcpt_Import'], 'Import', df_H2_space, rca = 0, seed = 70) # selection of a seed that pictures the space best

    merged_df1 = pd.merge(imp_dist, exp_dist, on='Product_code')
    merged_df1['RCAcpt'] = merged_df1[['RCAcpt_Import', 'RCAcpt_Export']].max(axis=1)
            
    if std_display == True:
        merged_df1['Phi'] = merged_df1[['%s_Import_relative' % (country), '%s_Export_relative' % (country)]].min(axis=1)
        merged_df1['Direction'] = merged_df1.apply(lambda row: 'Import' if row['Phi'] == row['%s_Import_relative' % (country)] else 'Export', axis=1)
        merged_df1['Value'] = merged_df1.apply(lambda row: row[('%s_Value_Import' % (country))] if row['Phi'] == row['%s_Import_relative' % (country)] else row[('%s_Value_Export' % (country))], axis=1)

    else:
        merged_df1['Phi'] = merged_df1[['%s_Import' % (country), '%s_Export' % (country)]].min(axis=1)
        merged_df1['Direction'] = merged_df1.apply(lambda row: 'Import' if row['Phi'] == row['%s_Import' % (country)] else 'Export', axis=1)
        merged_df1['Value'] = merged_df1.apply(lambda row: row[('%s_Value_Import' % (country))] if row['Phi'] == row['%s_Import' % (country)] else row[('%s_Value_Export' % (country))], axis=1)
 
    ultimate_distance = merged_df1[['Product_code', 'RCAcpt', 'Phi', 'Direction', 'Value']]
    ultimate_distance = ultimate_distance.merge(df_H2_space[['Product_code', 'Color', 'Weight']], on=['Product_code'], how='left')
    
    # Include PCIs of the hydrogen products
    index = (pci_df['HS6 ID'].isin(df_H2_space['Product_code']))
    h2_pci_df = pci_df.loc[index][['%d' % (year), 'HS6 ID']]
    h2_pci_df = h2_pci_df.rename(columns={"HS6 ID": "Product_code"})
        
    country_final = pd.merge(ultimate_distance, h2_pci_df, on = 'Product_code', how = 'left')
    country_final = country_final.rename(columns={"%d" % (year): "PCI_%d" % (year)})    
    country_final['Phi'].fillna(0, inplace=True)
    country_final['PCI_%d' % (year)].fillna(0, inplace=True)

    """ Correlation between a products' PCI & distance reveals insights about the nature of the economy """    
    correlation_coefficient, p_value = stats.pearsonr(country_final['Phi'], country_final['PCI_%d' % (year)])
    print('Pearson coefficient r for %s is' % (country), correlation_coefficient)
    
    return country_final
    
country_hci_list1 = np.zeros(len(country_list))

for i, country in enumerate(country_list):
    """ Calculating country list HCI """
    country_final1 = outputs_calc(country, hyperparameter = (1 + 1e-9), plot = 1) # no weighting applied, but has to be > 1.
    hci_value1, sectoral_distance_values1, basket_values1 = hci_calc(country_final1, country, aggregate = 0, plot = 1)
    country_hci_list1[i] = hci_value1[0]
        
#%%

""" Sensitivity Analysis - Hyperparameter changed by 25%"""

country_list = ['Mexico']

for i, country in enumerate(country_list):
    
    country_final1 = outputs_calc(country, hyperparameter = 1.0, plot = 0)
    hci_value1, sectoral_distance_values1, basket_values1 = hci_calc(country_final1, country, aggregate = 1, plot = 0)

    country_final2 = outputs_calc(country, hyperparameter = 1.25, plot = 0)
    hci_value2, sectoral_distance_values2, basket_values2 = hci_calc(country_final2, country, aggregate = 1, plot = 0)
    
    y_boundary = max(abs((hci_value1[1:] - np.mean(hci_value1[1:]))/np.std(hci_value1[1:])))
    x_boundary = max(abs((sectoral_distance_values1 - np.mean(sectoral_distance_values1)) / np.std(sectoral_distance_values1)))
    
    vals_ssci = (hci_value1[1:] - np.mean(hci_value1[1:]))/np.std(hci_value1[1:])
    dist2 = (sectoral_distance_values2 - np.mean(sectoral_distance_values2)) / np.std(sectoral_distance_values2)

    # 
    plt.scatter((sectoral_distance_values2 - np.mean(sectoral_distance_values2)) / np.std(sectoral_distance_values2), vals_ssci, marker = 'o', s = basket_values2, c = ['navy', 'royalblue', 'gold'], label = categories, alpha = 0.5)
    plt.scatter((sectoral_distance_values1 - np.mean(sectoral_distance_values1)) / np.std(sectoral_distance_values1), vals_ssci, marker = 'o', s = basket_values1, c = ['navy', 'royalblue', 'gold'], label = categories)

    plt.xlabel('Distance, normalised', fontdict=font)
    plt.ylabel('SSCI, normalised', fontdict=font)
    plt.title('Opportunity screening, %s' % (country), fontdict=font)
    
    plt.xlim(-x_boundary * 1.4, x_boundary * 1.4)
    plt.xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    plt.yticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    plt.ylim(-y_boundary * 1.5, y_boundary * 1.5)
    plt.axvline(x=0, color='grey', linestyle='-', alpha = 0.5)
    plt.axhline(y=0, color='grey', linestyle='-', alpha = 0.5)
    plt.show()

    delta = np.array(sectoral_distance_values2) - np.array(sectoral_distance_values1)
    relative_delta = delta / sectoral_distance_values1
    print(sectoral_distance_values1,'\n', sectoral_distance_values2, '\n', delta, '\n', relative_delta)
    

#%%
""" Complementary metrices """

''' End-use distance '''

country_list = ['Germany', 'Italy', 'USA', 'Denmark', 'China', 'France', 'Japan', 'United Kingdom', 'Spain', 'Finland', 'Mexico', \
                  'Turkey', 'Rep. of Korea', 'India', 'Thailand', 'Netherlands', 'Norway', 'South Africa', 'Canada', 'Philippines', \
                  'Brazil', 'Indonesia', 'Russian Federation', 'New Zealand', 'United Arab Emirates', 'United Rep. of Tanzania', 'Argentina', 'Oman', 'Australia', \
                  'Kazakhstan', 'Kuwait', 'Iran', 'Chile', 'Saudi Arabia', 'Qatar', 'Algeria', 'Iraq', 'Azerbaijan', 'Malaysia', \
                  'Viet Nam', 'Nigeria', 'Hungary', 'Libya', 'Croatia', 'Angola', 'Tunisia', 'Egypt', 'Austria', 'Czechia', 'Sweden', 'Poland', \
                  'Portugal', 'Ireland', 'Slovenia', 'Sri Lanka', 'Malawi', 'Guatemala', 'Costa Rica', 'Bangladesh', 'Nicaragua', 'Panama', 'Yemen', \
                  'Paraguay', 'Ecuador', 'Jamaica', 'Ghana', 'Estonia', 'Switzerland', 'Romania', 'Slovakia', 'Bulgaria', 'Georgia', 'Pakistan', 'Morocco', 'Lithuania', \
                  'Israel', 'Latvia', 'Lebanon', 'Singapore', 'Ukraine', 'Bosnia Herzegovina', 'Belarus', 'Greece', 'Rep. of Moldova', 'Jordan', 'TFYR of Macedonia', \
                  'Dominican Rep.', 'Honduras', 'Cameroon', 'Albania', 'Colombia', 'Kyrgyzstan', 'Peru', 'Mauritius', 'Zimbabwe', 'Madagascar', 'Uruguay', 'Uzbekistan', \
                  'Mozambique', 'Ethiopia']

demand_product = [721810, 281410, 290511] #steel, ammonia, methanol

def demand_calc(country, rca_dataset, cppt_dataset, calc_type, hyperparameter = 1.03, weighting = False):
    
    ''' Export-based distance calcultions for 4 demand products '''
    
    country_index = int(df_countries[df_countries['country_name_abbreviation'] == country]['country_code'])
    country_export_products = set(rca_dataset[(rca_dataset['Year'] == year) & (rca_dataset[calc_type] == country_index) & (rca_dataset['Mcp'] == 1)].sort_values(by=['RCAcpt'],ascending=False)['Product_code'])
    distance = []
    
    for i, demand_product in tqdm(enumerate(demand_product)):
        index = (cppt_dataset['Product_code_1'] == demand_product)
            
        no_export_products = cppt_dataset.loc[index & ~cppt_dataset['Product_code_2'].isin(country_export_products), 'phi'].astype(float)
        dummy = sum(no_export_products)
        total = cppt_dataset[index]['phi'].sum()
        
        distance.append(dummy / total)
            
    return distance

demand_results = []
for country in tqdm(country_list):
    demand_results += [demand_calc(country, df_exp_rca, df_cppt_exp, 'Exporter')]
        
demand_results = np.array(demand_results)

plt.figure(figsize=(4, 4))
steel = demand_results[:, 0]
ammonia = demand_results[:, 1]
methanol = demand_results[:, 2]

# Combine the datasets into a list
all_data = [steel, ammonia, methanol]
product_types = ['Steel', 'Ammonia', 'Methanol'] 
box_colors = ['darkgrey', 'limegreen', 'lightsteelblue']

# Create the vertical box plot for all four datasets with smaller box sizes and different colors
bp = plt.boxplot(all_data, vert=True, showfliers=False,
                boxprops=dict(color='dimgrey', linewidth=1.25),
                medianprops=dict(color='dimgrey', linewidth=1.25),
                whiskerprops=dict(color='dimgrey', linewidth=1.25),
                capprops=dict(color='dimgrey', linewidth=1.25),
                patch_artist=True, widths=0.3)  # Use patch_artist for box fill color


for box, color in zip(bp['boxes'], box_colors):
    box.set(facecolor=color, linewidth=1.25)  # Set color and linewidth for the box outlines
    
plt.xticks(np.arange(1, len(all_data) + 1), product_types)
#plt.xlabel('Datasets')
plt.ylabel('Product distance')
plt.title('Assessment of local opportunities')
plt.show()


''' Resource Endowment '''

products = [271111, 271121] # LNG & pipeline gas
country_index_list_1 = [int(df_countries[df_countries['country_name_abbreviation'] == country]['country_code']) for country in country_list]
rca_max_ng = [max(df_exp_rca.loc[(df_exp_rca['Exporter'] == country) & (df_exp_rca['Product_code'].isin(products)), 'RCAcpt'].astype(float)) for country in country_index_list_1]

solar_wind_df = pd.read_csv('Analysis.csv')

solar = [solar_wind_df.loc[(solar_wind_df['Country'] == country), 'Solar'].astype(float).reset_index() for country in country_list]
solar2 = [element['Solar'] for element in solar]
solar3 = [s for s in solar2 if not s.empty]
solar4 = [solar3[i][0] for i in range(0, len(solar3))]

wind = [solar_wind_df.loc[(solar_wind_df['Country'] == country), 'Wind'].astype(float).reset_index() for country in country_list]
wind2 = [element['Wind'] for element in wind]
wind3 = [s for s in wind2 if not s.empty]
wind4 = [wind3[i][0] for i in range(0, len(wind3))]

for select_country in country_list:
    plt.figure(figsize=(5, 4))
    
    all_data = [np.array(solar4)]
    product_types = ['Solar'] 
    box_colors = ['Gold']
    
    bp = plt.boxplot(all_data, vert=True, showfliers=0,
                     boxprops=dict(color='dimgrey', linewidth=1.25),
                     medianprops=dict(color='dimgrey', linewidth=1.25),
                     whiskerprops=dict(color='dimgrey', linewidth=1.25),
                     capprops=dict(color='dimgrey', linewidth=1.25),
                     patch_artist=True, widths=0.3) 
    
    for box, color in zip(bp['boxes'], box_colors):
        box.set(facecolor=color, linewidth=1.25) 

    plt.xticks(np.arange(1, len(all_data) + 1), product_types)
    plt.ylabel('Practical potential (kWh/kWp/day)')
    plt.show()

plt.figure(figsize=(6, 4))
all_data = [np.array(wind4)]
product_types = ['Wind']
box_colors = ['Aqua']

bp = plt.boxplot(all_data, vert=True, showfliers=0,
                 boxprops=dict(color='dimgrey', linewidth=1.25),
                 medianprops=dict(color='dimgrey', linewidth=1.25),
                 whiskerprops=dict(color='dimgrey', linewidth=1.25),
                 capprops=dict(color='dimgrey', linewidth=1.25),
                 patch_artist=True, widths=0.3)  

for box, color in zip(bp['boxes'], box_colors):
    box.set(facecolor=color, linewidth=1.25) 
    
plt.xticks(np.arange(1, len(all_data) + 1), product_types)
plt.ylabel('Mean power density (W/$m^{2}$)')
plt.show()

plt.figure(figsize=(5, 4))
all_data = [np.array(rca_max_ng)]
product_types = ['NG/LNG']
box_colors = ['Royalblue']

bp = plt.boxplot(all_data, vert=True, showfliers=0,
                 boxprops=dict(color='dimgrey', linewidth=1.25),
                 medianprops=dict(color='dimgrey', linewidth=1.25),
                 whiskerprops=dict(color='dimgrey', linewidth=1.25),
                 capprops=dict(color='dimgrey', linewidth=1.25),
                 patch_artist=True, widths=0.3)  


for box, color in zip(bp['boxes'], box_colors):
    box.set(facecolor=color, linewidth=1.25)  
    
plt.xticks(np.arange(1, len(all_data) + 1), product_types)
plt.ylabel('Export-based RCA (NG/LNG)')
plt.show()
