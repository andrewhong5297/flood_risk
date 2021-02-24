# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:10:25 2020

@author: Andrew
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from urllib.request import urlopen
import json

###dictionary
regions = pd.read_excel(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\States_Abrev_Regions.xlsx', engine='openpyxl')
abbrev_to_state_dict = dict(zip(regions["State Code"],regions["State"]))
state_region_dict = dict(zip(regions["State"],regions["Region"]))

def abbrev_to_state(x):
    try:
        state_name = abbrev_to_state_dict[x]
        return state_name
    except:
        print('{} does not exist'.format(x))
        return "skip"
    
###historical claims, need geographic representation of the data
nfip = pd.read_csv(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\FimaNfipClaims.csv', dtype={'countyCode': object})

counties_risk = nfip.pivot_table(index='countyCode',
                                 values='amountPaidOnContentsClaim',
                                 aggfunc="sum")

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
    
def plot_unemploy_choro_county(df,rangesize,datatype):
    fig_map = px.choropleth_mapbox(df, geojson=counties, locations=df.index, color=datatype,
                                color_continuous_scale="Reds",
                                range_color=rangesize,
                                mapbox_style="carto-positron",
                                zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                                opacity=0.5,
                              )
    fig_map.update_layout(
        title_text = '{} by County'.format(datatype),
        margin={"r":10,"t":0,"l":0,"b":0},
        )
    return fig_map

fig = plot_unemploy_choro_county(counties_risk,[0,1000000],'amountPaidOnContentsClaim')
# plot(fig,filename="counties_flood_NFIP.html")

###plot by state
state_risk = nfip.pivot_table(index='state',
                            values='amountPaidOnContentsClaim',
                            aggfunc="sum")

def plot_unemploy_choro_state(df):
    fig = go.Figure(data=go.Choropleth(
        locations=df.index, # Spatial coordinates
        z = df['amountPaidOnContentsClaim'], # Data to be color-coded
        locationmode = 'USA-states', # set of locations match entries in `locations`
        colorscale = 'Reds',
        zmin = 0,
        zmax = 100000000,
        colorbar_title = 'Amount Paid on Contents Claim',
    ))
    
    fig.update_layout(
        title_text = 'Amount Paid on Contents Claim by State',
        geo_scope='usa', # limite map scope to USA
    )
    return fig

fig = plot_unemploy_choro_state(state_risk)
# plot(fig,filename="state_flood_NFIP.html")

'''final comparison'''
###sea level and precipitation  (by region in NOA)
import seaborn as sns
water_risks = pd.read_excel(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\States_Sea_Rain_FSF.xlsx', engine='openpyxl')

#take NFIP pivot and attach to water_risks 
state_risk.reset_index(inplace=True)
state_risk["state_name"] = state_risk["state"].apply(lambda x: abbrev_to_state(x))
state_risk = state_risk[state_risk["state_name"]!="skip"]

water_risks = water_risks.set_index('State').join(state_risk.set_index('state_name'))
water_risks.reset_index(inplace=True)
sns.set_theme(style="whitegrid")

### Make the PairGrid
g = sns.PairGrid(water_risks.sort_values("PercentPropertiesAtRisk2050FSF", ascending=False),
                  x_vars=["PercentPropertiesAtRisk2050FSF",
                        "PrecipitationPercentInc1958",
                        'PrecipitationPercentIncrease2100',
                        "SeaLevelRiseInches1950",
                        "amountPaidOnContentsClaim"], 
                  y_vars=["State"],
                  height=10, aspect=.25)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h",
      palette="flare_r", linewidth=1, edgecolor="w")

# Use semantically meaningful titles for the columns
titles = ["FSF Properties at Risk 2050","Rain Increase Since 1958","Rain Increase 2100 (RCP8.5)","Sea Level Rise Since 1950","Amount Paid on FEMA Claims"]

xlabels = {1:"Percentage",2:"Percentage",3:"Percentage",4:"Inches",5:"Dollars"}

idx=1
for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)
    ax.set(xlabel=xlabels[idx])
    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    idx+=1

sns.despine(left=True, bottom=True)

'''calculations'''
#need population data to standardize this? or just sklearn normalize? 
#standard scaled NFIP * sea level (if applicable) * ((precipiation increase+forecast)/2)?

#use a regression!
import numpy as np
import statsmodels.api as sm
water_risks.fillna(0, inplace=True)
mod = sm.OLS(exog=water_risks[["PercentPropertiesAtRisk2050FSF",
                        "PrecipitationPercentInc1958",
                        "SeaLevelRiseInches1950"]], 
             endog=water_risks["amountPaidOnContentsClaim"])
             
res = mod.fit()
print(res.summary())

'''scaling data'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(water_risks[["PrecipitationPercentIncrease2100",
                        "PrecipitationPercentInc1958",
                        "SeaLevelRiseInches1950",
                        "amountPaidOnContentsClaim"]]))

standard_data = scaler.transform(water_risks[["PrecipitationPercentIncrease2100",
                        "PrecipitationPercentInc1958",
                        "SeaLevelRiseInches1950",
                        "amountPaidOnContentsClaim"]])

standardized_risks = pd.DataFrame(index=water_risks["State"],
                                  columns=["PrecipitationPercentIncrease2100", 
                                            "PrecipitationPercentInc1958",
                                            "SeaLevelRiseInches1950",
                                            "amountPaidOnContentsClaim"],
                                  data=standard_data)

standardized_risks.reset_index(inplace=True)
standardized_risks["regions"] = standardized_risks["State"].apply(lambda x:state_region_dict[x])
# sns.pairplot(standardized_risks, kind="scatter", hue="regions")

standardized_risks["cumulative_score"] = standardized_risks.sum(axis=1)
standardized_risks = standardized_risks.sort_values(by="cumulative_score",ascending=False)

import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(8,15))
ax1 = sns.barplot(x = standardized_risks["cumulative_score"], y = standardized_risks["State"])
standardized_risks.to_csv(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\risk_severity_states.csv')
