# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:38:20 2020

@author: Andrew
"""
import pandas as pd
import numpy as np
import datetime
import re
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
'''prep dictionaries'''
regions = pd.read_excel(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\States_Abrev_Regions.xlsx', engine='openpyxl')
state_abbrev_dict = dict(zip(regions["State"],regions["State Code"]))
abbrev_to_state_dict = dict(zip(regions["State Code"],regions["State"]))
state_region_dict = dict(zip(regions["State"],regions["Region"]))

def state_to_abbrev(x):
    try:
        abbrev_result = state_abbrev_dict[x]
        return abbrev_result
    except:
        print("result not found for {}, skipping".format(x))
    return "skip"

'''prep bill data to get counts'''
bills_df = pd.read_csv(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\bills_flood_pruned.csv',index_col=0)

print('cleaning up bills')
bills_df = bills_df[bills_df['Bill_Text'].notna()]
bills_df["Summary"]=bills_df["Bill_Text"] #setting for rest of nlp
bills_df.drop(columns="Bill_Text",inplace=True)

#no duplicate bills
bills_df["Agg_Name"] = bills_df["State"] + bills_df["Legislature"] + bills_df["Bill_Number"].astype(str)
bills_df = bills_df.drop_duplicates(subset=['Agg_Name'])
bills_df["Date"] = pd.to_datetime(bills_df["Date"])
bills_df["Year"] = bills_df["Date"].apply(lambda x: x.year)

bill_state_counts = bills_df["State"].value_counts()
bill_state_counts = bill_state_counts.reset_index()
bill_state_counts.columns = ["state","bill_count"]

'''prep FEMA project data to get spending and property coverage'''
FEMA = pd.read_csv(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\HazardMitigationAssistanceProjects.csv')
FEMA["projectAmountMillions"]=FEMA["projectAmount"].div(1000000)

regions = pd.read_excel(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\States_Abrev_Regions.xlsx', engine='openpyxl')
state_region_dict = dict(zip(regions["State"],regions["Region"]))
def state_to_region(x):
    try:
        region_result = state_region_dict[x]
        return region_result
    except:
        print("result not found for {}, skipping".format(x))
    return "skip"

pivot_state = FEMA.pivot_table(index="state",values=["projectAmountMillions", "numberOfProperties"],
                       aggfunc="sum")

pivot_state_count = FEMA.pivot_table(index="state",values=["projectAmountMillions"],
                       aggfunc="count")

pivot_state_count.columns=["projectCount"]
pivot_state = pd.concat([pivot_state,pivot_state_count],axis=1)
pivot_state.reset_index(inplace=True)
pivot_state["state"] = pivot_state["state"].apply(lambda x: state_to_abbrev(x))
pivot_state = pivot_state[pivot_state['state']!="skip"]

'''putting score together'''
pivot_state = pivot_state.set_index("state").join(bill_state_counts.set_index("state"))
pivot_state.reset_index(inplace=True)

sns.set_theme(style="whitegrid")

### Make the PairGrid
g = sns.PairGrid(pivot_state.sort_values("projectAmountMillions", ascending=False),
                  x_vars=["projectAmountMillions",
                        "numberOfProperties",
                        'projectCount',
                        "bill_count"], 
                  y_vars=["state"],
                  height=10, aspect=.25)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h",
      palette="flare_r", linewidth=1, edgecolor="w")

# Use semantically meaningful titles for the columns
titles = ["Sum $mm Spent Projects","Properties Covered by Project","Count of Flood Projects","Count of Flood Bills"]

xlabels = {1:"Dollars",2:"Count",3:"Count",4:"Count"}

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

'''scaling data'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(pivot_state[["projectAmountMillions",
                        "numberOfProperties",
                        'projectCount',
                        "bill_count"]]))

standard_data = scaler.transform(pivot_state[["projectAmountMillions",
                        "numberOfProperties",
                        'projectCount',
                        "bill_count"]])

standardized_risks = pd.DataFrame(index=pivot_state["state"],
                                  columns=["projectAmountMillions",
                                    "numberOfProperties",
                                    'projectCount',
                                    "bill_count"],
                                  data=standard_data)

standardized_risks.reset_index(inplace=True)
standardized_risks["regions"] = standardized_risks["state"].apply(lambda x:state_region_dict[abbrev_to_state_dict[x]])
# sns.pairplot(standardized_risks, kind="scatter", hue="regions")

standardized_risks["cumulative_score"] = standardized_risks.sum(axis=1)
standardized_risks = standardized_risks.sort_values(by="cumulative_score",ascending=False)

import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(nrows=1, ncols=1,figsize=(8,15))
ax1 = sns.barplot(x = standardized_risks["cumulative_score"], y = standardized_risks["state"])
standardized_risks.to_csv(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\response_strength_states.csv')
