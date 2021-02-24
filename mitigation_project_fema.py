# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 23:09:25 2020

@author: Andrew
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\HazardMitigationAssistanceProjects.csv')
df["projectAmountMillions"]=df["projectAmount"].div(1000000)

regions = pd.read_excel(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\States_Abrev_Regions.xlsx', engine='openpyxl')
state_region_dict = dict(zip(regions["State"],regions["Region"]))
def state_to_region(x):
    try:
        region_result = state_region_dict[x]
        return region_result
    except:
        print("result not found for {}, skipping".format(x))
    return "skip"

#all state are in the FEMA program.
state_fema = df["state"].unique()
state_all = regions["State"]
state_exists = []
for idx,state in enumerate(state_all):
    if state in state_fema:
        state_exists.append(state)
    else:
        print("{} not in FEMA program".format(state))

#loop through all states, can put this in plotly dash later
# busyStates = df["state"].value_counts().index
# for state in busyStates:
#     temp_df = df[df["state"]==state]
#     projectTypeBar = temp_df["projectType"].value_counts()[:15]
#     fig, ax = plt.subplots(figsize=(10,10))
#     ax.barh(y=projectTypeBar.index,width = projectTypeBar)
#     ax.set(title="Most Common Mitigation Project Types FEMA in {}".format(state))
    
###Date pivot
df["dateInitiallyApproved"] = pd.to_datetime(df['dateInitiallyApproved'])
pivot_date = df.pivot_table(index="dateInitiallyApproved",values=["projectAmountMillions", "numberOfProperties"],
                       aggfunc="sum")

#growth is fairly linear
pivot_date.cumsum().plot(kind="line",title="Timeline of Project Funding and Properties Covered",figsize=(10,10),subplots=True)

###State pivot
pivot_state = df.pivot_table(index="state",values=["projectAmountMillions", "numberOfProperties"],
                       aggfunc="sum")
pivot_state.reset_index(inplace=True)
pivot_state["region"]=pivot_state["state"].apply(lambda x: state_to_region(x))
pivot_state = pivot_state[pivot_state["region"]!="skip"] #removing regions not included in 50 states

#pairplot for states and amount funded/properties covered
ax = sns.pairplot(pivot_state.iloc[:,1:],kind="scatter",hue="region")
ax.fig.suptitle("Total Project Amount and Number of Properties Covered In FEMA Projects by US State")
ax.fig.tight_layout()
ax.fig.subplots_adjust(top=0.95) # Reduce plot to make room 
plt.show()

###plotly express scatter, for dash app later
import plotly.express as px
from plotly.offline import plot
available_states = pivot_state['state'].unique()

fig_px = px.scatter(pivot_state, x="numberOfProperties", y="projectAmountMillions", hover_data=['state','region'],color="region")
# plot(fig_px,filename='fema_state_propnumber_amountspent_scatter.html')