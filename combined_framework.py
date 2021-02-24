# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 18:40:07 2020

@author: Andre
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('seaborn')

risk_sev = pd.read_csv(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\risk_severity_states.csv', index_col=0)
response_str = pd.read_csv(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\response_strength_states.csv', index_col=0)

regions = pd.read_excel(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\States_Abrev_Regions.xlsx', engine='openpyxl')
state_abbrev_dict = dict(zip(regions["State"],regions["State Code"]))
state_region_dict = dict(zip(regions["State Code"],regions["Region"]))

risk_sev["State Code"] = risk_sev["State"].apply(lambda x: state_abbrev_dict[x])

plot_df = risk_sev.set_index("State Code").join(response_str.set_index("state"))
plot_df.reset_index(inplace=True)
plot_df["Region"] = plot_df["State Code"].apply(lambda x: state_region_dict[x])
plot_df.to_csv(r'C:\Users\Andre\OneDrive\Documents\PythonScripts\Flood_Research\flood_dash_app\framework_plot.csv')
'''plot'''
x = np.linspace(-10, 10, 100)
y = ((x+10)**2)/20-4

fig, ax = plt.subplots(figsize=(8,6))
ax.set(xlim=[-10,8],ylim=[-4,12],title="Flood Framework US States", xlabel="Risk Severity",ylabel="Response Strength")
ax.set_yticklabels([])
ax.set_xticklabels([])

ax.scatter(plot_df["cumulative_risk_score"], plot_df["cumulative_response_score"])

for i, txt in enumerate(plot_df["State Code"]):
    ax.annotate(txt, (plot_df["cumulative_risk_score"][i]+0.15, plot_df["cumulative_response_score"][i]-0.15))

import plotly.express as px
from plotly.offline import plot
fig = px.scatter(plot_df,x="cumulative_risk_score",y="cumulative_response_score",
           color="Region",hover_data=["State Code"],text="State Code")
fig.update_traces(textposition='top center')
plot(fig, filename="floodframework.html")
    
'''plot FSF'''
water_risks = pd.read_excel(r'C:\Users\Andre\OneDrive - nyu.edu\Documents\Python Script Backup\flood files\States_Sea_Rain_FSF.xlsx', engine='openpyxl')
plot_df = plot_df.set_index("State Code").join(water_risks[["State Code","PercentPropertiesAtRisk2050FSF"]].set_index("State Code"))
plot_df.reset_index(inplace=True)

x = np.linspace(0, 1.4, 100)
y = 100*((x)**2)-4

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(plot_df["PercentPropertiesAtRisk2050FSF"], plot_df["cumulative_response_score"])
ax.set(xlim=[0,0.4],ylim=[-4,12],title="Flood Framework US States", xlabel="FSF Risk Severity",ylabel="Response Strength")
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.plot(x, y, ls="--", c=".3")

for i, txt in enumerate(plot_df["State Code"]):
    ax.annotate(txt, (plot_df["PercentPropertiesAtRisk2050FSF"][i], plot_df["cumulative_response_score"][i]))

