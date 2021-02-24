# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 09:33:35 2020

@author: Andrew
"""

import pandas as pd
import numpy as np
import datetime

import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome()

cols = ["State","Legislature","Date","Status","Bill_Number","Bill_Name","Bill_Summary","Bill_Text","Bill_Topics","Bill_Author","Bill_URL"]

data = pd.DataFrame(columns=cols, index=np.arange(1000))

import time

print('searching for bills')

myurl = "https://www.ncsl.org/research/environment-and-natural-resources/environment-and-natural-resources-state-bill-tracking-database.aspx"
driver.get(myurl)

soup = BeautifulSoup(
    driver.page_source, "html.parser"
)

bills_container = soup.find('div', {'id':'dnn_ctr85765_StateNetDB_linkList'})
bills = bills_container.prettify().split("<hr/>")
df_idx = 0
for bill in bills:
    bill_soup = BeautifulSoup(
    bill, "html.parser"
    )
    if len(bill_soup.find_all('a')) > 2:  #contains header, so there is an extra 'a' tag
        first_ahref= bill_soup.find_all('a')[1]
    else:
        first_ahref = bill_soup.find_all('a')[0]
    data["State"][df_idx] = first_ahref.text.strip().split(' ')[0]
    data["Legislature"][df_idx] = first_ahref.text.strip().split(' ')[1]
    data["Bill_Number"][df_idx] = first_ahref.text.strip().split(' ')[2] 
    data["Bill_URL"][df_idx] = first_ahref["href"]    
    #below not dependent on if it is header or not 
    
    #everything else is in <b></b> tags
    b_tags = bill.split("<br/>")
    b_tags_soup = []
    for b_tag in b_tags:
        temp_b_tag_soup = BeautifulSoup(
                            b_tag, "html.parser"
                            )
        b_tags_soup.append(temp_b_tag_soup) #messy way to re-soup an array 
    
    data["Bill_Name"][df_idx] = b_tags_soup[2].text.strip()
    data["Status"][df_idx] = b_tags_soup[3].text.split(':')[1].strip()
    data["Date"][df_idx] = pd.to_datetime(b_tags_soup[4].text.split('*')[1].strip().split('-')[0]) #Getting just date, removing "enacted"
    data["Bill_Author"][df_idx] = b_tags_soup[5].text.split(':')[1].strip() #could create a party column later
    data["Bill_Topics"][df_idx] = b_tags_soup[6].text.split(':')[1].strip()
    data["Bill_Summary"][df_idx] = b_tags_soup[7].text.split(':')[1].strip()  
    
    df_idx+=1
data.to_csv(r'C:\Users\Andrew\Documents\PythonScripts\NLP\webscraping\flood_research\bills_flood_full.csv')
data = pd.read_csv(r'C:\Users\Andrew\Documents\PythonScripts\NLP\webscraping\flood_research\bills_flood_full.csv',index_col=0)

#need to clean through and remove repeat bills (i.e. keep only the most recent bill or first occurance of a bill)
print('cleaning up duplicate bills')
data.dropna(how="all",inplace=True)

print('go through links to get bill text')
i_idx = 0
for url in data["Bill_URL"][i_idx:]:
    print(i_idx)
    req = requests.get(url)

    soup = BeautifulSoup(
        req.text, "html.parser"
    )
    try:
        data["Bill_Text"][i_idx] = soup.find('div',{'class':'documentBody'}).text
    except:
        print("didn't work, skipping")
    i_idx+=1
    
data.to_csv(r'C:\Users\Andrew\Documents\PythonScripts\NLP\webscraping\flood_research\bills_flood_pruned.csv')
    