#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import bs4 as bs


# In[2]:


import requests
#Analyzing the data and keeping whats necessary for us to prdict prices


# In[3]:


df1 = pd.read_csv("blr_real_estate_prices.csv")


# In[4]:


df1


# In[5]:


df1.shape


# 

# In[6]:


df1.head()


# In[7]:


df2= df1.drop(["area_type","availability","society","balcony"],axis="columns")


# In[8]:


df2


# In[9]:


df2.head()


# In[10]:


df2.isnull()


# In[11]:


df2.isnull().sum()


# In[12]:


df3=df2.dropna()
df3.isnull().sum()
df3.head()


# 

# In[13]:


df3["size"].unique()


# In[14]:


df3["BHK"]=df3["size"].apply(lambda x: int(x.split(' ')[0]))


# In[15]:


df3


# In[16]:


df3.drop(["size"],axis="columns")


# 

# In[17]:


df3["BHK"].unique()


# In[18]:


df3[df3.BHK>20]
#DATA CLEANING


# In[19]:


df3["total_sqft"].unique()


# In[20]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[21]:


df3[~df3["total_sqft"].apply(is_float)].head(10)


# In[22]:


def sqft_to_num(x):
    tokens= x.split('-')
    if len(tokens)==2:
      return (float(tokens[0])+float(tokens[1]))/2
    try:
         return float(x)
    except:
         return None 


# 

# In[ ]:





# In[23]:


sqft_to_num('131-142')


# 

# In[24]:


df4=df3.copy()


# In[25]:


df4 ["total_sqft"]=df4["total_sqft"].apply(sqft_to_num)


# In[26]:


df4.head()


# In[27]:


df4["total_sqft"].unique()


# In[28]:


df4["Price _per_sqft"]=df4["price"]*100/df4["total_sqft"]


# In[29]:


df4


# In[30]:


df5=df4.copy()


# In[31]:


df5["location"].unique()


# In[32]:


len(df5["location"].unique())


# In[33]:


location_stats=df5.groupby("location")["location"].agg("count").sort_values(ascending=False)


# In[34]:


location_stats


# In[35]:


len(location_stats[location_stats<10])
    


# In[36]:


location_less_than_10=location_stats[location_stats<=10]


# In[37]:


def compact(x):
    if x in location_less_than_10:
        x="other"
    return x
    


# In[38]:


location_less_than_10


# In[39]:


def compact(x):
    if x in location_less_than_10:
        x="other"
    return x    


# 

# In[40]:


compact("Kanakapura  Rod")


# In[41]:


df6=df5.copy()


# In[42]:


df6["location"]=df6["location"].apply(compact)


# In[43]:


df6.head(20)


# In[44]:


#OUTLIER REMOVAL
df7=df6.drop(["size"],axis="columns")


# In[45]:


df7


# In[46]:


df7[df7["total_sqft"]/df7["BHK"]<300].head()


# In[47]:


df8=df7[df7["total_sqft"]/df7["BHK"]>300]


# In[48]:


len(df8)


# In[49]:


df8["Price _per_sqft"].describe()


# In[50]:


#remove outliers, prices per location should lie bertween m+st and m-st
def remove_pps_outlier(sub_df):
    m=np.mean(sub_df["Price _per_sqft"])
    st=np.std(sub_df["Price _per_sqft"])
    new_df=sub_df[(sub_df["Price _per_sqft"] >= (m-st)) & (sub_df["Price _per_sqft"] <= (m+st)) ]
    return new_df
    
    


# In[51]:


df9=pd.DataFrame()
count=0
for x,y in df8.groupby("location"):
    #temp=remove_pps_outlier(y)
    #f9.append(temp)
    temp=remove_pps_outlier(y)
    df9=pd.concat([df9,temp], ignore_index=True)

  
    
    
    
   
    
    

    
    
    


# In[52]:


df9


# In[53]:


def plot(df,Location):
    bhk2=df[(df["location"]==Location) & (df["BHK"]==2)]
    bhk3=df[(df["location"]==Location) & (df["BHK"]==3)]
    plt.scatter(bhk2.total_sqft,bhk2["Price _per_sqft"], label="2 BHK")
    plt.scatter(bhk3.total_sqft,bhk3["Price _per_sqft"], label="3 bhk")
    plt.legend()
        


# In[54]:


plot(df9,"7th Phase JP Nagar")


# In[55]:


plot(df9,"Rajaji Nagar")


# In[57]:


df10=df7.head(70)
        


# In[58]:


df10


# In[59]:


blr_data = df10.to_csv('blr.csv', index = True)
print('\nCSV String:\n', blr_data)


# In[65]:


import matplotlib.pyplot as plt
import numpy as np

# Generate x values from -10 to 10
y = np.linspace(1, 6, 100)
# Calculate y values using the equation y = 0.97x + 2.87
x = 2.30*y - 2.98

# Plot the line
plt.scatter(x, y, label='x = 2.30y - 2.98')

# Add labels and a legend
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Display the plot
plt.show()


# In[ ]:




