#!/usr/bin/env python
# coding: utf-8

# # PYTHON LIBRARY : SEABORN 

# In[1]:


# import libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt  
import seaborn as sns


# In[2]:


# Total no. of rows and Columns
a= pd.read_csv(r"C:\Users\ASUS\Downloads\Delinquency_prediction_dataset_1.csv")
a


# In[3]:


a.head()


# In[4]:


print(len(a))


# # LINE PLOT 📈

# In[5]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
a = pd.read_csv(r"C:\Users\ASUS\Downloads\Delinquency_prediction_dataset_1.csv")  

# Melt month columns to long format
month_columns = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
a_melted = a.melt(id_vars=['Customer_ID'], value_vars=month_columns,
                    var_name='Month', value_name='Payment_Status')

# Create an ordered category for months to ensure proper line order
month_order = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
a_melted['Month'] = pd.Categorical(a_melted['Month'], categories=month_order, ordered=True)

# Count number of payment statuses per month
status_counts = a_melted.groupby(['Month', 'Payment_Status']).size().reset_index(name='Count')

# Plotting
plt.figure(figsize=(12, 6))
sns.lineplot(data=status_counts, x='Month', y='Count', hue='Payment_Status', marker='o',palette='prism_r')
plt.title('Monthly Payment Status Trends')
plt.ylabel('Number of Customers')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # HISTOGRAM 📊

# In[6]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create the dataset
data = {
    "Customer_ID": [
        "CUST0001", "CUST0002", "CUST0003", "CUST0004", "CUST0005", "CUST0006", "CUST0007", "CUST0008",
        "CUST0009", "CUST0010", "CUST0011", "CUST0012", "CUST0013", "CUST0014", "CUST0015", "CUST0016",
        "CUST0017", "CUST0018", "CUST0019", "CUST0020", "CUST0021", "CUST0022", "CUST0023"
    ],
    "Credit_Score": [
        398, 493, 500, 413, 487, 700, 354, 415, 405, 679, 340, 679, 320, 528, 617, 601,
        543, 850, 831, 200, 783, 690, 321
    ]
}

df = pd.DataFrame(data)

# Plot histogram
plt.figure(figsize=(5, 6))
sns.displot(df['Credit_Score'], bins=[200,300,400,500,600,700,800,900], kde=True,rug=True,color="purple")
plt.title('Distribution of Customer Credit Scores ')
plt.xlabel('Credit Score')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()


# # COUNT PLOT

# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create the dataset
data_bar = {
    "Customer_ID": [
        "CUST0001", "CUST0002", "CUST0003", "CUST0004", "CUST0005", "CUST0006", "CUST0007", "CUST0008",
        "CUST0009", "CUST0010", "CUST0011", "CUST0012", "CUST0013", "CUST0014", "CUST0015", "CUST0016",
        "CUST0017", "CUST0018", "CUST0019", "CUST0020", "CUST0021", "CUST0022", "CUST0023"
    ],
    "Employment_Status": [
        "EMP", "Self-employed", "Self-employed", "Unemployed", "Self-employed", "Unemployed", "employed", "EMP",
        "Employed", "EMP", "Unemployed", "Unemployed", "EMP", "retired", "Unemployed", "Self-employed",
        "Self-employed", "Unemployed", "Unemployed", "retired", "Self-employed", "Unemployed", "Unemployed"
    ]
}

df_bar = pd.DataFrame(data_bar)

# Create bar plot
plt.figure(figsize=(12, 6))
sns.countplot(data=df_bar, x='Employment_Status', 
              order=df_bar['Employment_Status'].value_counts().index, 
              palette='Set2')
plt.title('Customer Count by Employment Status')
plt.xlabel('Employment Status')
plt.ylabel('Number of Customers')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# In[8]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create the dataset
data_bar = {
    "Customer_ID": [
        "CUST0001", "CUST0002", "CUST0003", "CUST0004", "CUST0005", "CUST0006", "CUST0007", "CUST0008",
        "CUST0009", "CUST0010", "CUST0011", "CUST0012", "CUST0013", "CUST0014", "CUST0015", "CUST0016",
        "CUST0017", "CUST0018", "CUST0019", "CUST0020", "CUST0021", "CUST0022", "CUST0023"
    ],
    "Employment_Status": [
        "EMP", "Self-employed", "Self-employed", "Unemployed", "Self-employed", "Unemployed", "employed", "EMP",
        "Employed", "EMP", "Unemployed", "Unemployed", "EMP", "retired", "Unemployed", "Self-employed",
        "Self-employed", "Unemployed", "Unemployed", "retired", "Self-employed", "Unemployed", "Unemployed"
    ]
}

df_bar = pd.DataFrame(a)

# Create bar plot
plt.figure(figsize=(12, 6))
sns.countplot(data=df_bar, x='Employment_Status', 
              order=df_bar['Employment_Status'].value_counts().index, 
              palette='plasma',hue="Credit_Card_Type")
plt.title('Customer Count by Employment Status')
plt.xlabel('Employment Status')
plt.ylabel('Number of Customers')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# # SCATTER PLOT

# In[9]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(a)

plt.figure(figsize=(15, 6))
sns.scatterplot(x="Income",y="Credit_Score",data=a,hue="Missed_Payments",style="Missed_Payments",size="Missed_Payments")
plt.show()


# # HEATMAP 
# 

# In[10]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
a = pd.read_csv(r"C:\Users\ASUS\Downloads\Delinquency_prediction_dataset_1.csv")  

# Select only the first 10 rows and the month columns
month_cols = ['Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
status_a = a.loc[:9, month_cols]  # First 10 rows (index 0 to 9)

# Map the status values to numeric codes
status_map = {'On-time': 0, 'Late': 1, 'Missed': 2}
status_encoded = status_a.replace(status_map)

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(status_encoded, annot=True,cmap="rocket",cbar_kws={'label': 'Payment Status'})
plt.title('Payment Status Heatmap for First 10 Customers')
plt.xlabel('Month')
plt.ylabel('Customer Index (First 10)')
plt.tight_layout()
plt.show()


# # BAR PLOT

# In[11]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.DataFrame(a)

# Create a barplot of average income by employment status
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Employment_Status', y='Income', estimator=np.mean, ci=20, palette='viridis')
plt.title('Average Income by Employment Status')
plt.xlabel('Employment Status')
plt.ylabel('Average Income')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # VIOLIN PLOT

# In[12]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(a)

plt.figure(figsize=(11, 6))
sns.violinplot(data=df, x='Credit_Card_Type', y='Credit_Score', palette='plasma_r',linewidth=1.5,inner="quartile")
plt.title('Credit Score Distribution by Credit Card Type')
plt.xlabel('Credit Card Type')
plt.ylabel('Credit Score')
plt.tight_layout()
plt.show()


# # PAIR PLOT

# In[13]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(a)

# Select numerical features to include
features = ['Income', 'Credit_Score', 'Credit_Utilization', 'Loan_Balance', 'Debt_to_Income_Ratio']

# Create the pair plot colored by Employment Status
sns.pairplot(df[features + ['Employment_Status']], hue='Employment_Status', palette='nipy_spectral')
plt.suptitle("Pair Plot of Financial Features Grouped by Employment Status", y=1.02)
plt.show()


# # STRIP PLOT

# In[14]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(a)

plt.figure(figsize=(10, 6))
sns.stripplot(data=df, x='Employment_Status', y='Credit_Score', hue='Delinquent_Account', jitter=True, palette='rocket', dodge=True)
plt.title('Credit Score by Employment Status (Colored by Delinquency)')
plt.xlabel('Employment Status')
plt.ylabel('Credit Score')
plt.legend(title='Delinquent')
plt.tight_layout()
plt.show()


# # BOX PLOT

# In[15]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(a)

# Box plot with hue and color variation
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Credit_Card_Type', y='Income', hue='Delinquent_Account', palette='Set2')
plt.title('Income Distribution by Credit Card Type and Delinquency')
plt.xlabel('Credit Card Type')
plt.ylabel('Income')
plt.legend(title='Delinquent')
plt.tight_layout()
plt.show()


# In[16]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(a)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Credit_Card_Type', y='Credit_Score', notch=True, palette='coolwarm',
            showmeans=True,meanprops={"markeredgecolor":"blue"})
plt.title('Notched Box Plot of Credit Score by Credit Card Type')
plt.xlabel('Credit Card Type')
plt.ylabel('Credit Score')
plt.tight_layout()
plt.show()


# In[17]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(a)

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Location', y='Income', palette='husl')
plt.title('Income Distribution by Location')
plt.xlabel('City')
plt.ylabel('Income')
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()


# In[18]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(a)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Credit Score by Segment
sns.boxplot(data=df, x='Employment_Status', y='Credit_Score', palette='Set2', ax=axes[0])
axes[0].set_title('Credit Score by Segment')

# Loan Balance by Employment Status
sns.boxplot(data=df, x='Employment_Status', y='Loan_Balance', palette='Set1', ax=axes[1])
axes[1].set_title('Loan Balance by Employment Status')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# # CAT PLOT

# In[19]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(a)

sns.catplot(data=df, kind="point", x="Location", y="Debt_to_Income_Ratio", hue="Delinquent_Account",
            col="Credit_Card_Type", jitter=True, palette="husl", height=4, aspect=0.9)
plt.suptitle("Debt-to-Income Ratio by Segment and Credit Card Type", y=1.05)
plt.figure(figsize=(12, 6))
plt.tight_layout()
plt.show()


# # MULTIPLE PLOT (Facet - Grip)

# In[20]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(a)

g = sns.FacetGrid(df, row="Location", col="Credit_Card_Type", height=3.5, aspect=1.2)
g.map(sns.kdeplot, "Loan_Balance", fill=True, color="coral")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Loan Balance Distribution by Location and Credit Card Type")
plt.show()


# In[21]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(a)

g = sns.FacetGrid(df, col="Employment_Status", col_wrap=3, height=4, aspect=1.2)
g.map(sns.scatterplot, "Income", "Credit_Score", color="teal", alpha=0.6)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Income vs Credit Score by Employment Status")
plt.show()


# In[22]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(a)

g = sns.FacetGrid(df, col="Delinquent_Account", row="Location", height=4, aspect=1.2)
g.map(sns.histplot, "Credit_Score", kde=True, color="skyblue")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Credit Score Distribution by Delinquency and Segment")
plt.show()


# In[ ]:




