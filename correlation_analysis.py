# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# Importing the dataset
df = pd.read_csv('covid prediction.csv')

# Caculate Preason correlation cofficient
# Convert dataframe into series 

list1 = df['covid'] 
list2 = df['new_cases'] 
  
# Apply the pearsonr() 
corr, _ = pearsonr(list1, list2) 
print('Pearsons correlation: %.3f' % corr) 

list3 = df['COVID-19'] 
list4 = df['new_cases'] 
  
# Apply the pearsonr() 
corr, _ = pearsonr(list3, list4) 
print('Pearsons correlation: %.3f' % corr) 


list4 = df['coronavirus'] 
list5 = df['new_cases'] 
  
# Apply the pearsonr() 
corr, _ = pearsonr(list4, list5) 
print('Pearsons correlation: %.3f' % corr)


list6 = df['coronavirus in bangla'] 
list7 = df['new_cases'] 
  
# Apply the pearsonr() 
corr, _ = pearsonr(list6, list7) 
print('Pearsons correlation: %.3f' % corr)

list8 = df['pneumonia'] 
list9 = df['new_cases'] 
  
# Apply the pearsonr() 
corr, _ = pearsonr(list8, list9) 
print('Pearsons correlation: %.3f' % corr)

list10 = df['Coronavirus vaccine'] 
list11 = df['new_cases'] 
  
# Apply the pearsonr() 
corr, _ = pearsonr(list10, list11) 
print('Pearsons correlation: %.3f' % corr)

list12 = df['high temperature'] 
list13 = df['new_cases'] 
  
# Apply the pearsonr() 
corr, _ = pearsonr(list12, list13) 
print('Pearsons correlation: %.3f' % corr)

list14 = df['cough'] 
list15 = df['new_cases'] 
  
# Apply the pearsonr() 
corr, _ = pearsonr(list14, list15) 
print('Pearsons correlation: %.3f' % corr)

list16 = df['coronavirus test'] 
list17 = df['new_cases'] 
  
# Apply the pearsonr() 
corr, _ = pearsonr(list16, list17) 
print('Pearsons correlation: %.3f' % corr)

list18 = df['Musk'] 
list19 = df['new_cases'] 
  
# Apply the pearsonr() 
corr, _ = pearsonr(list18, list19) 
print('Pearsons correlation: %.3f' % corr)
#correlation coefficients

coefficient = {'Keyword':['covid','COVID-19','coronavirus','coronavirus in bangla','pneumonia','coronavirus vaccine','high temperature','cough','coronavirus test','musk'],
               'R' : [0.579, 0.942, 0.320, 0.822, -0.261, 0.498, -0.127, -0.491, -0.250, -0.148]}
df2 = pd.DataFrame(coefficient)
print(df2)                                          
                                           
                                           
#Caculate lag Cross-correlation cofficient 
                                     
dfx = df[(df['Day'] > '09-03-20')]
dfx.head(5)

fields = ['Day','covid','new_cases'] 
covid = dfx[fields]
covid.head(60)

def df_derived_by_shift(df,lag=0,NON_DER=[]):
    df = df.copy()
    if not lag:
        return df
    cols ={}
    for i in range(1,lag+1):
        for x in list(df.columns):
            if x not in NON_DER:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k,v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)    
        i = 1
        for c in columns:
            dfn[c] = df[k].shift(periods=i)
            i+=1
        df = pd.concat([df, dfn], axis=1)
    return df

NON_DER = ['Day',]
df_new = df_derived_by_shift(covid, 16)

df_new.head(10)

df_new = df_new.dropna()
df_new.head(10)

y = df_new.corr() 


import seaborn as sns
colormap = plt.cm.RdBu
plt.figure(figsize=(40,35))
plt.title(u'covid', y=1.05, size=46)

mask = np.zeros_like(df_new.corr())
mask[np.triu_indices_from(mask)] = True

svm = sns.heatmap(df_new.corr(), mask=mask, linewidths=2.0,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

#lagccofCOVID-19 keyword
fields = ['Day','COVID-19','new_cases'] 
COVID_19 = dfx[fields]
COVID_19.head(10)


NON_DER = ['Day',]
df_new = df_derived_by_shift(COVID_19, 16)

df_new.head(10)

df_new = df_new.dropna()
df_new.head(10)

y1 = df_new.corr() 



colormap = plt.cm.RdBu
plt.figure(figsize=(40,35))
plt.title(u'COVID-19', y=1.05, size=46)

mask = np.zeros_like(df_new.corr())
mask[np.triu_indices_from(mask)] = True

svm = sns.heatmap(df_new.corr(), mask=mask, linewidths=2.0,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

#lagccofcoronavirus
fields = ['Day','coronavirus','new_cases'] 
coronavirus = dfx[fields]
coronavirus.head(10)

NON_DER = ['Day',]
df_new = df_derived_by_shift(coronavirus, 16)

df_new.head(10)

df_new = df_new.dropna()
df_new.head(10)

y3 = df_new.corr() 



colormap = plt.cm.RdBu
plt.figure(figsize=(40,35))
plt.title(u'coronavirus', y=1.05, size=46)

mask = np.zeros_like(df_new.corr())
mask[np.triu_indices_from(mask)] = True

svm = sns.heatmap(df_new.corr(), mask=mask, linewidths=2.0,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

#lagccofcoronavirusinbangla
fields = ['Day','coronavirus in bangla','new_cases'] 
coronavirusinbangla = dfx[fields]
coronavirusinbangla.head(10)


NON_DER = ['Day',]
df_new = df_derived_by_shift(coronavirusinbangla, 16)

df_new.head(10)

df_new = df_new.dropna()
df_new.head(10)

y4 = df_new.corr() 



colormap = plt.cm.RdBu
plt.figure(figsize=(40,35))
plt.title(u'coronavirus in bangla', y=1.05, size=46)

mask = np.zeros_like(df_new.corr())
mask[np.triu_indices_from(mask)] = True

svm = sns.heatmap(df_new.corr(), mask=mask, linewidths=2.0,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

#lagccofpneumonia

fields = ['Day','pneumonia','new_cases'] 
pneumonia = dfx[fields]
pneumonia.head(10)


NON_DER = ['Day',]
df_new = df_derived_by_shift(pneumonia, 16)

df_new.head(10)

df_new = df_new.dropna()
df_new.head(10)

y5 = df_new.corr() 

colormap = plt.cm.RdBu
plt.figure(figsize=(40,35))
plt.title(u'pneumonia', y=1.05, size=46)

mask = np.zeros_like(df_new.corr())
mask[np.triu_indices_from(mask)] = True

svm = sns.heatmap(df_new.corr(), mask=mask, linewidths=2.0,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

#lagccofCoronavirusvaccine

fields = ['Day','Coronavirus vaccine','new_cases'] 
Coronavirusvaccine = dfx[fields]
Coronavirusvaccine.head(10)


NON_DER = ['Day',]
df_new = df_derived_by_shift(Coronavirusvaccine, 16)

df_new.head(10)

df_new = df_new.dropna()
df_new.head(10)

y6 = df_new.corr() 



colormap = plt.cm.RdBu
plt.figure(figsize=(40,35))
plt.title(u'Coronavirus vaccine', y=1.05, size=46)

mask = np.zeros_like(df_new.corr())
mask[np.triu_indices_from(mask)] = True

svm = sns.heatmap(df_new.corr(), mask=mask, linewidths=2.0,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


#lagccofhightemperature

fields = ['Day','high temperature','new_cases'] 
temp = dfx[fields]
temp.head(10)

NON_DER = ['Day',]
df_new = df_derived_by_shift(temp, 16)

df_new.head(10)

df_new = df_new.dropna()
df_new.head(10)

y7 = df_new.corr() 



colormap = plt.cm.RdBu
plt.figure(figsize=(40,35))
plt.title(u'high temperature', y=1.05, size=46)

mask = np.zeros_like(df_new.corr())
mask[np.triu_indices_from(mask)] = True

svm = sns.heatmap(df_new.corr(), mask=mask, linewidths=2.0,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

#lagccofcough
fields = ['Day','cough','new_cases'] 
cough = dfx[fields]
cough.head(10)

NON_DER = ['Day',]
df_new = df_derived_by_shift(cough, 16)

df_new.head(10)

df_new = df_new.dropna()
df_new.head(10)

y8 = df_new.corr() 



colormap = plt.cm.RdBu
plt.figure(figsize=(40,35))
plt.title(u'cough', y=1.05, size=46)

mask = np.zeros_like(df_new.corr())
mask[np.triu_indices_from(mask)] = True

svm = sns.heatmap(df_new.corr(), mask=mask, linewidths=2.0,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

#lagccofcoronavirustest
fields = ['Day','coronavirus test','new_cases'] 
coronavirustest = dfx[fields]
coronavirustest.head(10)

NON_DER = ['Day',]
df_new = df_derived_by_shift(coronavirustest, 16)

df_new.head(10)

df_new = df_new.dropna()
df_new.head(10)

y9 = df_new.corr() 



colormap = plt.cm.RdBu
plt.figure(figsize=(40,35))
plt.title(u'coronavirustest', y=1.05, size=46)

mask = np.zeros_like(df_new.corr())
mask[np.triu_indices_from(mask)] = True

svm = sns.heatmap(df_new.corr(), mask=mask, linewidths=2.0,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

#lagccofMusk
fields = ['Day','Musk','new_cases'] 
Musk = dfx[fields]
Musk.head(10)

NON_DER = ['Day',]
df_new = df_derived_by_shift(Musk, 16)

df_new.head(10)

df_new = df_new.dropna()
df_new.head(10)

y10 = df_new.corr() 



colormap = plt.cm.RdBu
plt.figure(figsize=(40,35))
plt.title(u'Musk', y=1.05, size=46)

mask = np.zeros_like(df_new.corr())
mask[np.triu_indices_from(mask)] = True

svm = sns.heatmap(df_new.corr(), mask=mask, linewidths=2.0,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)



#plotting covid cases againts the keywords rsv 

x = df.iloc[:, 1].values
y = df.iloc[:, 12].values
plt.xlabel('covid')
plt.ylabel('COVID Cases')
plt.scatter(x ,y , color = 'green')

x = df.iloc[:, 2].values
y = df.iloc[:, 12].values
plt.xlabel('RSV of COVID-19')
plt.ylabel('COVID Cases')
plt.scatter(x ,y , color = 'green')

x = df.iloc[:, 3].values
y = df.iloc[:, 12].values
plt.xlabel('coronavirus')
plt.ylabel('COVID Cases')
plt.scatter(x ,y , color = 'green')

x = df.iloc[:, 4].values
y = df.iloc[:, 12].values
plt.xlabel('coronavirus in bangla')
plt.ylabel('COVID Cases')
plt.scatter(x ,y , color = 'green')

x = df.iloc[:, 5].values
y = df.iloc[:, 12].values
plt.xlabel('pneumonia')
plt.ylabel('COVID Cases')
plt.scatter(x ,y , color = 'green')

x = df.iloc[:, 6].values
y = df.iloc[:, 12].values
plt.xlabel('coronavirus vaccine')
plt.ylabel('COVID Cases')
plt.scatter(x ,y , color = 'green')

x = df.iloc[:, 7].values
y = df.iloc[:, 12].values
plt.xlabel('high temperature')
plt.ylabel('COVID Cases')
plt.scatter(x ,y , color = 'green')

x = df.iloc[:, 8].values
y = df.iloc[:, 12].values
plt.xlabel('cough')
plt.ylabel('COVID Cases')
plt.scatter(x ,y , color = 'green')

x = df.iloc[:, 9].values
y = df.iloc[:, 12].values
plt.xlabel('coronairus test')
plt.ylabel('COVID Cases')
plt.scatter(x ,y , color = 'green')

x = df.iloc[:, 10].values
y = df.iloc[:, 12].values
plt.xlabel('Musk')
plt.ylabel('COVID Cases')
plt.scatter(x ,y , color = 'green')