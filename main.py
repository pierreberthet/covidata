import pandas as pnd
import geopandas as gpd
import seaborn as sns
from tqdm import trange, tqdm

import shapely
from shapely import wkt
from shapely.geometry import Point

from matplotlib import pyplot as plt
from datetime import timedelta, date
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import argparse
import os

import scipy.signal as spsig
import scipy.stats as stats
import itertools


from dataform import *

#%config InlineBackend.figure_format ='retina'
#%matplotlib inline
plt.rcParams['figure.figsize'] = [20, 13]

def toGeoDF(df, geometry="geog"):
    tempDF = df.copy()
    tempDF[geometry] = df[geometry].apply(wkt.loads)
    return gpd.GeoDataFrame(tempDF, geometry=geometry)

cwd = os.getcwd()
fcsv = '../coronavirus-csv/coronavirus_dataset.csv'
assert os.path.isfile(fcsv), f"{fcsv} does not exist in {os.getcwd()}"


df = pnd.read_csv(fcsv)
print(f"loading JHH data, cleaning and reformating dataset")
ndf = transform_metro(df)

ddf = ndf.copy()



ndf.confirmed.plot()


worst_nb = 5
worst_days = 15
worstdf = get_Xworst_Ydays(ndf, worst_nb, worst_days)
print(f" {worst_nb} worst countries for the last {worst_days} days")
print(f"{worstdf}")

timeline_global(ndf, True, 'line')

plt.show()

focus_countries = ['Norway', 'Sweden', 'France', 'United Kingdom', 'Germany', 'Italy', 'China', 'Spain', 'Taiwan*', 'Korea, South', 'Singapore', 'Israel', 'Netherlands', 'Nigeria', 'US']
#focus_countries = ['Norway', 'Sweden', 'United Kingdom', 'Italy', 'Spain', 'Taiwan*', 'Korea, South', 'Singapore', 'Israel', 'Netherlands', 'Nigeria']
focus_countries.sort()
focus_countries = remove_country_with_province(ndf, focus_countries)
print(focus_countries)
focusdf = global_description(ndf, focus_countries)
focusdf.sort_values(by='new deaths')


sns.barplot(x="date", y="cumul death", hue="Country", data=ndf[ndf.Country.isin(worstdf.Country[-10:])])


ff.create_table(ndf.query("'France' in Country"))



## DATA ANALYSIS

focus = ['Norway', 'Sweden', 'France', 'United Kingdom', 'Italy', 'Iran', 'China', 'Spain', 'Taiwan*', 'Korea, South', 'Singapore', 'Israel', 'Netherlands', 'Nigeria', 'US']
combinations = itertools.combinations(focus, 2)
corrcoeff = np.zeros((len(focus), len(focus)))
pvalues = np.zeros((len(focus), len(focus)))
res = {}
for country in focus:
    res[country] = (ndf.query("@country in Country").confirmed.values)

res = pnd.DataFrame(res, columns=focus)


# Correlation
for combi in combinations:
    corrcoeff[focus.index(combi[0])][focus.index(combi[1])], pvalues[focus.index(combi[0])][focus.index(combi[1])] = stats.pearsonr(ndf.query("@combi[0] in Country").confirmed, ndf.query("@combi[1] in Country").confirmed)    

corrcoeff = np.triu(corrcoeff) + np.triu(corrcoeff,1).T
np.fill_diagonal(corrcoeff, 1)
pvalues = np.triu(pvalues) + np.triu(pvalues,1).T
np.fill_diagonal(pvalues, 1)

sns.heatmap(corrcoeff, xticklabels=focus, yticklabels=focus, annot=True)
plt.show()
sns.heatmap(res.corr(), annot=True)
plt.show()
sns.heatmap(res.cov(), annot=True)
plt.show()

# Cross correlations

focus = ['Norway', 'Sweden', 'France', 'United Kingdom', 'Germany', 'Italy', 'Iran', 'China', 'Spain', 'Taiwan*', 'Korea, South', 'Singapore', 'Israel', 'Netherlands', 'Nigeria', 'US']
combinations = itertools.combinations(focus, 2)
#crosscorr = np.zeros((len(focus), len(focus)))
#pvalues = np.zeros((len(focus), len(focus)))
res = {}
for country in focus:
    res[country] = (ndf.query("@country in Country").confirmed.values)

res = pnd.DataFrame(res, columns=focus)
crosscorr = {}
for combi in combinations:
    crosscorr[combi] =  spsig.correlate(ndf.query("@combi[0] in Country").confirmed, ndf.query("@combi[1] in Country").confirmed)    

crosscorr = pnd.DataFrame(crosscorr)

plt.plot(spsig.correlate(ndf.query("'France' in Country").death / ndf.query("'France' in Country").death.max(),
                ndf.query("'Italy' in Country").death / ndf.query("'Italy' in Country").death.max()))


reference = 'France'
con, death, reco = crosscorr(reference, True, ndf)
m = con.max()
m.sort_values()[-10:].index.tolist()
con[con.max().sort_values()[-10:].index.tolist()].plot(title=f"Cross-correlation {reference.upper()} confirmed")
