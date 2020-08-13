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

import difflib

import scipy.signal as spsig
import scipy.stats as stats
import itertools

from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA

from coronapy.dataform import *

# %config InlineBackend.figure_format ='retina'
# %matplotlib inline
plt.rcParams['figure.figsize'] = [20, 13]

def toGeoDF(df, geometry="geog"):
    tempDF = df.copy()
    tempDF[geometry] = df[geometry].apply(wkt.loads)
    return gpd.GeoDataFrame(tempDF, geometry=geometry)


cwd = os.getcwd()
# fcsv = '../coronavirus-csv/coronavirus_dataset.csv'
fcsv = '../coronavirus/csv/coronavirus.csv'
assert os.path.isfile(fcsv), f"{fcsv} does not exist in {os.getcwd()}"


df = pnd.read_csv(fcsv)
print(f"loading John Hopkins University of Medecine data, cleaning and reformating dataset")
ndf, error = transform_metro(df)

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


worst_plot = sns.barplot(x="date", y="cumul death", hue="Country", data=ndf[ndf.Country.isin(worstdf.Country[-10:])])
worst_plot.set_xticklabels(worst_plot.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()

ff.create_table(ndf.query("'France' in Country"))


####################################
# # DATA ANALYSIS

focus = ['Norway', 'Sweden', 'France', 'United Kingdom', 'Italy', 'Iran',
         'China', 'Spain', 'Taiwan*', 'Korea, South', 'Singapore', 'Israel',
         'Netherlands', 'Nigeria', 'US']
combinations = itertools.combinations(focus, 2)
corrcoeff = np.zeros((len(focus), len(focus)))
pvalues = np.zeros((len(focus), len(focus)))
res = {}
for country in focus:
    res[country] = (ndf.query("@country in Country").confirmed.values)

res = pnd.DataFrame(res, columns=focus)

####################################
# Correlation
for combi in combinations:
    corrcoeff[focus.index(combi[0])][focus.index(combi[1])], pvalues[focus.index(combi[0])][focus.index(combi[1])] = stats.pearsonr(ndf.query("@combi[0] in Country").confirmed, ndf.query("@combi[1] in Country").confirmed)    

corrcoeff = np.triu(corrcoeff) + np.triu(corrcoeff, 1).T
np.fill_diagonal(corrcoeff, 1)
pvalues = np.triu(pvalues) + np.triu(pvalues, 1).T
np.fill_diagonal(pvalues, 1)

corrcf_plot = sns.heatmap(corrcoeff, xticklabels=focus, yticklabels=focus, annot=True)
corrcf_plot.set_xticklabels(corrcf_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
corrcf_plot.set_title('Correlations')

#sns.heatmap(res.corr(), annot=True).set_title('Correlations')
plt.show()
# Covariance, not Correct?
covariances_plot = sns.heatmap(res.cov())
covariances_plot.set_title('Covariances')
covariances_plot.set_xticklabels(covariances_plot.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()

####################################
# Cross correlations

focus = ['Norway', 'Sweden', 'France', 'United Kingdom', 'Germany', 'Italy', 'Iran',
         'China', 'Spain', 'Taiwan*', 'Korea, South', 'Singapore', 'Israel', 'Netherlands',
         'Nigeria', 'US']
combinations = itertools.combinations(focus, 2)
# crosscorr = np.zeros((len(focus), len(focus)))
# pvalues = np.zeros((len(focus), len(focus)))
res = {}
for country in focus:
    res[country] = (ndf.query("@country in Country").confirmed.values)

res = pnd.DataFrame(res, columns=focus)
crosscorr_df = {}
for combi in combinations:
    crosscorr_df[combi] = spsig.correlate(ndf.query("@combi[0] in Country").confirmed, ndf.query("@combi[1] in Country").confirmed)

crosscorr_df = pnd.DataFrame(crosscorr_df)

ax = plt.figure().add_subplot(111)
ax.plot(spsig.correlate(ndf.query("'France' in Country").death / ndf.query("'France' in Country").death.max(),
                         ndf.query("'Italy' in Country").death / ndf.query("'Italy' in Country").death.max()))
ax.set_title('France Italy correlation death')

reference = 'Norway'
con, death, reco = crosscorr(reference, True, ndf)
m = con.max()
m.sort_values()[-10:].index.tolist()
con[con.max().sort_values()[-10:].index.tolist()].plot(title=f"Cross-correlation {reference.upper()} confirmed")
death[death.max().sort_values()[-10:].index.tolist()].plot(title=f"Cross-correlation {reference.upper()} death")
reco[reco.max().sort_values()[-10:].index.tolist()].plot(title=f"Cross-correlation {reference.upper()} recovered")
plt.show()


undf = pnd.read_csv('external_data/UNData_Population, Surface Area and Density.csv', encoding='ISO-8859-1')
undf.rename(columns={'Unnamed: 3': 'series'}, inplace=True)
undf.rename(columns={'Unnamed: 4': 'numerics'}, inplace=True)
undf.rename(columns={'Unnamed: 6': 'source'}, inplace=True)
undf.rename(columns={'Unnamed: 5': 'notes'}, inplace=True)
undf.rename(columns={'Unnamed: 2': 'years'}, inplace=True)
undf.rename(columns={'Population, density and surface area': 'countries'}, inplace=True)
undf.rename(columns={'T02': 'area_code'}, inplace=True)
undf.drop([0], inplace=True)  # drop redundant labels row

# Manual correction of the name format between the two dataframe:
print('Looking for mismatch country name between the UN dataset and the JHU covid dataset')
mismatched = []
closest = []
for c in tqdm(ndf.Country.unique()):
    if c not in undf.countries.unique():
        mismatched.append(c)
        closest.append(difflib.get_close_matches(c, undf.countries.unique()[1:], cutoff=.3))

for i, c in enumerate(mismatched):
    print(f"{c} not found in UN dataset, closest automatic match: {closest[i]}")

print('Manual Correction')
undf.loc[undf.countries == 'Bolivia (Plurin. State of)', 'countries'] = 'Bolivia'
undf.loc[undf.countries == 'Brunei Darussalam', 'countries'] = 'Brunei'
undf.loc[undf.countries == 'Myanmar', 'countries'] = 'Burma'
undf.loc[undf.countries == 'Côte d\x92Ivoire', 'countries'] = "Cote d'Ivoire"
undf.loc[undf.countries == 'Iran (Islamic Republic of)', 'countries'] = 'Iran'
undf.loc[undf.countries == 'Dem. Rep. of the Congo', 'countries'] = 'Korea, South'
undf.loc[undf.countries == "Lao People's Dem. Rep.", 'countries'] = 'Laos'
undf.loc[undf.countries == 'Republic of Moldova', 'countries'] = 'Moldova'
undf.loc[undf.countries == 'Russian Federation', 'countries'] = 'Russia'
undf.loc[undf.countries == 'Saint Vincent & Grenadines', 'countries'] = 'Saint Vincent and the Grenadines'
undf.loc[undf.countries == 'Syrian Arab Republic', 'countries'] = 'Syria'
undf.loc[undf.countries == 'United Rep. of Tanzania', 'countries'] = 'Tanzania'
undf.loc[undf.countries == 'United States of America', 'countries'] = 'US'
undf.loc[undf.countries == 'Venezuela (Boliv. Rep. of)', 'countries'] = 'Venezuela'
undf.loc[undf.countries == 'State of Palestine', 'countries'] = 'West Bank and Gaza'
undf.loc[undf.countries == 'Viet Nam', 'countries'] = 'Vietnam'


# undf.loc[undf.countries == 'Viet Nam', 'countries'] = 'Taiwan*'
# China does not include Taiwan  numbers in the UN dataset. However there is no entry fo rTaiwan. DATA MISSING
# undf.loc[undf.countries == 'Bolivia (Plurin. State of)', 'countries'] = 'Kosovo'
# Serbia i nUN dataset includes Kosovo numbers

# Fix Congo entries
# undf.loc[undf.countries == 'Dem. Rep. of the Congo', 'countries'] = 'Congo (Brazzaville)'
# undf.loc[undf.countries == 'Dem. Rep. of the Congo', 'countries'] = 'Congo (Kinshasa)'

print('Looking for uncorrected mismatch country name between the UN dataset and the JHU covid dataset')
mismatched = []
closest = []
for c in tqdm(ndf.Country.unique()):
    if c not in undf.countries.unique():
        mismatched.append(c)
        closest.append(difflib.get_close_matches(c, undf.countries.unique()[1:], cutoff=.3))

for i, c in enumerate(mismatched):
    print(f"{c} not found in UN dataset, closest automatic match: {closest[i]}")




focus = ['Norway', 'Sweden', 'France', 'United Kingdom', 'Germany', 'Italy', 'Iran',
         'China', 'Spain', 'Korea, South', 'Singapore', 'Israel', 'Netherlands',
         'Switzerland', 'US', 'Brazil', 'India', 'Japan']


fig = plt.figure()

mini = ddf.query("@focus in Country")  


#for c in focus:

for serie in list_series(undf):
    # 2 plots: confirmed, deaths

    # Do something ofr figures 15 countries
    for c in focus:
        # plot
        get_latest_data(undf, c, serie)


# compute and display the X worst and Y best faring countries in % of pop, and other variables: GDP, % old pop, ...


# compute and assess per capita results with some wikipedia entries.

################
################
# ###  TODO  ####
'''
* fix multiple congo entries
* broken: plotly display figures (should display on a new browser page)

* mutual information
* CCA canonical correlation analysis: canonical-correlation analysis will find linear combinations of X and Y which have maximum correlation with each other (wikipedia)
** scikit-learn cross decomposition, split training / test data
* update import to name instead of *
* load dataset once, then only run analysis
* finish import jupyter notebooks
* rename columns addtional dataset (demographics)

* get total population by country (wikipedia crawler?)
* get previous years death counts, bad flu years (what are the bad flu years?)
* get GDP, density of country and plot death / cases
* plot by continents
* compute recovery time from recovered data?

* clustering
** train on  raw data, then classify per capita data

* GIS

correlation with:
* air quality
* industrial indicators
* UN rank
* datasets: WID, Humanitarian Data Exchange

** correlation evolution wrt time  
'''
