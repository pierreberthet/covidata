import pandas as pnd
import geopandas as gpd
import seaborn as sns
from tqdm import trange, tqdm

import shapely
from shapely import wkt
from shapely.geometry import Point

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import timedelta, date
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import argparse
import os


import scipy.signal as spsig
import scipy.stats as stats
import itertools


def country_basic_data(df, country) -> dict:
    # total cases all types, change since previous update (day before check date of upload then)
    ctry = get_country(df, country)
    return {'confirmed': ctry['cumul confirmed'].iloc[-1],
            'new confirmed': ctry['confirmed'].iloc[-1],
            'deaths': ctry['cumul death'].iloc[-1],
            'new deaths': ctry['death'].iloc[-1]}


def global_description(df, focus=None) -> pnd.DataFrame:
    print(f"Data from {df['date'].unique().max()}")
    res = {}
    if focus is None:
        for country in list_all_countries(df):
            res[country] = country_basic_data(df, country)
    else:
        for country in focus:
            res[country] = country_basic_data(df, country)
    res = pnd.DataFrame(res).T
    res.loc['Total'] = res.sum()
    return res

    
def get_country(df, country) -> pnd.DataFrame:
    assert country in df['Country'].unique().tolist(), print(f"{country} not in list, options are {list_all_countries(df)}")
    return df.query("@country in Country")


def get_province(df, province) -> pnd.DataFrame:
    assert province in df['Province'].unique().tolist(), print(f"{province} not in list, options are {list_provinces(df)}")
    return df.query("@province in Province")


def list_all_countries(df) -> list:
    return df['Country'].unique().tolist()


def list_provinces(df) -> list:
    return df['Province'].unique().tolist()


def get_status(df, status):
    '''deprecated'''
    assert status in df['type'].unique().tolist(), print(f"possible status are {(df['type'].unique().tolist())}")
    return df[df['type'] == status]
          
                                                         
def get_countryWith_provinces(df) -> list:
    '''returns a list of the name of countries having provinces'''                                                         
    res = []
    for country in list_all_countries(df):
        if has_province(get_country(df, country)):
            res.append(country)
    return sorted(res)
                                                         
                                                        
def has_province(df) -> bool:
    return len(df['Province'].unique()) > 1 


def add_running_sum(df) -> pnd.DataFrame:
    '''Returns a dataframe with cumulative entries for the cases [confirmed, recovered, death]
    Specific to a country
    For a multi country list, call 'global_running_sum(df)' instead'''
    if has_province(df):
        for province in list_provinces(df):
            add_running_sum(get_province(df, province))
    extra = np.zeros(len(df), dtype=int)  # extra column entries
    rsum = np.zeros(2)  # running sum of the 3 types
    i = 1
    for _, entry in df[1:].iterrows():
        #print(f"confirmed!!! entry {entry} cumul {rsum[0]}")

        if entry['type'] == 'confirmed':
            rsum[0] += entry['cases']
            extra[i] = rsum[0] 
        elif entry['type'] == 'death':
            rsum[1] += entry['cases']
            extra[i] = rsum[1] 
        i += 1
    df = df.assign(cumul=extra)
    return df
                                                         
def global_running_sum(df) -> pnd.DataFrame:
    for country in get_all_countries(df):
        if has_province(df[df['Country.Region'] == country]):
            for province in get_provinces(df[df['Country.Region'] == country]):
                pass
    return None
                                                         

def remove_country_with_province(df: pnd.DataFrame, countries=None) -> list:
    if countries is None:
        countries = list_all_countries(ndf)
    res = []
    for country in countries:
        if country not in get_countryWith_provinces(df):
            res.append(country)            
    return res


def transform_metro(df:pnd.DataFrame) -> pnd.DataFrame:
    '''Reorganize data with cumul columns and specific case types columns: recovered, death, confirmed.
    Also, for Australia, Canada and China, we sum the data from provinces and return one entry for the country.
    For Denmark, France, Netherlands and the UK, we here discard the non mainland territories. For a different
    approach, please see transform_province(). Also, it puts the 'recovered' data from the Canada entries with
    the cumulative data from the provinces.'''
    #ndf = pnd.DataFrame(columns=['Country', 'Province', 'lat', 'lon', 'date', 'confirmed', 'death', 'recovered'])
    res = []
    for country in tqdm(df['Country.Region'].unique().tolist()):
        tdf = df[df['Country.Region'] == country].fillna(country)
        if country != 'Canada':  ## Canada does not have 'recovered' data, special case below.
            if len(tdf['Province.State'].unique()) > 1:
                if country in ['China', 'Australia']:
                    rsum = np.zeros(3, dtype=int)  # running sum of the 3 types
                    for date in tdf['date'].unique():
                        dsum = np.zeros(3, dtype=int)
                        for province in tdf['Province.State'].unique().tolist():
                            ttdf = tdf[tdf['Province.State'] == province]
                            if province != country and 'Princess' not in province:
                                c = ttdf[ttdf['date'] == date]
                                dsum[0] += c[c['type']=='confirmed']['cases'].values[0]
                                dsum[1] += c[c['type']=='death']['cases'].values[0]
                                dsum[2] += c[c['type']=='recovered']['cases'].values[0]
                        rsum[0] += dsum[0]
                        rsum[1] += dsum[1]
                        rsum[2] += dsum[2]
                        res.append({'Country':country, 'Province': 'mainland', 'Lat':c.iloc[0]['Lat'], 'Long':c.iloc[0]['Long'], 'date':date,
                                    'confirmed':dsum[0],
                                    'death':dsum[1],
                                    'recovered':dsum[2],
                                    'cumul confirmed':rsum[0], 'cumul death':rsum[1], 'cumul recovered':rsum[2]
                                   })
                        
                elif country in ['France', 'Denmark', 'Netherlands', 'United Kingdom']:
                    rsum = np.zeros(3, dtype=int)  # running sum of the 3 types
                    for date in tdf['date'].unique():
                        dsum = np.zeros(3, dtype=int)
                        ttdf = tdf[tdf['Province.State'] == country]
                        c = ttdf[ttdf['date'] == date]
                        dsum[0] = c[c['type']=='confirmed']['cases'].values[0]
                        dsum[1] = c[c['type']=='death']['cases'].values[0]
                        dsum[2] = c[c['type']=='recovered']['cases'].values[0]
                        rsum[0] += dsum[0]
                        rsum[1] += dsum[1]
                        rsum[2] += dsum[2]
                        res.append({'Country':country, 'Province': 'mainland', 'Lat':c.iloc[0]['Lat'], 'Long':c.iloc[0]['Long'], 'date':date,
                                    'confirmed':dsum[0],
                                    'death':dsum[1],
                                    'recovered':dsum[2],
                                    'cumul confirmed':rsum[0], 'cumul death':rsum[1], 'cumul recovered':rsum[2]
                                   })
            else:
                rsum = np.zeros(3, dtype=int)  # running sum of the 3 types
                for date in tdf['date'].unique():
                    c = tdf[tdf['date'] == date]
                    rsum[0] += c[c['type']=='confirmed']['cases'].values[0]
                    rsum[1] += c[c['type']=='death']['cases'].values[0]
                    rsum[2] += c[c['type']=='recovered']['cases'].values[0]
                    res.append({'Country':country, 'Province':None, 'Lat':c.iloc[0]['Lat'], 'Long':c.iloc[0]['Long'], 'date':date,
                                'confirmed':c[c['type']=='confirmed']['cases'].values[0],
                                'death':c[c['type']=='death']['cases'].values[0],
                                'recovered':c[c['type']=='recovered']['cases'].values[0],
                                'cumul confirmed':rsum[0], 'cumul death':rsum[1], 'cumul recovered':rsum[2]
                                 })
        else: # Canada specific
            #for province in tdf['Province.State'].unique().tolist():
                #if province != 'Canada':
            drecov = tdf[tdf['Province.State'] == 'Canada']
            rsum = np.zeros(3, dtype=int)  # running sum of the 3 types
            for date in tdf['date'].unique():
                dsum = np.zeros(3, dtype=int)
                for province in tdf['Province.State'].unique().tolist():
                    if province != country and 'Princess' not in province:
                        ttdf = tdf[tdf['Province.State'] == province]
                        c = ttdf[ttdf['date'] == date]
                        dsum[0] += c[c['type']=='confirmed']['cases'].values[0]
                        dsum[1] += c[c['type']=='death']['cases'].values[0]
                        dsum[2] += drecov[drecov['type']=='recovered']['cases'].values[0]
                rsum[0] += dsum[0]
                rsum[1] += dsum[1]
                rsum[2] += dsum[2]
                res.append({'Country':country, 'Province': 'mainland', 'Lat':c.iloc[0]['Lat'], 'Long':c.iloc[0]['Long'], 'date':date,
                            'confirmed':dsum[0],
                            'death':dsum[1],
                            'recovered':dsum[2],
                            'cumul confirmed':rsum[0], 'cumul death':rsum[1], 'cumul recovered':rsum[2]
                           })
    return pnd.DataFrame(res)

def transform(df:pnd.DataFrame) -> pnd.DataFrame:
    #ndf = pnd.DataFrame(columns=['Country', 'Province', 'lat', 'lon', 'date', 'confirmed', 'death', 'recovered'])
    res = []
    for country in df['Country.Region'].unique().tolist():
        tdf = df[df['Country.Region'] == country].fillna(country)
        if country != 'Canada':  ## Canada does not have 'recovered' data, special case below.
            if len(tdf['Province.State'].unique()) > 1:
                for province in tdf['Province.State'].unique().tolist():
                    ttdf = tdf[tdf['Province.State'] == province]
                    rsum = np.zeros(3, dtype=int)  # running sum of the 3 types
                    i = 1
                    for date in ttdf['date'].unique():
                        c = ttdf[ttdf['date'] == date]
                        rsum[0] += c[c['type']=='confirmed']['cases'].values[0]
                        rsum[1] += c[c['type']=='death']['cases'].values[0]
                        rsum[2] += c[c['type']=='recovered']['cases'].values[0]
                        res.append({'Country': country, 'Province':province, 'Lat': c.iloc[0]['Lat'], 'Long': c.iloc[0]['Long'], 'date':date,
                                    'confirmed': c[c['type'] == 'confirmed']['cases'].values[0],
                                    'death': c[c['type'] == 'death']['cases'].values[0],
                                    'recovered': c[c['type'] == 'recovered']['cases'].values[0],
                                    'cumul confirmed': rsum[0], 'cumul death': rsum[1], 'cumul recovered': rsum[2]
                                   })
            else:
                rsum = np.zeros(3, dtype=int)  # running sum of the 3 types
                i = 1
                for date in tdf['date'].unique():
                    c = tdf[tdf['date'] == date]
                    rsum[0] += c[c['type']=='confirmed']['cases'].values[0]
                    rsum[1] += c[c['type']=='death']['cases'].values[0]
                    rsum[2] += c[c['type']=='recovered']['cases'].values[0]
                    res.append({'Country':country, 'Province':None, 'Lat':c.iloc[0]['Lat'], 'Long':c.iloc[0]['Long'], 'date':date,
                                'confirmed':c[c['type']=='confirmed']['cases'].values[0],
                                'death':c[c['type']=='death']['cases'].values[0],
                                'recovered':c[c['type']=='recovered']['cases'].values[0],
                                'cumul confirmed':rsum[0], 'cumul death':rsum[1], 'cumul recovered':rsum[2]
                                 })
        else:
            for province in tdf['Province.State'].unique().tolist():
                if province != 'Canada':
                    ttdf = tdf[tdf['Province.State'] == province]
                    rsum = np.zeros(2, dtype=int)  # running sum of the 2 types for 'Canada'
                    i = 1
                    for date in ttdf['date'].unique():
                        c = ttdf[ttdf['date'] == date]
                        rsum[0] += c[c['type']=='confirmed']['cases'].values[0]
                        rsum[1] += c[c['type']=='death']['cases'].values[0]
                        res.append({'Country':country, 'Province':province, 'Lat':c.iloc[0]['Lat'], 'Long':c.iloc[0]['Long'], 'date':date,
                                    'confirmed':c[c['type']=='confirmed']['cases'].values[0],
                                    'death':c[c['type']=='death']['cases'].values[0],
                                    'cumul confirmed':rsum[0], 'cumul death':rsum[1],
                                    'cumul recovered':None, 'recovered':None
                                   })
    return pnd.DataFrame(res)


def add_running_sum2(df) -> pnd.DataFrame:
    '''Returns a dataframe with cumulative entries for the cases [confirmed, recovered, death]
    Specific to a country
    For a multi country list, call 'global_running_sum(df)' instead'''
    for country in list_all_countries(df):
        if has_province(df):
            for province in list_provinces(df):
                pass
        extra = np.zeros(len(df), dtype=int)  # extra column entries
        rsum = np.zeros(2, dtype=int)  # running sum of the 3 types
        i = 1
        for _, entry in df[1:].iterrows():
            rsum[0] += int(entry['confirmed'])
            extra[i] = rsum[0] 
            rsum[1] += int(entry['death'])
            extra[i] = rsum[1] 
            i += 1
        df = df.assign(cumul=extra)
    return df


def test_raw_df(df:pnd.DataFrame) -> dict:
    res = {'recovered': True, 'confirmed': True, 'death': True, 'date': True, 'positive': True}
    for country in df['Country.Region'].unique():
        c = df[df['Country.Region'] == country]
        if len(c['Province.State'].unique()) > 1:
            for province in c['Province.State'].unique():
                if 'recovered' not in c[c['Province.State'] == province].type.tolist():
                    res['recovered'] = False
                    print(f"{country} {province} does not have recovered entries")                
                if 'confirmed' not in c[c['Province.State'] == province].type.tolist():
                    res['confirmed'] = False
                    print(f"{country} {province} does not have confirmed entries")
                if 'death' not in c[c['Province.State'] == province].type.tolist():
                    res['death'] = False
                    print(f"{country} {province} does not have death entries")
                elif np.min(c[c['Province.State'] == province].query("'death' in type").cases) < 0:
                    print(f"{country} {province} has a negative death entry")
                    res['positive'] = False
    
        else:
            if 'recovered' not in c.type.tolist():
                res['recovered'] = False
                print(f"{country} does not have recovered entries")                
            if 'confirmed' not in c.type.tolist():
                res['confirmed'] = False
                print(f"{country} does not have confirmed entries")
            if 'death' not in c.type.tolist():
                res['death'] = False
                print(f"{country}  does not have death entries")

            elif np.min(c.query("'death' in type").cases) < 0:
                    print(f"{country} has a negative death entry")
                    res['positive'] = False
    
    return res

def test_transformed_df(df:pnd.DataFrame) -> dict:
    res = {'recovered': True, 'confirmed': True, 'death': True, 'date': True, 'positive': True}
    for country in df['Country.Region'].unique():
        c = df[df['Country.Region'] == country]
        for province in c['Province.State'].unique():
            if 'recovered' not in c[c['Province.State'] == province].type:
                res['recovered'] = False
                print(f"{country} {province} does not have recovered entries")                
            if 'confirmed' not in c[c['Province.State'] == province].type:
                res['confirmed'] = False
                print(f"{country} {province} does not have confirmed entries")
            if 'death' not in c[c['Province.State'] == province].type:
                res['death'] = False
                print(f"{country} {province} does not have death entries")
                
            if np.min(c[c['Province.State'] == province].cases) < 0:
                print(f"{country} {province} has a negative value case")
    return res


def transform_debug(df: pnd.DataFrame) -> pnd.DataFrame:
    #ndf = pnd.DataFrame(columns=['Country', 'Province', 'lat', 'lon', 'date', 'confirmed', 'death', 'recovered'])
    res = []
    #df = df.fillna('Metro')
    
    #df = df.drop(df['Province.State']=='Recovered')
    #df = remove_Princess(df)
    for country in df['Country.Region'].unique().tolist():
        print(country)
        tdf = df[df['Country.Region'] == country].fillna(country)
        if len(tdf['Province.State'].unique()) > 1:
            for province in tdf['Province.State'].unique().tolist():
                if province != 'Canada':  # Canada has provinces, and a generic entry with only recovered cases.
                    print(province)
                    ttdf = tdf[tdf['Province.State'] == province]
                    rsum = np.zeros(2, dtype=int)  # running sum of the 3 types
                    i = 1
                    for date in ttdf['date'].unique():
                        c = ttdf[ttdf['date'] == date]
                        print('--------')
                        print(c[c['type'] == 'confirmed']['cases'].values)
                        rsum[0] += c[c['type'] == 'confirmed']['cases'].values[0]
                        rsum[1] += c[c['type'] == 'death']['cases'].values[0]
                        res.append({'Country': country, 'Province': province, 'Lat': c.iloc[0]['Lat'],
                                    'Long': c.iloc[0]['Long'], 'date': date,
                                    'confirmed': c[c['type'] == 'confirmed']['cases'].values[0],
                                    'death': c[c['type'] == 'death']['cases'].values[0],
                                    'cumul confirmed': rsum[0], 'cumul death': rsum[1]
                                    })
        else:
            rsum = np.zeros(2, dtype=int)  # running sum of the 3 types
            i = 1
            for date in tdf['date'].unique():
                c = tdf[tdf['date'] == date]
                rsum[0] += c[c['type']=='confirmed']['cases'].values[0]
                rsum[1] += c[c['type']=='death']['cases'].values[0]
                res.append({'Country':country, 'Province':None, 'Lat':c.iloc[0]['Lat'], 'Long':c.iloc[0]['Long'], 'date':date,
                            'confirmed':c[c['type']=='confirmed']['cases'].values[0],
                            'death':c[c['type']=='death']['cases'].values[0],
                            'cumul confirmed':rsum[0], 'cumul death':rsum[1]
                             })
    return pnd.DataFrame(res)



def get_Xworst_Ydays(df: pnd.DataFrame, ncountries: int, days: int) -> pnd.DataFrame:
    if ncountries is None:
        ncountries = 10
    if days is None:
        days = 7
    df.date.unique()
    #ndf[ndf.date.]ndf[ndf.date.isin(focus_countries)]
    timeframe = sorted(df.date.unique())[-days:]
    tf = df[df.date.isin(timeframe)]
    #ndf.sort_values(by=['Country', 'date', 'death'])
    #ndf
    res = []

    for c in tf.Country.unique():
        #print(f"country {c} sum {ndf[ndf.Country.isin([c])].cumsum('death')}")
        # print(f"country {c} df {tf[tf.Country.isin([c])]['death'].sum()}")
        res.append({'Country': c ,'death_total': tf[tf.Country.isin([c])]['death'].sum()})

    res.sort(key=lambda k: k['death_total'])
    res
    max_countries = [k['Country'] for k in res[-ncountries:]]
    #print(f"worst countries: {max_countries} ")
    return pnd.DataFrame(res[-ncountries:])


def print_list_provinces(df: pnd.DataFrame):
    '''Print the provinces for each Country which has provinces listed.
    It can be useful as some countries have provinces that are really not mainland (eg France, UK, Denmark),
    while some other countries have provinces which are part of mainland (eg Canada, China, Australia).'''
    for c in get_countryWith_provinces(ndf):
        print('------------------------')
        print(f"{c}: ")
        [print(f"   - {prov}") for prov in list_provinces(get_country(ndf, c))]
    return True





def crosscorr(country: str, normalized: True, ndf: pnd.DataFrame):
    '''plot the 10 'best' pairwise cross correlations with the 
    country specified on the mode type provided: ['confirmed', 'death', 'recovered']'''
    cc_recovered = {}
    cc_confirmed = {}
    cc_death = {}
    
    if normalized:
        for cnty in ndf.Country.unique():
            if country != cnty:
                cc_confirmed[cnty] =  spsig.correlate(ndf.query("@country in Country").confirmed / ndf.query("@country in Country").confirmed.max(),
                                                      ndf.query("@cnty in Country").confirmed / ndf.query("@cnty in Country").confirmed.max())    
                cc_recovered[cnty] =  spsig.correlate(ndf.query("@country in Country").recovered / ndf.query("@country in Country").recovered.max(),
                                                      ndf.query("@cnty in Country").recovered / ndf.query("@cnty in Country").recovered.max())    
                cc_death[cnty] =  spsig.correlate(ndf.query("@country in Country").death / ndf.query("@country in Country").death.max(),
                                                  ndf.query("@cnty in Country").death / ndf.query("@cnty in Country").death.max())    
 
    
    else: 
        for i, cnty in enumerate(df.Country.unique()):
            if country != cnty:
                cc_confirmed[cnty] =  spsig.correlate(ndf.query("@country in Country").confirmed, ndf.query("@cnty in Country").confirmed)    
                cc_recovered[cnty] =  spsig.correlate(ndf.query("@country in Country").recovered, ndf.query("@cnty in Country").recovered)    
                cc_death[cnty] =  spsig.correlate(ndf.query("@country in Country").death, ndf.query("@cnty in Country").death)    

    cc_recovered = pnd.DataFrame(cc_recovered)
    cc_confirmed = pnd.DataFrame(cc_confirmed)
    cc_death = pnd.DataFrame(cc_death)
    return (cc_confirmed, cc_death, cc_recovered)



## PLOTTING


def timeline_global(df: pnd.DataFrame, cumul=False, kind='bar'):
    assert 'cumul confirmed' in df.columns, f"need a transformed dataset"
    res = []
    conf = 0
    death = 0
    for day in df.date.unique():
        conf += df.query("@day in date").confirmed.sum()
        death += df.query("@day in date").death.sum()
        res.append({'date': day, 'confirmed': df.query("@day in date").confirmed.sum(), 
                    'death': df.query("@day in date").death.sum(),
                    'cumul confirmed': conf, 'cumul death': death})    
    res = pnd.DataFrame(res)
    if cumul:
        res.plot(x='date', y=['confirmed', 'death', 'cumul confirmed', 'cumul death'], kind=kind)
    else:    
        res.plot(x='date', y=['confirmed', 'death'], kind=kind)


def plot_3dimensions(df: pnd.DataFrame, country_list=None):
    if country_list is None:
        country_list = df.Country.unique()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for country in country_list:
        ax.scatter3D(df[df['Country'] == country].iloc[-1]['cumul confirmed'],
                     df[df['Country'] == country].iloc[-1]['cumul death'],
                     df[df['Country'] == country].iloc[-1]['cumul recovered'])
    return ax


def plot_cumul_death_confirmed(df: pnd.DataFrame, country_list=None):
    if country_list is None:
        country_list = df.Country.unique()
    # fig = go.Figure()
    # for country in country_list:
    #     fig.add_trace(go.Scatter(x=df[df['Country'] == country].iloc[-1]['cumul confirmed'],
    #                              y=df[df['Country'] == country].iloc[-1]['cumul death'],
    #                              mode='markers', name=country))

    fig = px.scatter(df.query('@country_list in Country and @df.date.max() in date'))
                     #x="cumul confirmed", y="cumul death",)
                     #color="Country",
                     #size='cumul recovered', hover_data=['Country'])
    return fig
