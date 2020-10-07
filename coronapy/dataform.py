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
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go
from plotly.colors import n_colors

import argparse
import os
import typing

import ptitprince as pt

from sklearn.linear_model import LinearRegression

import scipy.signal as spsig
import scipy.stats as stats
import itertools

import plotly.io as pio
pio.renderers.default = "browser"


kwargs = {'xsize': 10, 'ysize': 6, 'palette': 'Set2', 'width_viol': .8,  # width of the pt rainplots
          'alpha': .6,      # transparency value for pt plots
          'orient': 'h'    # orientation of pt plots, either 'h' or 'v' ?
          }


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
        # print(f"confirmed!!! entry {entry} cumul {rsum[0]}")

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
        if has_province(df[df['country'] == country]):
            for province in get_provinces(df[df['country'] == country]):
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


def transform_metro(df: pnd.DataFrame) -> pnd.DataFrame:
    '''Reorganize data with cumul columns and specific case types columns: recovered, death, confirmed.
    Also, for Australia, Canada and China, we sum the data from provinces and return one entry for the country.
    For Denmark, France, Netherlands and the UK, we here discard the non mainland territories. For a different
    approach, please see transform_province(). Also, it puts the 'recovered' data from the Canada entries with
    the cumulative data from the provinces.'''
    #ndf = pnd.DataFrame(columns=['Country', 'Province', 'lat', 'lon', 'date', 'confirmed', 'death', 'recovered'])
    res = []
    error = []
    for country in tqdm(df['country'].unique().tolist()):
        tdf = df[df['country'] == country].fillna(country)
        # Canada does not have 'recovered' data, special case below.
        if country != 'Canada':
            if len(tdf['province'].unique()) > 1:
                if country in ['China', 'Australia']:
                    rsum = np.zeros(3, dtype=int)  # running sum of the 3 types
                    for date in tdf['date'].unique():
                        dsum = np.zeros(3, dtype=int)
                        for province in tdf['province'].unique().tolist():
                            ttdf = tdf[tdf['province'] == province]
                            if province != country and 'Princess' not in province:
                                c = ttdf[ttdf['date'] == date]
                                # dsum[0] += c[c['type'] == 'confirmed']['cases'].values[0]
                                # dsum[1] += c[c['type'] == 'death']['cases'].values[0]
                                # dsum[2] += c[c['type'] == 'recovered']['cases'].values[0]

                                try:
                                    confirmed = c[c['type'] == 'confirmed']['cases'].values[0]
                                except IndexError:
                                    error.append(f"missing data for {country} date: {date} type: confirmed, assuming 0")
                                    confirmed = 0
                                try:
                                    death = c[c['type'] == 'death']['cases'].values[0]
                                except IndexError:
                                    error.append(f"missing data for {country} date: {date} type: death assuming 0")
                                    death = 0
                                try:
                                    recovered = c[c['type'] == 'recovered']['cases'].values[0]
                                except IndexError:
                                    error.append(f"missing data for {country} date: {date} type: recovered, assuming 0")
                                    recovered = 0

                                dsum[0] += confirmed
                                dsum[1] += death
                                dsum[2] += recovered

                        rsum[0] += dsum[0]
                        rsum[1] += dsum[1]
                        rsum[2] += dsum[2]
                        res.append({'Country': country, 'Province': 'mainland', 'lat': c.iloc[0]['lat'], 'long': c.iloc[0]['long'], 'date': date,
                                    'confirmed': dsum[0],
                                    'death': dsum[1],
                                    'recovered': dsum[2],
                                    'cumul confirmed': rsum[0], 'cumul death': rsum[1], 'cumul recovered': rsum[2]
                                    })

                elif country in ['France', 'Denmark', 'Netherlands', 'United Kingdom']:
                    rsum = np.zeros(3, dtype=int)  # running sum of the 3 types
                    for date in tdf['date'].unique():
                        dsum = np.zeros(3, dtype=int)
                        ttdf = tdf[tdf['province'] == country]
                        c = ttdf[ttdf['date'] == date]
                        try:
                            confirmed = c[c['type'] == 'confirmed']['cases'].values[0]
                        except IndexError:
                            error.append(f"missing data for {country} date: {date} type: confirmed, assuming 0")
                            confirmed = 0
                        try:
                            death = c[c['type'] == 'death']['cases'].values[0]
                        except IndexError:
                            error.append(f"missing data for {country} date: {date} type: death assuming 0")
                            death = 0
                        try:
                            recovered = c[c['type'] == 'recovered']['cases'].values[0]
                        except IndexError:
                            error.append(f"missing data for {country} date: {date} type: recovered, assuming 0")
                            recovered = 0

                        rsum[0] += confirmed
                        rsum[1] += death
                        rsum[2] += recovered
                        res.append({'Country': country, 'Province': 'mainland', 'lat': c.iloc[0]['lat'], 'long': c.iloc[0]['long'], 'date': date,
                                    'confirmed': confirmed,
                                    'death': death,
                                    'recovered': recovered,
                                    'cumul confirmed': rsum[0], 'cumul death': rsum[1], 'cumul recovered': rsum[2]
                                    })
            else:
                rsum = np.zeros(3, dtype=int)  # running sum of the 3 types
                for date in tdf['date'].unique():
                    c = tdf[tdf['date'] == date]
                    try:
                        confirmed = c[c['type'] == 'confirmed']['cases'].values[0]
                    except IndexError:
                        error.append(f"missing data for {country} date: {date} type: confirmed, assuming 0")
                        confirmed = 0
                    try:
                        death = c[c['type'] == 'death']['cases'].values[0]
                    except IndexError:
                        error.append(f"missing data for {country} date: {date} type: death assuming 0")
                        death = 0
                    try:
                        recovered = c[c['type'] == 'recovered']['cases'].values[0]
                    except IndexError:
                        error.append(f"missing data for {country} date: {date} type: recovered, assuming 0")
                        recovered = 0

                    rsum[0] += confirmed
                    rsum[1] += death
                    rsum[2] += recovered

                    res.append({'Country': country, 'Province': None, 'lat': c.iloc[0]['lat'], 'long': c.iloc[0]['long'], 'date': date,
                                'confirmed': confirmed,
                                'death': death,
                                'recovered': recovered,
                                'cumul confirmed': rsum[0], 'cumul death': rsum[1], 'cumul recovered': rsum[2]
                                })
        else:  # Canada specific
            drecov = tdf[tdf['province'] == 'Canada']
            rsum = np.zeros(3, dtype=int)  # running sum of the 3 types
            for date in tdf['date'].unique():
                dsum = np.zeros(3, dtype=int)
                for province in tdf['province'].unique().tolist():
                    if province != country and 'Princess' not in province:
                        ttdf = tdf[tdf['province'] == province]
                        c = ttdf[ttdf['date'] == date]
                        try:
                            confirmed = c[c['type'] == 'confirmed']['cases'].values[0]
                        except IndexError:
                            error.append(f"missing data for {country} date: {date} type: confirmed, assuming 0")
                            confirmed = 0
                        try:
                            death = c[c['type'] == 'death']['cases'].values[0]
                        except IndexError:
                            error.append(f"missing data for {country} date: {date} type: death assuming 0")
                            death = 0
                        try:
                            recovered = drecov[drecov['type'] == 'recovered']['cases'].values[0]
                        except IndexError:
                            error.append(f"missing data for {country} date: {date} type: recovered, assuming 0")
                            recovered = 0

                        dsum[0] += confirmed
                        dsum[1] += death
                        dsum[2] += recovered
                rsum[0] += dsum[0]
                rsum[1] += dsum[1]
                rsum[2] += dsum[2]
                res.append({'Country': country, 'Province': 'mainland', 'lat': c.iloc[0]['lat'], 'long': c.iloc[0]['long'], 'date': date,
                            'confirmed': dsum[0],
                            'death': dsum[1],
                            'recovered': dsum[2],
                            'cumul confirmed': rsum[0], 'cumul death': rsum[1], 'cumul recovered': rsum[2]
                            })
    print(error)
    return pnd.DataFrame(res), error


def transform(df: pnd.DataFrame) -> pnd.DataFrame:
    #ndf = pnd.DataFrame(columns=['Country', 'Province', 'lat', 'lon', 'date', 'confirmed', 'death', 'recovered'])
    res = []
    for country in df['country'].unique().tolist():
        tdf = df[df['country'] == country].fillna(country)
        # Canada does not have 'recovered' data, special case below.
        if country != 'Canada':
            if len(tdf['province'].unique()) > 1:
                for province in tdf['province'].unique().tolist():
                    ttdf = tdf[tdf['province'] == province]
                    rsum = np.zeros(3, dtype=int)  # running sum of the 3 types
                    i = 1
                    for date in ttdf['date'].unique():
                        c = ttdf[ttdf['date'] == date]
                        rsum[0] += c[c['type'] == 'confirmed']['cases'].values[0]
                        rsum[1] += c[c['type'] == 'death']['cases'].values[0]
                        rsum[2] += c[c['type'] == 'recovered']['cases'].values[0]
                        res.append({'Country': country, 'Province': province, 'lat': c.iloc[0]['lat'], 'long': c.iloc[0]['long'], 'date': date,
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
                    rsum[0] += c[c['type'] == 'confirmed']['cases'].values[0]
                    rsum[1] += c[c['type'] == 'death']['cases'].values[0]
                    rsum[2] += c[c['type'] == 'recovered']['cases'].values[0]
                    res.append({'Country': country, 'Province': None, 'lat': c.iloc[0]['lat'], 'long': c.iloc[0]['long'], 'date': date,
                                'confirmed': c[c['type'] == 'confirmed']['cases'].values[0],
                                'death': c[c['type'] == 'death']['cases'].values[0],
                                'recovered': c[c['type'] == 'recovered']['cases'].values[0],
                                'cumul confirmed': rsum[0], 'cumul death': rsum[1], 'cumul recovered': rsum[2]
                                })
        else:
            for province in tdf['province'].unique().tolist():
                if province != 'Canada':
                    ttdf = tdf[tdf['province'] == province]
                    # running sum of the 2 types for 'Canada'
                    rsum = np.zeros(2, dtype=int)
                    i = 1
                    for date in ttdf['date'].unique():
                        c = ttdf[ttdf['date'] == date]
                        rsum[0] += c[c['type'] == 'confirmed']['cases'].values[0]
                        rsum[1] += c[c['type'] == 'death']['cases'].values[0]
                        res.append({'Country': country, 'Province': province, 'lat': c.iloc[0]['lat'], 'long': c.iloc[0]['long'], 'date': date,
                                    'confirmed': c[c['type'] == 'confirmed']['cases'].values[0],
                                    'death': c[c['type'] == 'death']['cases'].values[0],
                                    'cumul confirmed': rsum[0], 'cumul death': rsum[1],
                                    'cumul recovered': None, 'recovered': None
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


def test_raw_df(df: pnd.DataFrame) -> dict:
    res = {'recovered': True, 'confirmed': True,
           'death': True, 'date': True, 'positive': True}
    for country in df['country'].unique():
        c = df[df['country'] == country]
        if len(c['province'].unique()) > 1:
            for province in c['province'].unique():
                if 'recovered' not in c[c['province'] == province].type.tolist():
                    res['recovered'] = False
                    print(f"{country} {province} does not have recovered entries")
                if 'confirmed' not in c[c['province'] == province].type.tolist():
                    res['confirmed'] = False
                    print(f"{country} {province} does not have confirmed entries")
                if 'death' not in c[c['province'] == province].type.tolist():
                    res['death'] = False
                    print(f"{country} {province} does not have death entries")
                elif np.min(c[c['province'] == province].query("'death' in type").cases) < 0:
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


def test_transformed_df(df: pnd.DataFrame) -> dict:
    res = {'recovered': True, 'confirmed': True,
           'death': True, 'date': True, 'positive': True}
    for country in df['country'].unique():
        c = df[df['country'] == country]
        for province in c['province'].unique():
            if 'recovered' not in c[c['province'] == province].type:
                res['recovered'] = False
                print(f"{country} {province} does not have recovered entries")
            if 'confirmed' not in c[c['province'] == province].type:
                res['confirmed'] = False
                print(f"{country} {province} does not have confirmed entries")
            if 'death' not in c[c['province'] == province].type:
                res['death'] = False
                print(f"{country} {province} does not have death entries")

            if np.min(c[c['province'] == province].cases) < 0:
                print(f"{country} {province} has a negative value case")
    return res


def transform_debug(df: pnd.DataFrame) -> pnd.DataFrame:
    #ndf = pnd.DataFrame(columns=['Country', 'Province', 'lat', 'lon', 'date', 'confirmed', 'death', 'recovered'])
    res = []
    #df = df.fillna('Metro')

    #df = df.drop(df['province']=='Recovered')
    #df = remove_Princess(df)
    for country in df['country'].unique().tolist():
        print(country)
        tdf = df[df['country'] == country].fillna(country)
        if len(tdf['province'].unique()) > 1:
            for province in tdf['province'].unique().tolist():
                # Canada has provinces, and a generic entry with only recovered cases.
                if province != 'Canada':
                    print(province)
                    ttdf = tdf[tdf['province'] == province]
                    rsum = np.zeros(2, dtype=int)  # running sum of the 3 types
                    i = 1
                    for date in ttdf['date'].unique():
                        c = ttdf[ttdf['date'] == date]
                        print('--------')
                        print(c[c['type'] == 'confirmed']['cases'].values)
                        rsum[0] += c[c['type'] == 'confirmed']['cases'].values[0]
                        rsum[1] += c[c['type'] == 'death']['cases'].values[0]
                        res.append({'Country': country, 'Province': province, 'lat': c.iloc[0]['lat'],
                                    'long': c.iloc[0]['long'], 'date': date,
                                    'confirmed': c[c['type'] == 'confirmed']['cases'].values[0],
                                    'death': c[c['type'] == 'death']['cases'].values[0],
                                    'cumul confirmed': rsum[0], 'cumul death': rsum[1]
                                    })
        else:
            rsum = np.zeros(2, dtype=int)  # running sum of the 3 types
            i = 1
            for date in tdf['date'].unique():
                c = tdf[tdf['date'] == date]
                rsum[0] += c[c['type'] == 'confirmed']['cases'].values[0]
                rsum[1] += c[c['type'] == 'death']['cases'].values[0]
                res.append({'Country': country, 'Province': None, 'lat': c.iloc[0]['lat'], 'long': c.iloc[0]['long'], 'date': date,
                            'confirmed': c[c['type'] == 'confirmed']['cases'].values[0],
                            'death': c[c['type'] == 'death']['cases'].values[0],
                            'cumul confirmed': rsum[0], 'cumul death': rsum[1]
                            })
    return pnd.DataFrame(res)


def get_Xworst_Ydays(df: pnd.DataFrame, ncountries: int, days: int) -> pnd.DataFrame:
    if ncountries is None:
        ncountries = 10
    if days is None:
        days = 7
    df.date.unique()
    # ndf[ndf.date.]ndf[ndf.date.isin(focus_countries)]
    timeframe = sorted(df.date.unique())[-days:]
    tf = df[df.date.isin(timeframe)]
    #ndf.sort_values(by=['Country', 'date', 'death'])
    # ndf
    res = []

    for c in tf.Country.unique():
        # print(f"country {c} sum {ndf[ndf.Country.isin([c])].cumsum('death')}")
        # print(f"country {c} df {tf[tf.Country.isin([c])]['death'].sum()}")
        res.append(
            {'Country': c, 'death_total': tf[tf.Country.isin([c])]['death'].sum()})

    res.sort(key=lambda k: k['death_total'])
    res
    max_countries = [k['Country'] for k in res[-ncountries:]]
    # print(f"worst countries: {max_countries} ")
    return pnd.DataFrame(res[-ncountries:])



def get_countries_over_threshold(df: pnd.DataFrame, undf: pnd.DataFrame, threshold: int = 20/100000., window: int = 14):
    """
    Returns a list of the countries with a ratio new case per 100000 persons is over a threshold,
    based on the previous window days.
    :params df: a pandas dataframe containing daily country data (cases)
    :params undf: a pandas dataframe containing the population of the countries
    :params threshold: an integer specifying the selection for the ratio cases / 100000
    :params window: number of past days taken into account for the analysis
    :return: pandas dataframe of countries and their values 
    """
    pass



def print_list_provinces(df: pnd.DataFrame):
    '''Print the provinces for each Country which has provinces listed.
    It can be useful as some countries have provinces that are really not mainland (eg France, UK, Denmark),
    while some other countries have provinces which are part of mainland (eg Canada, China, Australia).'''
    for c in get_countryWith_provinces(ndf):
        print('------------------------')
        print(f"{c}: ")
        [print(f"   - {prov}") for prov in list_provinces(get_country(ndf, c))]
    return True



def get_latest_deaths_cumul(df: pnd.DataFrame, country=None):
    res = []
    if type(country) == str:
        country = [country]
    if country is None:
        country = df.Country.unique()
    for c in country:
        res.append({'country': c, 'deaths':df.query("@c in Country")['death'][-1],
                    'cases':df.query("@c in Country")['confirmed'][-1]})
    return pnd.DataFrame(res)



def get_latest_DeathsCases_days(df: pnd.DataFrame, country=None, days: int = 14):
    res = []
    if type(country) == str:
        country = [country]
    if country is None:
        country = df.Country.unique()
    for c in country:
        res.append({'country': c, 'deaths':sum(df.query("@c in Country")['death'][-days:]),
                    'cases':sum(df.query("@c in Country")['confirmed'][-days:])})
    return pnd.DataFrame(res)


def get_latest_DeathsCases_days_per100000(df: pnd.DataFrame, undf: pnd.DataFrame, country = None, days: int = 14):
    res = []
    if type(country) == str:
        country = [country]
    if country is None:
        country = df.Country.unique()
    for c in country:
        res.append({'country': c,
                    'deaths':sum(df.query("@c in Country")['death'][-days:]),
                    'cases':sum(df.query("@c in Country")['confirmed'][-days:]),
                    'deaths_per100000':sum(df.query("@c in Country")['death'][-days:]) / (get_latest_data(undf, c, list_series(undf)[0]) * 10),
                    'cases_per100000':sum(df.query("@c in Country")['confirmed'][-days:]) / (get_latest_data(undf, c, list_series(undf)[0]) * 10)
                    })
    return pnd.DataFrame(res)


def get_sliding_window_per100000(df: pnd.DataFrame, undf: pnd.DataFrame, country = None, days: int = 14):
    res = []
    if type(country) == str:
        country = [country]
    if country is None:
        country = df.Country.unique()
    for c in tqdm(country):
        country_data = df.query("@c in Country")
        for dx in range(days, len(country_data)):

            res.append({'country': c, 'date': country_data.iloc[dx]['date'],
                        'deaths':sum(country_data.iloc[dx-days + 1:dx + 1]['death']),
                        'cases':sum(country_data.iloc[dx-days + 1:dx + 1]['confirmed']),
                        'deaths_per100000':sum(country_data.iloc[dx-days + 1:dx + 1]['death']) / (get_latest_data(undf, c, list_series(undf)[0]) * 10),
                        'cases_per100000':sum(country_data.iloc[dx-days + 1:dx + 1]['confirmed']) / (get_latest_data(undf, c, list_series(undf)[0]) * 10)
                        })
    return pnd.DataFrame(res)





def plot_sliding_per100000(df: pnd.DataFrame, days: int = 14, **kwargs):
    fig = px.line(df, x='date', y='cases_per100000', color='country')
    fig.update_layout(title=f'New cases per 100 000 per country per sliding {days} days' ,
                      xaxis_title="date",
                      yaxis_title="new cases per 100 000 inhabitants")
    fig.show()
    return None


def pred_basic_peak_now(df: pnd.DataFrame, country = None, day: int = 5):
    """
    Return the predicted below 20/100000 new cases per 14 days if the peak is today, only supposing the time from
    under 20 / 1000000 to peak will be the same as from peak back to under 20/100000.
    """
    today = date.today()
    res = []
    if type(country) == str:
        country = [country]
    if country is None:
        country = df.Country.unique()
    for c in tqdm(country):
        cdf = df.query("@c in country")
        print(f"country: {c}")
        # compute peak
        if LinearRegression().fit(np.arange(day).reshape((-1, 1)), cdf['cases_per100000'].iloc[-day:]).coef_ > 0:
            trend = 'up'
            peak_day = datetime.today()
        else:
            trend = 'down'
            delay = 0
            while LinearRegression().fit(np.arange(day).reshape((-1, 1)), cdf['cases_per100000'].iloc[-(day+delay):len(cdf)-delay]).coef_ < 0:
                day+=1
                peak_day = cdf['date'].iloc[-day]
            peak_day = datetime.strptime(peak_day, '%Y-%m-%d')

        res.append({'country': c, 'last day below threshold': cdf[cdf['cases_per100000']<20.]['date'].iloc[-1] if cdf[cdf['cases_per100000']<20.]['date'].iloc[-1] != cdf['date'].iloc[-1] else None,
                    'trend': trend,
                    'half_peak_span': (peak_day - datetime.strptime(cdf[cdf['cases_per100000']<20.]['date'].iloc[-1], '%Y-%m-%d')).days if cdf[cdf['cases_per100000']<20.]['date'].iloc[-1] != cdf['date'].iloc[-1] else None,
                    'predicted end': str((peak_day + timedelta(days=peak_day - datetime.strptime(cdf[cdf['cases_per100000']<20.]['date'].iloc[-1], '%Y-%m-%d'))).days.date()) if cdf[cdf['cases_per100000']<20.]['date'].iloc[-1] != cdf['date'].iloc[-1] else None})

    return pnd.DataFrame(res)
        



# ## UN dataset

def get_latest_data(undf, country, serie):
    # returns the latest available value for the specified country and serie name.
    return float(undf.query("@country in countries and @serie in series").sort_values('years', ascending=True).iloc[-1].numerics)


def list_series(undf):
    # returns a list of the available series name
    return undf.series.unique().tolist()


def get_random_qualitative_color_map(
        categories:list,
        colors: typing.List[str] = px.colors.cyclical.mygbm
        ) -> typing.List[str]:
    """
    Returns a color coding for a given series (one color for every unique value). Will repeat colors if not enough are
    provided.
    For some color scales see: https://plotly.com/python/builtin-colorscales/
    Some options are: Phase, Edge, mygbm, mrybm, HSV, IceFire.
    :param categories: A series of categorial data
    :param colors: color codes (everything plotly accepts)
    :return: Array of colors matching the index of the objects
    """
    # get unique identifiers
    unique_series = categories

    # create lookup table - colors will be repeated if not enough
    color_lookup_table = dict((value, color) for (value, color) in zip(unique_series, itertools.cycle(colors)))

    # look up the colors in the table
    return [color_lookup_table[key] for key in categories]




def plot_overview_list(df, undf, country_list):
    series = list_series(undf)
    fig = make_subplots(rows=len(series), cols=1,
                        subplot_titles=series)
    color_rgb = get_random_qualitative_color_map(country_list)
    for sx, serie in enumerate(series):
        '''for c in country_list:
            cur = df.query("@c in Country")
            latest = get_latest_data(undf, c, serie)'''
        fig.add_trace(go.Scatter(x=[df.query("@c in Country").iloc[-1]['cumul death'] / get_latest_data(undf, c, serie)
                                    for c in country_list],
                                 y=[df.query("@c in Country").iloc[-1]['cumul confirmed'] / get_latest_data(undf, c, serie)
                                    for c in country_list],
                                 text=[c for c in country_list], # hover_data=[['A', 'B'] for c in country_list],
                                 customdata=[f"{get_latest_data(undf, c, serie)} {serie}" for c in country_list],
                                 hovertemplate="%{text}<br>%{customdata}<br><br>cases: %{y}<br>deaths: %{x} ",
                                 mode='markers', marker=dict(color=color_rgb), name=None),
                      row=sx + 1, col=1,)
    

    for sx, serie in enumerate(series):
        fig.update_xaxes(title_text=f"Deaths / {serie}", row=sx + 1, col=1)
        fig.update_yaxes(title_text=f"Cases / {serie}", row=sx + 1, col=1)
    fig.update_layout(height=5000, width=1000,
                  title_text="Death / Cases raported to various metrics (Population, Density, Surface, ...)")
    fig.show()

    return None



def plot_overview_list_beta(df, undf, country_list):
    series = list_series(undf)
    fig = make_subplots(rows=len(series), cols=1,
                        subplot_titles=series)
    color_rgb = get_random_qualitative_color_map(country_list)
    for sx, serie in enumerate(series):
        '''for c in country_list:
            cur = df.query("@c in Country")
            latest = get_latest_data(undf, c, serie)'''
        #for cx, c in enumerate(country_list):
        fig.add_trace([go.Scatter(x=df.query("@c in Country").iloc[-1]['cumul death'] / get_latest_data(undf, c, serie),
                                 y=df.query("@c in Country").iloc[-1]['cumul confirmed'] / get_latest_data(undf, c, serie),
                                 text=c, # hover_data=[['A', 'B'] for c in country_list],
                                 hovertemplate="%{text}<br><br>cases: %{y}<br>deaths: %{x} ",
                                 mode='markers', marker=dict(color=color_rgb[cx]), name=c) for cx, c in enumerate(country_list)],
                      row=sx + 1, col=1)
    

    for sx, serie in enumerate(series):
        fig.update_xaxes(title_text=f"Deaths / {serie}", row=sx + 1, col=1)
        fig.update_yaxes(title_text=f"Cases / {serie}", row=sx + 1, col=1)
    fig.update_layout(height=5000, width=1000,
                  title_text="Death / Cases for various metrics")
    fig.show()

    return None





def plot_overview_bubbles(df, undf, country_list):
    series = list_series(undf)
    fig = make_subplots(rows=len(series), cols=1,
                        subplot_titles=series)
    color_rgb = get_random_qualitative_color_map(country_list)
    for sx, serie in enumerate(series):
        '''for c in country_list:
            cur = df.query("@c in Country")
            latest = get_latest_data(undf, c, serie)'''
        fig.add_trace(go.Scatter(x=[df.query("@c in Country").iloc[-1]['cumul death']
                                    for c in country_list],
                                 y=[df.query("@c in Country").iloc[-1]['cumul confirmed']
                                    for c in country_list],
                                 text=[c for c in country_list], # hover_data=[['A', 'B'] for c in country_list],
                                 customdata=[f"{get_latest_data(undf, c, serie)} {serie}" for c in country_list],
                                 hovertemplate="%{text}<br>%{customdata}<br><br>cases: %{y}<br>deaths: %{x} ",
                                 mode='markers', marker=dict(color=color_rgb),
                                 marker_size=[get_latest_data(undf, c, serie)  for c in country_list], name=None),
                      row=sx + 1, col=1,)
    

    for sx, serie in enumerate(series):
        fig.update_xaxes(title_text=f"Deaths (absolute)", row=sx + 1, col=1)
        fig.update_yaxes(title_text=f"Cases (absolute)", row=sx + 1, col=1)
    fig.update_layout(height=5000, width=1000,
                  title_text="Death / Cases for selected countries, bubble size wrt to different metrics")
    fig.show()

    return None



def rainplot_full(df, undf, countries,
                  xsize=10, ysize=7,
                  palette='Set2', bw=.2, width_viol=.8, orient='h', alpha=.6, **kwargs):

    colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', len(countries), colortype='rgb')
    fig = go.Figure()
    df = df.query("@countries in Country")
    updated_df = pnd.DataFrame()
    series = list_series(undf)
    for cx, country in enumerate(tqdm(countries)):
        
        # if args.violin:
        #     sns.catplot(x='Dataset', y='Age', hue='Sex', kind='violin', scale='count', legend=False, data=ddf).set(title=f"{dx} n={len(ddf)}", ylim=[0, agemax])
        #     plt.legend(loc='upper left')
        #     plt.tight_layout()
        # else:
        tdf = df.query('@country in Country')
        tdf['percapita_death'] = tdf['death'] / get_latest_data(undf, country, series[0])
        tdf['percapita_cases'] = tdf['confirmed'] / get_latest_data(undf, country, series[0])
        tdf['density_death'] = tdf['death'] / get_latest_data(undf, country, series[6])
        tdf['density_cases'] = tdf['confirmed'] / get_latest_data(undf, country, series[6])
        tdf['over60_death'] = tdf['death'] / get_latest_data(undf, country, series[5])
        tdf['over60_cases'] = tdf['confirmed'] / get_latest_data(undf, country, series[5])
        tdf['surface_death'] = tdf['death'] / get_latest_data(undf, country, series[7])
        tdf['surface_cases'] = tdf['confirmed'] / get_latest_data(undf, country, series[7])


        updated_df.append(tdf)
        #f, ax = plt.subplots(figsize=(xsize, ysize))

        #ax = pt.RainCloud(x='date', y='percapita_death', data=tdf, palette=palette, bw=.2,
        #                  width_viol=width_viol, ax=ax, orient=orient, alpha=alpha, dodge=True, pointplot=False)
        #ax.set_xlim([0, agemax])
        # handles, labels = ax.get_legend_handles_labels()
        # f.legend(handles[0:ddf.Sex.nunique()], labels[0:ddf.Sex.nunique()], loc='upper right',
        #          bbox_to_anchor=(1.15, 1), borderaxespad=0.)
        # ax.legend(loc='lower right', shadow=True, ncol=1)
        # ax.set_title(f"{dx} n={len(across_dx(ldataset, dx))}")
        #plt.title(f"{country}")
        #handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles[0:ddf.Sex.nunique()], labels[0:ddf.Sex.nunique()], loc='upper right')
        #plt.tight_layout()
        #plt.show()

        fig.add_trace(go.Violin(x=tdf.percapita_death, line_color=colors[cx], name=f"{country}"))
        fig.update_traces(orientation='h', side='positive', width=3, points=False)
    fig.update_xaxes(title_text='date')
    fig.update_yaxes(title_text='country')
    fig.update_layout(xaxis_showgrid=True, xaxis_zeroline=False, title='daily death per capita')
    fig.show()

    return updated_df



def restricted_undf_per_capita(df, undf, focus):
    df = df.query("@focus in Country")
    return none


def ranked_per_capita():
    return None


def test_per_capita(df, undf, country_list):
    for serie in list_series(undf):
        print(f"{serie} *************")
        for c in country_list:
            print(c)
            cur = df.query("@c in Country")
            latest = get_latest_data(undf, c, serie)
            print(f"    death {round(cur.iloc[-1]['cumul death'] / latest, 2)}\
                    cases {round(cur.iloc[-1]['cumul confirmed'] / latest, 2)}")


def data_per_capita(df: pnd.DataFrame, undf: pnd.DataFrame, country):
    # res = []
    # for c in country:
    #     c.append({'country':c, 'cases': , 'deaths':})
    pass




'''
fig = px.bar([mini.query("@m in Country").iloc[-1] for m in mini.Country.unique()], x="Country", y="cumul death", 
                  color='Country', 
                  height=500) 
     fig.show() 
'''




def crosscorr(country: str, normalized: True, ndf: pnd.DataFrame):
    '''plot the 10 'best' pairwise cross correlations with the 
    country specified on the mode type provided: ['confirmed', 'death', 'recovered']'''
    cc_recovered = {}
    cc_confirmed = {}
    cc_death = {}

    if normalized:
        for cnty in ndf.Country.unique():
            if country != cnty:
                cc_confirmed[cnty] = spsig.correlate(ndf.query("@country in Country").confirmed / ndf.query("@country in Country").confirmed.max(),
                                                     ndf.query("@cnty in Country").confirmed / ndf.query("@cnty in Country").confirmed.max())
                cc_recovered[cnty] = spsig.correlate(ndf.query("@country in Country").recovered / ndf.query("@country in Country").recovered.max(),
                                                     ndf.query("@cnty in Country").recovered / ndf.query("@cnty in Country").recovered.max())
                cc_death[cnty] = spsig.correlate(ndf.query("@country in Country").death / ndf.query("@country in Country").death.max(),
                                                 ndf.query("@cnty in Country").death / ndf.query("@cnty in Country").death.max())

    else:
        for i, cnty in enumerate(df.Country.unique()):
            if country != cnty:
                cc_confirmed[cnty] = spsig.correlate(ndf.query(
                    "@country in Country").confirmed, ndf.query("@cnty in Country").confirmed)
                cc_recovered[cnty] = spsig.correlate(ndf.query(
                    "@country in Country").recovered, ndf.query("@cnty in Country").recovered)
                cc_death[cnty] = spsig.correlate(
                    ndf.query("@country in Country").death, ndf.query("@cnty in Country").death)

    cc_recovered = pnd.DataFrame(cc_recovered)
    cc_confirmed = pnd.DataFrame(cc_confirmed)
    cc_death = pnd.DataFrame(cc_death)
    return (cc_confirmed, cc_death, cc_recovered)


# PLOTTING


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
        res.plot(x='date', y=['confirmed', 'death',
                              'cumul confirmed', 'cumul death'], kind=kind)
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

    fig = px.scatter(df.query('@country_list in Country and @df.date.max() in date'),
                     x="cumul confirmed", y="cumul death",
                     color="Country",
                     size='cumul recovered', hover_data=['Country'])
    return fig
