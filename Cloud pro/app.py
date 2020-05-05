# storing and anaysis
import numpy as np
import pandas as pd
import os

import plotly
from plotly.offline import iplot


covid_19 = pd.read_csv('./covid_19_data.csv',parse_dates=['ObservationDate'])
print (covid_19.shape)
print ('Last update: ' + str(covid_19.ObservationDate.max()))

# visualization
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt



import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, send_file, make_response
from io import BytesIO

app = Flask(__name__)
# color pallette
cdr = ['#ffffff', '#ff4040', '#1e90ff'] # grey - red - blue
idr = ['#eeee00', '#ff4040', '#1e90ff'] # yellow - red - blue
h = '#563893'
a = '#89b4ee'
n = '#bcd750'
e = '#e4bd38'
s = '#ef977b'
hanes = [h,a,n,e,s]
hns = [h, n, s]

checkdup = covid_19.groupby(['Country/Region','Province/State','ObservationDate']).count().iloc[:,0]
checkdup[checkdup>1]
covid_19 = covid_19[(covid_19.Confirmed>0) | (covid_19['Province/State'] == 'Recovered')]
covid_19 = covid_19.drop(['SNo', 'Last Update'], axis=1)
covid_19 = covid_19.rename(columns={'Country/Region': 'Country', 'ObservationDate':'Date'})
# To check null values
covid_19.isnull().sum()

# Sort data
covid_19 = covid_19.sort_values(['Date','Country','Province/State'])
# Add column of days since first case
covid_19['first_date'] = covid_19.groupby('Country')['Date'].transform('min')
covid_19['days'] = (covid_19['Date'] - covid_19['first_date']).dt.days

#Grouping different types of cases as per the date
datewise=covid_19.groupby(["Date"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
datewise["Days Since"]=datewise.index-datewise.index.min()
print("Basic Information")
print("Totol number of countries with Disease Spread: ",len(covid_19["Country"].unique()))
print("Total number of Confirmed Cases around the World: ",datewise["Confirmed"].iloc[-1])
print("Total number of Recovered Cases around the World: ",datewise["Recovered"].iloc[-1])
print("Total number of Deaths Cases around the World: ",datewise["Deaths"].iloc[-1])
print("Total number of Active Cases around the World: ",(datewise["Confirmed"].iloc[-1]-datewise["Recovered"].iloc[-1]-datewise["Deaths"].iloc[-1]))
print("Total number of Closed Cases around the World: ",datewise["Recovered"].iloc[-1]+datewise["Deaths"].iloc[-1])
print("Approximate number of Confirmed Cases per Day around the World: ",np.round(datewise["Confirmed"].iloc[-1]/datewise.shape[0]))
print("Approximate number of Recovered Cases per Day around the World: ",np.round(datewise["Recovered"].iloc[-1]/datewise.shape[0]))
print("Approximate number of Death Cases per Day around the World: ",np.round(datewise["Deaths"].iloc[-1]/datewise.shape[0]))
print("Approximate number of Confirmed Cases per hour around the World: ",np.round(datewise["Confirmed"].iloc[-1]/((datewise.shape[0])*24)))
print("Approximate number of Recovered Cases per hour around the World: ",np.round(datewise["Recovered"].iloc[-1]/((datewise.shape[0])*24)))
print("Approximate number of Death Cases per hour around the World: ",np.round(datewise["Deaths"].iloc[-1]/((datewise.shape[0])*24)))
print("Number of Confirmed Cases in last 24 hours: ",datewise["Confirmed"].iloc[-1]-datewise["Confirmed"].iloc[-2])
print("Number of Recovered Cases in last 24 hours: ",datewise["Recovered"].iloc[-1]-datewise["Recovered"].iloc[-2])
print("Number of Death Cases in last 24 hours: ",datewise["Deaths"].iloc[-1]-datewise["Deaths"].iloc[-2])

@app.route("/co19_27", methods=['GET'])
def covid_27():
    plt.figure(figsize=(15,8))
    sns.barplot(x=datewise.index.date, y=datewise["Confirmed"]-datewise["Recovered"]-datewise["Deaths"])
    plt.title("Distribution Plot for Active Cases over Date")
    plt.xticks(rotation=90)
    plt.savefig(img)
    plt.savefig('static/images/new_plot27.png')
    return render_template('images.html', name = 'Number of Active Cases Over Date', url ='static/images/new_plot27.png')

@app.route("/", methods=['GET'])
def bye():
    return render_template("Vizualize.html")

@app.route("/co19_28", methods=['GET'])
def covid_28():
    plt.figure(figsize=(15, 8))
    sns.barplot(x=datewise.index.date, y=datewise["Recovered"] + datewise["Deaths"])
    plt.title("Distribution Plot for Closed Cases over Date")
    plt.xticks(rotation=90)
    plt.savefig(img)
    plt.savefig('static/images/new_plot28.png')
    return render_template('images.html', name = 'Number of Active Cases Over Date', url ='static/images/new_plot28.png')


datewise["WeekOfYear"]=datewise.index.weekofyear
week_num=[]
weekwise_confirmed=[]
weekwise_recovered=[]
weekwise_deaths=[]
w=1
for i in list(datewise["WeekOfYear"].unique()):
       weekwise_confirmed.append(datewise[datewise["WeekOfYear"]==i]["Confirmed"].iloc[-1])
       weekwise_recovered.append(datewise[datewise["WeekOfYear"]==i]["Recovered"].iloc[-1])
       weekwise_deaths.append(datewise[datewise["WeekOfYear"]==i]["Deaths"].iloc[-1])
       week_num.append(w)
       w=w+1
       
@app.route("/co19_29", methods=['GET'])
def covid_29():
    plt.figure(figsize=(8,5))
    plt.plot(week_num,weekwise_confirmed,linewidth=3)
    plt.plot(week_num,weekwise_recovered,linewidth=3)
    plt.plot(week_num,weekwise_deaths,linewidth=3)
    plt.ylabel("Number of Cases")
    plt.xlabel("Week Number")
    plt.title("Weekly progress of Different Types of Cases")
    plt.xlabel
    plt.savefig('static/images/new_plot29.png')
    return render_template('images.html', name = 'Weekwise Analysis of Confirmed, Death and Recovered Cases', url ='static/images/new_plot29.png')




    
    
@app.route("/co19_30", methods=['GET'])
def covid_30():
    fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(15,5))
    sns.barplot(x=week_num,y=pd.Series(weekwise_confirmed).diff().fillna(0),ax=ax1)
    sns.barplot(x=week_num,y=pd.Series(weekwise_deaths).diff().fillna(0),ax=ax2)
    ax1.set_xlabel("Week Number")
    ax2.set_xlabel("Week Number")
    ax1.set_ylabel("Number of Confirmed Cases")
    ax2.set_ylabel("Number of Death Cases")
    ax1.set_title("Weekly increase in Number of Confirmed Cases")
    ax2.set_title("Weekly increase in Number of Death Cases")
    plt.savefig(img)
    plt.savefig('static/images/new_plot30.png')
    return render_template('images.html', name = 'Weekwise Aanlysis of Increase in Number of Confirmed and Death Cases', url ='static/images/new_plot30.png')





latest = covid_19[covid_19.Date == covid_19.Date.max()]
cty = latest.groupby('Country').sum()
cty['Death Rate'] = cty['Deaths'] / cty['Confirmed'] * 100
cty['Recovery Rate'] = cty['Recovered'] / cty['Confirmed'] * 100
cty['Active'] = cty['Confirmed'] - cty['Deaths'] - cty['Recovered']
cty.drop('days',axis=1).sort_values('Confirmed', ascending=False).head(10)
img = BytesIO()

def plot_new(column, title):
       _ = cty.sort_values(column, ascending=False).head(15)
       g = sns.barplot(_[column], _.index)
       plt.title(title, fontsize=14)
       plt.ylabel(None)
       plt.xlabel(None)
       plt.grid(axis='x')
       for i, v in enumerate(_[column]):
              if column == 'Death Rate':
                     g.text(v * 1.01, i + 0.1, str(round(v, 2)))
              else:
                     g.text(v * 1.01, i + 0.1, str(int(v)))

@app.route("/co19", methods=['GET'])
def covid_1():
       global covid_19
       plt.figure(figsize=(9,16))
       plt.subplot(311)
       plot_new('Confirmed','Confirmed cases top 15 countries')
       plt.subplot(312)
       plot_new('Deaths','Death cases top 15 countries')
       plt.subplot(313)
       plot_new('Active','Active cases top 15 countries')
       plt.savefig(img)
       plt.savefig('static/images/new_plot1.png')
       return render_template('images.html', name = 'Confirmed,Deaths and Active cases for Top 15 countries', url ='static/images/new_plot1.png')

def plot_rate(rank, title):
    if rank == 'highest':
        _ = cty[cty.Deaths>=10].sort_values('Death Rate', ascending=False).head(15)
    else:
        _ = cty[cty.Confirmed>=500].sort_values('Death Rate').head(15)
    g = sns.barplot(_['Death Rate'], _.index)
    plt.title(title, fontsize=14)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.grid(axis='x')
    for i, v in enumerate(_['Death Rate']):
        g.text(v*1.01, i+0.1, str(round(v,2)))

plt.figure(figsize=(9,12))
plt.subplot(211)
plot_rate('highest','Highest death rate top 15 (>=10 deaths only)')
plt.subplot(212)
plot_rate('lowest','Lowest death rate top 15 (>=500 confirmed only)')

## Evolution of Cases in the world over time

import matplotlib.dates as mdates
months_fmt = mdates.DateFormatter('%b-%e')
evo = covid_19.groupby('Date')[['Confirmed','Deaths','Recovered']].sum()
evo['Active'] = evo['Confirmed'] - evo['Deaths'] - evo['Recovered']
evo['Death Rate'] = evo['Deaths'] / evo['Confirmed'] * 100
evo['Recover Rate'] = evo['Recovered'] / evo['Confirmed'] * 100
fig, ax = plt.subplots(2, 2, figsize=(12,9))
plt.title('Evolution of cases in the world')
def plot_evo(num, col, title):
    ax[num].plot(evo[col], lw=3)
    ax[num].set_title(title)
    ax[num].xaxis.set_major_locator(plt.MaxNLocator(7))
    ax[num].xaxis.set_major_formatter(months_fmt)
    ax[num].grid(True)

@app.route("/co19_2", methods=['GET'])
def covid_2():
    plot_evo((0,0), 'Confirmed', 'Confirmed cases')
    plot_evo((0,1), 'Deaths', 'Death cases')
    plot_evo((1,0), 'Active', 'Active cases')
    plot_evo((1,1), 'Death Rate', 'Death rate')
    plt.savefig(img)
    plt.savefig('static/images/new_plot2.png')
    return render_template('images.html', name = 'Evolution of Cases in the world over time', url ='static/images/new_plot2.png')


def plot_cty(num, evo_col, title):
    ax[num].plot(evo_col, lw=3)
    ax[num].set_title(title)
    ax[num].xaxis.set_major_locator(plt.MaxNLocator(7))
    ax[num].xaxis.set_major_formatter(months_fmt)
    ax[num].grid(True)

def evo_cty(country):
        global covid_19
        evo_cty = covid_19[covid_19.Country==country].groupby('Date')[['Confirmed','Deaths','Recovered']].sum()
        evo_cty['Active'] = evo_cty['Confirmed'] - evo_cty['Deaths'] - evo_cty['Recovered']
        evo_cty['Death Rate'] = evo_cty['Deaths'] / evo_cty['Confirmed'] * 100
        plot_cty((0,0), evo_cty['Confirmed'], 'Confirmed cases')
        plot_cty((0,1), evo_cty['Deaths'], 'Death cases')
        plot_cty((1,0), evo_cty['Active'], 'Active cases')
        plot_cty((1,1), evo_cty['Death Rate'], 'Death rate')
        fig.suptitle(country, fontsize=16)

import matplotlib.dates as mdates
months_fmt = mdates.DateFormatter('%b-%e')
evo = covid_19[covid_19.Country=='US'].groupby('Date')[['Confirmed','Deaths','Recovered']].sum()
evo['Active'] = evo['Confirmed'] - evo['Deaths'] - evo['Recovered']
evo['Death Rate'] = evo['Deaths'] / evo['Confirmed'] * 100
evo['Recover Rate'] = evo['Recovered'] / evo['Confirmed'] * 100
fig, ax = plt.subplots(2, 2, figsize=(12,9))
def plot_evo(num, col, title):
    ax[num].plot(evo[col], lw=3)
    ax[num].set_title(title)
    ax[num].xaxis.set_major_locator(plt.MaxNLocator(7))
    ax[num].xaxis.set_major_formatter(months_fmt)
    ax[num].grid(True)
@app.route("/co19_3", methods=['GET'])
def covid_3():
    plot_evo((0,0), 'Confirmed', 'Confirmed cases')
    plot_evo((0,1), 'Deaths', 'Death cases')
    plot_evo((1,0), 'Active', 'Active cases')
    plot_evo((1,1), 'Death Rate', 'Death rate')
    plt.savefig(img)
    plt.savefig('static/images/new_plot3.png')
    return render_template('images.html', name = 'Evolution of Cases in USA', url ='static/images/new_plot3.png')


import matplotlib.dates as mdates
months_fmt = mdates.DateFormatter('%b-%e')
evo = covid_19[covid_19.Country=='India'].groupby('Date')[['Confirmed','Deaths','Recovered']].sum()
evo['Active'] = evo['Confirmed'] - evo['Deaths'] - evo['Recovered']
evo['Death Rate'] = evo['Deaths'] / evo['Confirmed'] * 100
evo['Recover Rate'] = evo['Recovered'] / evo['Confirmed'] * 100
fig, ax = plt.subplots(2, 2, figsize=(12,9))
def plot_evo(num, col, title):
    ax[num].plot(evo[col], lw=3)
    ax[num].set_title(title)
    ax[num].xaxis.set_major_locator(plt.MaxNLocator(7))
    ax[num].xaxis.set_major_formatter(months_fmt)
    ax[num].grid(True)
@app.route("/co19_4", methods=['GET'])
def covid_4():
    plot_evo((0,0), 'Confirmed', 'Confirmed cases')
    plot_evo((0,1), 'Deaths', 'Death cases')
    plot_evo((1,0), 'Active', 'Active cases')
    plot_evo((1,1), 'Death Rate', 'Death rate')
    plt.title('Evolution of cases in India')
    plt.savefig(img)
    plt.savefig('static/images/new_plot4.png')
    return render_template('images.html', name = 'Evolution of Cases in India', url ='static/images/new_plot4.png')


import matplotlib.dates as mdates
months_fmt = mdates.DateFormatter('%b-%e')
evo = covid_19[covid_19.Country=='China'].groupby('Date')[['Confirmed','Deaths','Recovered']].sum()
evo['Active'] = evo['Confirmed'] - evo['Deaths'] - evo['Recovered']
evo['Death Rate'] = evo['Deaths'] / evo['Confirmed'] * 100
evo['Recover Rate'] = evo['Recovered'] / evo['Confirmed'] * 100
fig, ax = plt.subplots(2, 2, figsize=(12,9))
def plot_evo(num, col, title):
    ax[num].plot(evo[col], lw=3)
    ax[num].set_title(title)
    ax[num].xaxis.set_major_locator(plt.MaxNLocator(7))
    ax[num].xaxis.set_major_formatter(months_fmt)
    ax[num].grid(True)
@app.route("/co19_5", methods=['GET'])
def covid_5():
    plot_evo((0, 0), 'Confirmed', 'Confirmed cases')
    plot_evo((0, 1), 'Deaths', 'Death cases')
    plot_evo((1, 0), 'Active', 'Active cases')
    plot_evo((1, 1), 'Death Rate', 'Death rate')
    plt.title('Evolution of cases in China')
    plt.savefig(img)
    plt.savefig('static/images/new_plot5.png')
    return render_template('images.html', name = 'Evolution of Cases in China', url ='static/images/new_plot5.png')


import matplotlib.dates as mdates
months_fmt = mdates.DateFormatter('%b-%e')
evo = covid_19[covid_19.Country=='Italy'].groupby('Date')[['Confirmed','Deaths','Recovered']].sum()
evo['Active'] = evo['Confirmed'] - evo['Deaths'] - evo['Recovered']
evo['Death Rate'] = evo['Deaths'] / evo['Confirmed'] * 100
evo['Recover Rate'] = evo['Recovered'] / evo['Confirmed'] * 100
fig, ax = plt.subplots(2, 2, figsize=(12,9))
def plot_evo(num, col, title):
    ax[num].plot(evo[col], lw=3)
    ax[num].set_title(title)
    ax[num].xaxis.set_major_locator(plt.MaxNLocator(7))
    ax[num].xaxis.set_major_formatter(months_fmt)
    ax[num].grid(True)
@app.route("/co19_6", methods=['GET'])
def covid_6():
    plot_evo((0, 0), 'Confirmed', 'Confirmed cases')
    plot_evo((0, 1), 'Deaths', 'Death cases')
    plot_evo((1, 0), 'Active', 'Active cases')
    plot_evo((1, 1), 'Death Rate', 'Death rate')
    plt.title('Evolution of cases in Italy')
    plt.savefig(img)
    plt.savefig('static/images/new_plot6.png')
    return render_template('images.html', name = 'Evolution of Cases in Italy', url ='static/images/new_plot6.png')


#Comparison of Confirmed Cases Since First Case
df_break = covid_19.groupby(['Country','days'])['Confirmed','Deaths'].sum().reset_index()
top10case = latest.sort_values('Confirmed', ascending=False).head(10)['Country'].to_list()
_ = df_break[df_break.Country.isin(top10case)]
@app.route("/co19_7", methods=['GET'])
def covid_7():
    plt.figure(figsize=(14,8))
    sns.lineplot(x='days',y='Confirmed', data=_, hue='Country', lw=2)
    plt.legend(bbox_to_anchor=(1.02, 1), fontsize=10)
    plt.grid(True)
    plt.title('Comparison of Confirmed Case Since First Case')
    plt.savefig(img)
    plt.savefig('static/images/new_plot7.png')
    return render_template('images.html', name = 'Comparison of Confirmed cases', url ='static/images/new_plot7.png')

# Comparison of Death cases Since First Case
df_break = covid_19.groupby(['Country', 'days'])['Confirmed', 'Deaths'].sum().reset_index()
top10death = latest.sort_values('Deaths', ascending=False).head(10)['Country'].to_list()
_ = df_break[df_break.Country.isin(top10death)]
@app.route("/co19_8", methods=['GET'])
def covid_8():
    plt.figure(figsize=(14,8))
    sns.lineplot(x='days',y='Deaths', data=_, hue='Country', lw=2)
    plt.legend(bbox_to_anchor=(1.02, 1), fontsize=10)
    plt.grid(True)
    plt.title('Comparison of Death Since First Case')
    plt.savefig(img)
    plt.savefig('static/images/new_plot8.png')
    return render_template('images.html', name = 'Comparison of Death Cases', url ='static/images/new_plot8.png')


# Trajectory of first 10 countries with confirmed cases\
early_10 = covid_19.groupby('Country')['Date'].min().sort_values().head(10).index
@app.route("/co19_9", methods=['GET'])

def covid_9():
    _ = df_break[df_break.Country.isin(early_10)]
    plt.figure(figsize=(14,8))
    sns.lineplot(x='days',y='Confirmed', data=_, hue='Country', lw=2)
    plt.legend(bbox_to_anchor=(1.02, 1), fontsize=10)
    plt.yscale('log')
    plt.grid(True)
    plt.title('Comparison of First 10 Countries with Confirmed Cases')
    plt.savefig(img)
    plt.savefig('static/images/new_plot9.png')
    return render_template('images.html', name = 'Comparison of top 10 countries with confirmed Cases', url ='static/images/new_plot9.png')


#covid 19


#covid 19


covid_19 = covid_19.groupby(['Date', 'Country'])['Confirmed', 'Deaths', 'Recovered']
covid_19 = covid_19.sum().reset_index()
c_lat = covid_19[covid_19['Date'] == max(covid_19['Date'])].reset_index()
c_lat_grp = c_lat.groupby('Country')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

@app.route("/co19_10", methods=['GET'])
def covid_10():
    global c_lat_grp
    fig = px.choropleth(c_lat_grp, locations="Country", locationmode='country names',
                        color="Confirmed", hover_name="Country",
                        color_continuous_scale="ylorrd", title='COVID-19 Confirmed')
    fig.update(layout_coloraxis_showscale=False)
    #fig.show()
    plotly.offline.plot(fig, filename="choro", image='svg', auto_open=True)
    return render_template("Vizualize.html")

@app.route("/co19_11", methods=['GET'])
def covid_11():
    global c_lat_grp
    fig = px.choropleth(c_lat_grp[c_lat_grp['Deaths']>0], locations="Country", locationmode='country names',
                    color="Deaths", hover_name="Country",
                    color_continuous_scale="ylorrd", title='COVID-19 DEATHS')
    fig.update(layout_coloraxis_showscale=False)
    plotly.offline.plot(fig, filename="choro", image='svg', auto_open=True)
    return render_template("Vizualize.html")

@app.route("/co19_12", methods=['GET'])
def covid_12():
    global c_lat_grp
    fig = px.treemap(c_lat_grp.sort_values(by='Confirmed', ascending=False).reset_index(drop=True),
                 path=["Country"], values="Confirmed", title='COVID-19 Confirmed',
                 color_discrete_sequence = px.colors.qualitative.Dark2)
    fig.data[0].textinfo = 'label+text+value'
    #fig.show()
    #fig.savefig(img)
    #img.seek(0)
    plotly.offline.plot(fig, filename="choro", image='svg', auto_open=True)
    return render_template("Vizualize.html")

@app.route("/co19_13", methods=['GET'])
def covid_13():
    global c_lat_grp
    fig = px.treemap(c_lat_grp.sort_values(by='Deaths', ascending=False).reset_index(drop=True),
                 path=["Country"], values="Deaths", title='COVID-19 Deaths',
                 color_discrete_sequence = px.colors.qualitative.Dark2)
    fig.data[0].textinfo = 'label+text+value'
    #fig.show()
    #fig.savefig(img)
    #img.seek(0)
    plotly.offline.plot(fig, filename="choro", image='svg', auto_open=True)
    return render_template("Vizualize.html")

#EBOLA

ebola_14 = pd.read_csv("./ebola.csv",
                       parse_dates=['Date'])

ebola_14 = ebola_14[ebola_14['Date']!=max(ebola_14['Date'])]

# selecting important columns only
ebola_14 = ebola_14[['Date', 'Country', 'No. of confirmed, probable and suspected cases',
                     'No. of confirmed, probable and suspected deaths']]

# renaming columns
ebola_14.columns = ['Date', 'Country', 'Cases', 'Deaths']
ebola_14.head()

# group by date and country
ebola_14 = ebola_14.groupby(['Date', 'Country'])['Cases', 'Deaths']
ebola_14 = ebola_14.sum().reset_index()

# filling missing values
ebola_14['Cases'] = ebola_14['Cases'].fillna(0)
ebola_14['Deaths'] = ebola_14['Deaths'].fillna(0)

# converting datatypes
ebola_14['Cases'] = ebola_14['Cases'].astype('int')
ebola_14['Deaths'] = ebola_14['Deaths'].astype('int')

# latest
e_lat = ebola_14[ebola_14['Date'] == max(ebola_14['Date'])].reset_index()

# latest grouped by country
e_lat_grp = e_lat.groupby('Country')['Cases', 'Deaths'].sum().reset_index()

# nth day
ebola_14['nth_day'] = (ebola_14['Date'] - min(ebola_14['Date'])).dt.days

# day by day
e_dbd = ebola_14.groupby('Date')['Cases', 'Deaths'].sum().reset_index()

# nth day
e_dbd['nth_day'] = ebola_14.groupby('Date')['nth_day'].max().values

# no. of countries
temp = ebola_14[ebola_14['Cases']>0]
e_dbd['n_countries'] = temp.groupby('Date')['Country'].apply(len).values

e_dbd['new_cases'] = e_dbd['Cases'].diff()
e_dbd['new_deaths'] = e_dbd['Deaths'].diff()
e_dbd['epidemic'] = 'EBOLA'

@app.route("/co19_14", methods=['GET'])
def covid_14():
    fig = px.choropleth(e_lat_grp[e_lat_grp['Cases']>0], locations="Country", locationmode='country names',
                    color="Cases", hover_name="Country",
                    color_continuous_scale="ylorrd", title='EBOLA 2014')
    fig.update(layout_coloraxis_showscale=False)
    #fig.show()
    plotly.offline.plot(fig, filename="choro", image='svg', auto_open=True)
    return render_template("Vizualize.html")

@app.route("/co19_15", methods=['GET'])
def covid_15():
    fig = px.choropleth(e_lat_grp[e_lat_grp['Deaths']>0], locations="Country", locationmode='country names',
                    color="Deaths", hover_name="Country",
                    color_continuous_scale="ylorrd", title='EBOLA 2014 Deaths')
    fig.update(layout_coloraxis_showscale=False)
    #fig.show()
    plotly.offline.plot(fig, filename="choro", image='svg', auto_open=True)
    return render_template("Vizualize.html")


@app.route("/co19_16", methods=['GET'])
def covid_16():
    fig = px.treemap(e_lat_grp.sort_values(by='Cases', ascending=False).reset_index(drop=True),
                 path=["Country"], values="Cases", title='EBOLA',
                 color_discrete_sequence = px.colors.qualitative.Dark2)
    fig.data[0].textinfo = 'label+text+value'
    #fig.show()
    plotly.offline.plot(fig, filename="choro", image='svg', auto_open=True)
    return render_template("Vizualize.html")

@app.route("/co19_17", methods=['GET'])
def covid_17():
    fig = px.treemap(e_lat_grp.sort_values(by='Deaths', ascending=False).reset_index(drop=True),
                 path=["Country"], values="Deaths", title='EBOLA',
                 color_discrete_sequence = px.colors.qualitative.Dark2)
    fig.data[0].textinfo = 'label+text+value'
    #fig.show()
    plotly.offline.plot(fig, filename="choro", image='svg', auto_open=True)
    return render_template("Vizualize.html")

#SARS

# sars dataset
sars_03 = pd.read_csv("./sars_2003.csv",
                       parse_dates=['Date'])

# selecting important columns only
sars_03 = sars_03[['Date', 'Country', 'Cumulative number of case(s)',
                   'Number of deaths', 'Number recovered']]

# renaming columns
sars_03.columns = ['Date', 'Country', 'Cases', 'Deaths', 'Recovered']

# group by date and country
sars_03 = sars_03.groupby(['Date', 'Country'])['Cases', 'Deaths', 'Recovered']
sars_03 = sars_03.sum().reset_index()

# latest
s_lat = sars_03[sars_03['Date'] == max(sars_03['Date'])].reset_index()

# latest grouped by country
s_lat_grp = s_lat.groupby('Country')['Cases', 'Deaths', 'Recovered'].sum().reset_index()

# nth day
sars_03['nth_day'] = (sars_03['Date'] - min(sars_03['Date'])).dt.days

# day by day
s_dbd = sars_03.groupby('Date')['Cases', 'Deaths', 'Recovered'].sum().reset_index()

# nth day
s_dbd['nth_day'] = sars_03.groupby('Date')['nth_day'].max().values

# no. of countries
temp = sars_03[sars_03['Cases']>0]
s_dbd['n_countries'] = temp.groupby('Date')['Country'].apply(len).values


s_dbd['new_cases'] = s_dbd['Cases'].diff()
s_dbd['new_deaths'] = s_dbd['Deaths'].diff()
s_dbd['epidemic'] = 'SARS'

@app.route("/co19_18", methods=['GET'])
def covid_18():
    fig = px.choropleth(s_lat_grp[s_lat_grp['Deaths']>0], locations="Country", locationmode='country names',
                    color="Deaths", hover_name="Country",
                    color_continuous_scale="ylorrd", title='SARS 2003')
    fig.update(layout_coloraxis_showscale=False)
    #fig.show()
    plotly.offline.plot(fig, filename="choro", image='svg', auto_open=True)
    return render_template("Vizualize.html")

@app.route("/co19_19", methods=['GET'])
def covid_19():
    fig = px.choropleth(s_lat_grp[s_lat_grp['Deaths']>0], locations="Country", locationmode='country names',
                    color="Deaths", hover_name="Country",
                    color_continuous_scale="ylorrd", title='SARS Deaths')
    fig.update(layout_coloraxis_showscale=False)
    #fig.show()
    plotly.offline.plot(fig, filename="choro", image='svg', auto_open=True)
    return render_template("Vizualize.html")

@app.route("/co19_20", methods=['GET'])
def covid_20():
    fig = px.treemap(s_lat_grp.sort_values(by='Cases', ascending=False).reset_index(drop=True),
                 path=["Country"], values="Cases", title='SARS',
                 color_discrete_sequence = px.colors.qualitative.Dark2)
    fig.data[0].textinfo = 'label+text+value'
    #fig.show()
    plotly.offline.plot(fig, filename="choro", image='svg', auto_open=True)
    return render_template("Vizualize.html")

@app.route("/co19_21", methods=['GET'])
def covid_21():
    fig = px.treemap(s_lat_grp.sort_values(by='Deaths', ascending=False).reset_index(drop=True),
                 path=["Country"], values="Deaths", title='SARS',
                 color_discrete_sequence = px.colors.qualitative.Dark2)
    fig.data[0].textinfo = 'label+text+value'
    #fig.show()
    plotly.offline.plot(fig, filename="choro", image='svg', auto_open=True)
    return render_template("Vizualize.html")


#H1N1

#h1n1

States = ["United States", "Brazil", "India",
          "Mexico", "China", "Turkey",
          "Argentina", "Russia", "United Kingdom",
          "Canada","France", "Spain", "Egypt",
          "Germany", "South Korea", "Thailand",
          "Italy", "Columbia", "Peru", "Ukraine",
          "Ecuador", "Japan", "Australia",
          "Poland", "Chile", "Syria",
          "Greece", "Iran", "Venezuela", "Hungary",
          "Saudi Arabia", "Portugal", "Romania",
          "Czech Republic", "Israel", "South Africa",
          "Malaysia", "Belarus", "Serbia", "Hong Kong",
          "Cuba","Costa Rica","Morocco","Netherlands",
          "Bolivia","Vietnam","Algeria","Finland",
          "Slovakia","Paraguay","New Zeland","Taiwan",
          "Sri Lanka","Moldova","Palestinian Territories","Iraq","Austria","Bulgaria"]
Confirm = [113690,58178,33783,70715,120940,12316,11458,25339,28456,25828,1980000,1538,15812,222360,
              107939,31902,3064933,4310,9165,494,2251,11636,37484,2024,12258,452,17977,3672,2187,283,14500,
              166922,7006,2445,4330,12640,12210,102,695,33109,973,1867,2890,1473,2310,11186,916,6122,955,855,3199,5474,642,2524,1676,2880,964,766]
Deaths_value= [3433,2135,2024,1316,800,656,626,604,474,429,344,300,278,258,250,249,244,272,223,213,200,198,187,181,156,152,149,147,135,134,128,122,122,102,94,93,92,88,83,80,69,67,64,62,59,58,57,56,56,54,50,48,48,46,43,42,40,40]

df = pd.DataFrame(
    dict(States=States, Confirmed=Confirm, Deaths_Value=Deaths_value)
)
df["States_in_US"] = "States_in_US"
df["Deaths"]="Deaths"

df["Active_Value"]=  df["Active"]=df["Confirmed"]-df["Deaths_Value"]
df["Active"]="Active"
df.head(60)

fig3 = go.Figure()
fig3.add_trace(go.Treemap(
    ids=df["Confirmed"],
    labels=df["States"],
    parents=df["Active"],
    values=df["Active_Value"],
    textinfo='label+value',
    domain=dict(column=0)
))

fig3.add_trace(go.Treemap(
    ids=df["Confirmed"],
    labels=df["States"],
    parents=df["Deaths"],
    values=df["Deaths_Value"],
    textinfo='label+value',
    domain=dict(column=1),
))

fig3.update_layout(
    grid= dict(columns=2, rows=1),
    margin = dict(t=0, l=0, r=0, b=0),
)
@app.route("/co19_26", methods=['GET'])
def covid_26():
    global fig3
    plotly.offline.plot(fig3, filename="choro", image='svg', auto_open=True)
    #fig3.show()
    return render_template("Vizualize.html")

#comparison

c_cases = sum(c_lat_grp['Confirmed'])
c_deaths = sum(c_lat_grp['Deaths'])
c_no_countries = len(c_lat_grp['Country'].value_counts())
s_cases = sum(s_lat_grp['Cases'])
s_deaths = sum(s_lat_grp['Deaths'])
s_no_countries = len(s_lat_grp['Country'].value_counts())
e_cases = sum(e_lat_grp['Cases'])
e_deaths = sum(e_lat_grp['Deaths'])
e_no_countries = len(e_lat_grp['Country'].value_counts())

epidemics = pd.DataFrame({
    'epidemic' : ['COVID-19', 'SARS', 'EBOLA', 'H1N1'],
    'start_year' : [2019, 2003, 2014, 2009],
    'end_year' : [0000, 2004, 2016, 2010],
  'confirmed' : [c_cases, s_cases, e_cases, 6724149],
    'deaths' : [c_deaths, s_deaths, e_deaths, 19654],
    'no_of_countries' : [c_no_countries, s_no_countries, e_no_countries, 178]
})
epidemics['mortality'] = round((epidemics['deaths']/epidemics['confirmed'])*100, 2)
epidemics = epidemics.sort_values('end_year').reset_index(drop=True)
epidemics.head()

@app.route("/co19_22", methods=['GET'])
def covid_22():
    fig = px.bar(epidemics.sort_values('confirmed',ascending=False),
             x="confirmed", y="epidemic", color='epidemic',
             text='confirmed', orientation='h', title='No. of Cases',
             color_discrete_sequence = [h,a,n,e,s])
    fig.update_traces(textposition='auto')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    #fig.show()
    plotly.offline.plot(fig, filename="choro", image='svg', auto_open=True)
    return render_template("Vizualize.html")

@app.route("/co19_23", methods=['GET'])
def covid_23():
    fig = px.bar(epidemics.sort_values('deaths',ascending=False),
             x="deaths", y="epidemic", color='epidemic',
             text='deaths', orientation='h', title='No. of Deaths',
             color_discrete_sequence = [h,a,n,e,s])
    fig.update_traces(textposition='auto')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    #fig.show()
    plotly.offline.plot(fig, filename="choro", image='svg', auto_open=True)
    return render_template("Vizualize.html")

@app.route("/co19_24", methods=['GET'])
def covid_24():
    fig = px.bar(epidemics.sort_values('mortality',ascending=False),
             x="mortality", y="epidemic", color='epidemic',
             text='mortality', orientation='h', title='Moratlity rate',
             range_x=[0,100],
             color_discrete_sequence = [h,a,n,e,s])
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    #fig.show()
    plotly.offline.plot(fig, filename="choro", image='svg', auto_open=True)
    return render_template("Vizualize.html")

@app.route("/co19_25", methods=['GET'])
def covid_25():
    fig = px.bar(epidemics.sort_values('no_of_countries', ascending=False),
             x="no_of_countries", y="epidemic", color='epidemic',
             text='no_of_countries', orientation='h', title='No. of Countries',
             range_x=[0,200],
             color_discrete_sequence = [h,a,n,e,s])
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    #fig.show()
    plotly.offline.plot(fig, filename="choro", image='svg', auto_open=True)
    return render_template("Vizualize.html")

if __name__ == "__main__":
       app.run(debug=True)








