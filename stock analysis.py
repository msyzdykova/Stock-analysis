import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data
from datetime import datetime

stocks=['AAPL', 'GOOG', 'MSFT', 'AMZN']
end=datetime.now()
start=datetime(end.year-1, end.month, end.day)

for stock in stocks:
    globals()[stock]=data.DataReader(stock, 'yahoo', start, end)

globals=[AAPL, GOOG, MSFT, AMZN]

for stock in globals:
    print(stock.describe()) #not clear whats aapl, goog, etc

#automated graphs with a function

for stock in globals:
    stock['Daily Return']=stock['Adj Close'].pct_change()
           
def graph(cols, type):
    rows=len(globals)//cols
    rows+=len(globals)%cols
    pos=range(1, len(globals)+1)
    fig=plt.figure()  
    for n in range(len(globals)):
        ax=fig.add_subplot(rows,cols,pos[n])
        globals[n][type].plot(legend=True, figsize=(25,15))
        ax.set_title(type+' '+stocks[n])
    plt.show()
graph(2, 'Daily Return')

def m_ave(ma_day, cols):
    rows=len(globals)//cols
    rows+=len(globals)%cols
    pos=range(1, len(globals)+1)
    for ma in ma_day:
        col_name='MA for %s days'%(str(ma))
        for n in range(len(globals)):
            globals[n][col_name]=globals[n]['Adj Close'].rolling(window=ma).mean()
    fig=plt.figure() 
    for n in range(len(globals)):
        ax=fig.add_subplot(rows,cols,pos[n])
        globals[n]['Adj Close'].plot(legend=True, subplots=False, figsize=(19,16)) 
        for i in range(len(ma_day)):
            globals[n]['MA for '+str(ma_day[i])+' days'].plot(legend=True, subplots=False, figsize=(19,16)) 
            plt.legend(fontsize=5)
            ax.set_title(stocks[n])
ma_day=[10,20]            
m_ave(ma_day,2) 

closing_df=data.DataReader(stocks, 'yahoo', start, end)['Adj Close']
closing_df.head()

rets=closing_df.pct_change()
rets.head()

sns.jointplot('GOOG', 'MSFT', rets, kind='scatter')
sns.pairplot(rets.dropna())

returns_fig=sns.PairGrid(rets.dropna())
returns_fig.map_upper(plt.scatter, color='red')
returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
returns_fig.map_diag(plt.hist, bins=30)

returns_fig=sns.PairGrid(closing_df)
returns_fig.map_upper(plt.scatter, color='red')
returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
returns_fig.map_diag(plt.hist, bins=30)

plt.figure()
sns.heatmap(rets.dropna().corr(), annot=True)
plt.figure()
sns.heatmap(closing_df.corr(), annot=True)

#risk analysis
rts=rets.dropna()
area=np.pi*20
plt.scatter(rts.mean(), rts.std(), s=area)
plt.xlabel('Expected return')
plt.ylabel('Risk')
for label, x, y in zip(rts.columns, rts.mean(), rts.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))

#value at risk
#bootstrap method
def vr(cols):
    rows=len(globals)//cols
    rows+=len(globals)%cols
    pos=range(1, len(globals)+1)
    fig=plt.figure()  
    for n in range(len(globals)):
        ax=fig.add_subplot(rows,cols,pos[n])
        sns.distplot(globals[n]['Daily Return'].dropna(), bins=100)
        ax.set_title('Value at risk '+stocks[n])
    plt.tight_layout()
    plt.show()
vr(2)    

for stock in stocks:
    q=pd.Series(data=rts[stock].quantile(0.05), index=stocks)
    print(q)

#monte carlo method
days=365
dt=1/days
mu=rts.mean()['GOOG']
sigma=rts.std()['GOOG']

def stock_monte_carlo(start_price, days, mu, sigma):
    price=np.zeros(days)
    price[0]=start_price
    shock=np.zeros(days)
    drift=np.zeros(days)
    for x in range(1,days):        
        shock[x]=np.random.normal(loc=mu*dt, scale=sigma*np.sqrt(dt))
        drift[x]=mu*dt
        price[x]=price[x-1]+(price[x-1]*(drift[x]+shock[x]))
    return price

start_price=2510.459961

for run in range(100):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis')        

runs=1000
simulations=np.zeros(runs)
for run in range(runs):
    simulations[run]=stock_monte_carlo(start_price, days, mu, sigma)[days-1]   
            
q=np.percentile(simulations,1)
plt.hist(simulations, bins=200)
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)
plt.axvline(x=q, linewidth=4, color='r')
plt.title(u"Final price distribution for Google Stock after %s days" % days, weight='bold')   
    
    
    
