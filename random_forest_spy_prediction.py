import yfinance as yf
spyticker = yf.Ticker("SPY")
df_spy = spyticker.history(period="max", interval="1d", start="1993-12-01", end="2022-01-01" , auto_adjust=True, rounding=True)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


print(df_spy.head())


features = ['Open', 'High', 'Low']
target = 'Close'


x = df_spy[features]
y = df_spy[target]



from sklearn.ensemble import RandomForestClassifier
import numpy as np



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Create a random forest classification model.  Set min_samples_split high to ensure we don't overfit.




train=df_spy.loc[df_spy.index<'01-01-2021']
test=df_spy.loc[df_spy.index>='31-12-2020']

fig,ax=plt.subplots(figsize=(15,5))
train.plot(ax=ax,label='Training Set')
test.plot(ax=ax,label='Test Set')


rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=True)


rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)
print(y_pred)


test_ohl=test[features]
y_pred = rf.predict(test_ohl)

plt.rcParams["figure.figsize"] = (20,10)

plt.plot(test.index,test[target], color = 'red', label="Actual Data")
plt.plot(test.index,y_pred, color='black', label="Prediction")

plt.title('NSEI Stock Prices')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.legend()
plt.show()

import datetime


data = yf.Ticker("SPY")
end_date = datetime.date.today() - datetime.timedelta(days=1)
start_date = end_date - datetime.timedelta(days=365)
day = datetime.date.today()
stock = data.history(start=start_date, end=end_date)

def next_day_price(stock, day):
    x=stock[['Open','High','Low']]
    y=stock['Close']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    predicted_price=model.predict([stock.iloc[-1][['Open','High','Low']]])
    print(f"Predicted price for the {day}:", predicted_price)

next_day_price(stock,day)
