import yfinance as yf
import matplotlib.pyplot as plt 
import numpy as np
#!matplotlib inline
import sklearn 
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import torch 
import torch.nn as nn 






axon_data = yf.download("AXON","2010-01-01","2023-08-01")

#print(axon_data)

price = axon_data[["Close"]]

scaler = MinMaxScaler(feature_range=(-1,1))

price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))

#price['Close'].plot()
#plt.show()

#axon_data['Adj Close'].plot()
#plt.show()


######################
# split the data #####
######################

def data_split(stock, lookback):

    data_raw = stock.to_numpy()

    data = []



    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index:index+lookback])

    data = np.array(data)
    
    test_set_size = int(np.round(0.2*data.shape[0]))

    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size,:-1,:]

    y_train = data[:train_set_size,-1,:]


    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]


    return([x_train, y_train, x_test, y_test])
    
lookback = 20 #sequence length 

x_train, y_train, x_test, y_test = data_split(price, lookback)


""" data split """

x_train = torch.from_numpy(x_train).type(torch.Tensor)


x_test = torch.from_numpy(x_test).type(torch.Tensor)

y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)

y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)

y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)

y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)




#########################
# LSTM / GRU parameters #
#########################

input_dim = 1 

hidden_dim = 32 
num_layers = 2 
output_dim = 1 
num_epochs = 100



class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):

        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers 

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,batch_first=True)

        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        c0 = torch.zeros(self.num_layers, x.size(0),self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h_0.detach(), c0.detach()))
        #x =  self.lstm(x, h_0.detach(),c0.detach())
        print(out)
        out = self.linear(out[:,-1,:])

        return(out)
    



model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,output_dim=output_dim)
print(model)



criterion = torch.nn.MSELoss(reduction="mean")

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

import time 


hist = np.zeros(num_epochs)

start_time = time.time()

lstm = []

for t in range(num_epochs):

    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train_lstm)

    print("epoch: ", t, "MSE: ", loss.item())

    hist[t] = loss.item()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

training_time = time.time()- start_time
print("training time: {}".format(training_time))



predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))

original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))



sns.set_style("darkgrid")

fig = plt.figure()

fig.subplots_adjust(hspace=0.2, wspace=0.2)

plt.subplot(1,2,1)

ax = sns.lineplot(x=predict.index, y= predict[0], label="training prediction", color="tomato")

ax.set_title("stock price", size=14, fontweight="bold")

ax.set_xlabel("Days", size=14)

ax.set_ylabel("cost (USD): ",size=14)

ax.set_xticklabels('',size=10)

fig.set_figheight(16)
fig.set_figwidth(16)
plt.show()


#ax = sns.lineplot(data=hist, color="royalblue")
#ax.set_xlabel("Epoch: ",size=14)
#ax.set_ylabel("Loss: ", size=14)
#ax.set_title("Training Loss", size=14, fontweight="bold")


