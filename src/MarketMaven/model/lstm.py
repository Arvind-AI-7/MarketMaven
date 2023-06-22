import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class Model_LSTM(nn.Module):
    def __init__(self, train_x, test_x, train_y, test_y, X_forecast):
        super(Model_LSTM, self).__init__()
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y.values.reshape(-1, 1)
        self.test_y = test_y.values.reshape(-1, 1)
        self.X_forecast = X_forecast

        self.lstm1 = nn.LSTM(input_size=self.train_x.shape[1], hidden_size=50, num_layers=4, batch_first=True,
                             dropout=0.1)
        self.fc = nn.Linear(50, 1)

        self.sc_x = MinMaxScaler(feature_range=(0, 1))
        self.sc_y = MinMaxScaler(feature_range=(0, 1))

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def model(self):
        train_x_scaled = self.sc_x.fit_transform(self.train_x)
        test_x_scaled = self.sc_x.transform(self.test_x)

        train_y_scaled = self.sc_y.fit_transform(self.train_y)
        test_y_scaled = self.sc_y.transform(self.test_y)

        train_x_tensor = torch.tensor(train_x_scaled, dtype=torch.float32)
        train_y_tensor = torch.tensor(train_y_scaled, dtype=torch.float32)
        test_x_tensor = torch.tensor(test_x_scaled, dtype=torch.float32)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        num_epochs = 100
        batch_size = 32
        num_batches = len(train_x_tensor) // batch_size

        for epoch in range(num_epochs):
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = (batch + 1) * batch_size
                batch_x = train_x_tensor[start_idx:end_idx]
                batch_y = train_y_tensor[start_idx:end_idx]

                outputs = self(batch_x.unsqueeze(1))
                loss = criterion(outputs, batch_y.unsqueeze(1))

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.eval()

        with torch.no_grad():
            predictions_scaled = self(test_x_tensor.unsqueeze(1))
            predictions = self.sc_y.inverse_transform(predictions_scaled.numpy())
            #test_y_inverse = self.sc_y.inverse_transform(test_y_scaled)

        X_F = self.sc_x.transform(self.X_forecast.values.reshape(1, -1))
        X_F = torch.tensor(X_F, dtype=torch.float32)
        forecast = self.sc_y.inverse_transform(self(X_F.unsqueeze(1)).detach().numpy())

        return predictions, forecast


