import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

torch.manual_seed(20)

class Model_LSTM(nn.Module):
    def __init__(self, train_x, test_x, train_y, test_y, X_forecast):
        super(Model_LSTM, self).__init__()
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y.values.reshape(-1, 1)
        self.test_y = test_y.values.reshape(-1, 1)
        self.X_forecast = X_forecast

        self.lstm1 = nn.LSTM(input_size=self.train_x.shape[1], hidden_size=50, num_layers=3,
                             batch_first=True, dropout=0.2)
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
        test_y_tensor = torch.tensor(test_y_scaled, dtype=torch.float32)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        num_epochs = 100
        batch_size = 1

        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        early_stopping_counter = 0
        early_stopping_patience = 5
        best_model_state = None

        for epoch in range(num_epochs):
            for batch_x, batch_y in train_dataloader:
                optimizer.zero_grad()
                outputs = self(batch_x.unsqueeze(1))
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()

            self.eval()

            loss_sum = 0.0
            for batch_x, batch_y in train_dataloader:
                outputs = self(batch_x.unsqueeze(1))
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss_sum += loss.item()

            avg_loss = loss_sum / len(train_dataloader)

            if avg_loss < best_loss:
                best_loss = avg_loss
                early_stopping_counter = 0
                best_model_state = self.state_dict()
            else:
                early_stopping_counter += 1

            print(f"LSTM Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

            if early_stopping_counter >= early_stopping_patience:
                print("Optima reached")
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break

        predictions_scaled = self(test_x_tensor.unsqueeze(1))
        predictions = self.sc_y.inverse_transform(predictions_scaled.detach().numpy())

        X_F = self.sc_x.transform(self.X_forecast.values.reshape(1, -1))
        X_F = torch.tensor(X_F, dtype=torch.float32)
        forecast = self.sc_y.inverse_transform(self(X_F.unsqueeze(1)).detach().numpy())

        return predictions, forecast