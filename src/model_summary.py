import torch
import torchinfo
import torchviz
from keras.models import load_model

# For PyTorch Models
# model = torch.load("src/MarketMaven/pt_h5_pkl/lstm_rnn.pt")
# torchinfo.summary(model)

# For Keras Models
model = load_model("src/MarketMaven/pt_h5_pkl/sent_rnn.h5")
model.summary()
