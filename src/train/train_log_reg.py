import pandas as pd
import torch
import torch.nn as nn

import preprocessing as prep

dtype = torch.float
device = torch.device("cpu")


#
# load data
#
digits = prep.Digits('../../data/zip.train')
inputs, targets = digits.load_data()


#
# set dimensions and convert to tensors
#

N = inputs.shape[0]
D_in = inputs.shape[1]
D_out = int(targets.max()) + 1

x = torch.tensor(inputs.values, device=device, dtype=dtype)
y = torch.tensor(targets.values, device=device, dtype=torch.long).squeeze()


#
# Hyper-parameters
#
learning_rate = 0.0001
batch_size = 256


# Neuronal Network
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, D_out)
)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



# Train
loss_hist = []

epochs = range(20000)
idx = 0
for t in epochs:

    for batch in range(0, int(N / batch_size)):
        # Berechne den Batch

        batch_x = x[batch * batch_size: (batch + 1) * batch_size, :]
        batch_y = y[batch * batch_size: (batch + 1) * batch_size]

        # Berechne die Vorhersage (foward step)
        outputs = model(batch_x)

        # Berechne den Fehler (Ausgabe des Fehlers alle 100 Iterationen)
        loss = criterion(outputs, batch_y)

        # Berechne die Gradienten und Aktualisiere die Gewichte (backward step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Berechne den Fehler (Ausgabe des Fehlers alle 50 Iterationen)
    if t % 100 == 0:
        loss_hist.append(loss.item())
        print(t, loss.item())
        #torch.save(model, '../../models/digit_log_reg__' + str(t) + '__.pt')


# Save model and loss history
torch.save(model, '../../models/digit_log_reg__new.pt')
pd.DataFrame(loss_hist).to_csv('../../models/digit_log_reg__errors.csv')
