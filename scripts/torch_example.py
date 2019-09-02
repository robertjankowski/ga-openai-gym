import torch

device = torch.device('cuda:0')

N, D_in, H, D_out = 1, 4, 2, 1

x = torch.randn(N, D_in)
y = torch.randint(0, 2, (D_out, N), dtype=torch.float)
print(x)
print(y)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

criterion = torch.nn.BCEWithLogitsLoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    y_pred = model(x)
    print(y_pred, y)
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    model.zero_grad()

    loss.backward()

    optimizer.step()


class TwoLayerNN(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNN, self).__init__()
        self.layer1 = torch.nn.Linear(D_in, H).to(device=device)
        self.layer2 = torch.nn.Linear(H, D_out).to(device=device)

    def forward(self, x) -> torch.Tensor:
        h_relu = self.layer1(x).clamp(min=0)
        y_pred = self.layer2(h_relu)
        return y_pred


nn = TwoLayerNN(D_in, H, D_out)
optimizer = torch.optim.SGD(nn.parameters(), lr=1e-4)

for t in range(500):
    y_pred = nn(x)

    loss = criterion(y_pred, y)

    if t % 100 == 99:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# torch.save(nn, 'models/example_torch_2_layer.pt')
print(model.parameters())
