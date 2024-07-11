class ASLModel(nn.Module):
    def __init__(self):
        super(ASLModel, self).__init__()
        self.fc1 = nn.Linear(21 * 3, 512)
        self.fc2 = nn.Linear(512, 29)  # 29 classes

    def forward(self, x):
        x = x.view(-1, 21 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x