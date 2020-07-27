class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, (3, 3))
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d((2,2))
        self.conv2d_2 = nn.Conv2d(32, 64, (3, 3))
        self.flatten = Flatten() 
        self.linear1 = nn.Linear(2880, 64)
        self.dropout = nn.Dropout(p = 0.5)
        self.linear2 = nn.Linear(64, 20)
        
    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.maxpool2d(x) 
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        out_first_digit = x[:,:10]
        out_second_digit = x[:,10:]
        
        return out_first_digit, out_second_digit
