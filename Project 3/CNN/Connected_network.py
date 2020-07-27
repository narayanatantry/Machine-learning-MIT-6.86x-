class MLP(nn.Module):

    def __init__(self, input_dimension):
        super(MLP, self).__init__()
        self.flatten = Flatten()
        # TODO initialize model layers here

    def forward(self, x):
        xf = self.flatten(x)

        # TODO use model layers to predict the two digits

        return out_first_digit, out_second_digit
