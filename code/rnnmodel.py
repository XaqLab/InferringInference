import torch
import torch.nn as nn
from torch import optim

class RNN(nn.Module):
    """
    -- RNN model class --
    Architecture:
    Input layer --> Hidden recurrent layer --> Output recurrent layer
    """

    def __init__(self, input_dim, hidden_dim, output_dim, use_cuda):
        super(RNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_cuda = use_cuda
        
        self.input_to_hidden = nn.Linear(input_dim + hidden_dim, hidden_dim, bias = True)
        self.hidden_to_output = nn.Linear(hidden_dim + output_dim, output_dim, bias = True)
        self.activation_function = nn.ReLU()
        
    
    def forward(self, inputs):
        
        N_batches, T = inputs.shape[0], inputs.shape[1]
        
        # Initialize the activity of the recurrent layers
        output_activity, hidden_activity = torch.rand(N_batches, self.output_dim), torch.rand(N_batches, self.hidden_dim)
        #output_activity = np.sqrt(self.output_dim)*output_activity
        #hidden_activity = np.sqrt(self.hidden_dim)*hidden_activity
        # output_activity = torch.sqrt(torch.tensor(float(self.output_dim)))*output_activity
        # hidden_activity = torch.sqrt(torch.tensor(float(self.hidden_dim)))*hidden_activity
        output_activity = (self.output_dim**0.5)*output_activity
        hidden_activity = (self.hidden_dim**0.5)*hidden_activity
        
        # r = torch.zeros(N_batches, T, self.output_dim) # output layer activity
        # h = torch.zeros(N_batches, T, self.hidden_dim) # hidden layer activity
        r = []
        h = []
        
        if self.use_cuda and torch.cuda.is_available():
            # r = r.cuda()
            # h = h.cuda()
            output_activity = output_activity.cuda()
            hidden_activity = hidden_activity.cuda()
        
        for t in range(T):
            
            combined_inputs_hiddenlayer = torch.cat((inputs[:,t,:], hidden_activity),1)
            hidden_activity = self.activation_function(self.input_to_hidden(combined_inputs_hiddenlayer))
            
            combined_inputs_outputlayer = torch.cat((hidden_activity, output_activity),1)
            output_activity = self.activation_function(self.hidden_to_output(combined_inputs_outputlayer))
            
            # r[:,t,:] = output_activity
            # h[:,t,:] = hidden_activity
            r.append(output_activity)
            h.append(output_activity)

        r = torch.stack(r).transpose(0,1)
        h = torch.stack(h).transpose(0,1)


        
        return r, h