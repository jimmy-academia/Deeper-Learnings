import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BestNet(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(BestNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = 256
        self.embedding_dropout=0.6
        self.desc_rnn_size = 100

        self.rnn = nn.GRU(
            input_size=self.embedding_dim, hidden_size=self.hidden_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.rnn_desc = nn.GRU(
            input_size=self.embedding_dim, hidden_size=self.desc_rnn_size,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.emb_drop = nn.Dropout(self.embedding_dropout)
        self.M = nn.Parameter(torch.FloatTensor(2*self.hidden_dim, 2*self.hidden_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.Wc = nn.Parameter(torch.FloatTensor(2*self.hidden_dim, self.embedding_dim))
        self.We = nn.Parameter(torch.FloatTensor(self.embedding_dim, self.embedding_dim))
        self.attn = nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)
        self.init_params_()
        self.tech_w = 0.0

    def init_params_(self):
        #Initializing parameters
        nn.init.xavier_normal_(self.M)

        # Set forget gate bias to 2
        size = self.rnn.bias_hh_l0.size(0)
        self.rnn.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn.bias_ih_l0.size(0)
        self.rnn.bias_ih_l0.data[size//4:size//2] = 2

        size = self.rnn_desc.bias_hh_l0.size(0)
        self.rnn_desc.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn_desc.bias_ih_l0.size(0)
        self.rnn_desc.bias_ih_l0.data[size//4:size//2] = 2

    # def forward(self, context, options):
    #     logits = []
    #     for i, option in enumerate(options.transpose(1, 0)):
    #         gits = []
    #         for context in context.transpose(1,0):
    #             git = self.forward_one_option(context, option)
    #             gits.append(logit)
    #         logit = torch.stack(gits).mean(0)
    #     logits = torch.stack(logits, 1)

    #     return logits.squeeze()

    # def forward(self, context, options):
    #     logits = []
    #     for i, option in enumerate(options.transpose(1, 0)):
    #         logit = self.forward_one_option(context, option)
    #         logits.append(logit)
    #     logits = torch.stack(logits, 1)

    #     return logits.squeeze()
    def forward(self, context, options):
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            logit_ = []
            for utter in context.transpose(1,0):
                logit = self.forward_one_option(utter, option)  # 10,1,1
                logit_.append(logit)
            logits.append(torch.stack(logit_,1).mean(1))
        logits = torch.stack(logits, 1)

        return logits.squeeze()

    def forward_one_option(self, context, option):
        context, c_h, option, o_h = self.forward_crosspath(context, option)
        context_attn = self.forward_attn(context, o_h)
        option_attn = self.forward_attn(option, c_h)
        final = self.forward_fc(context_attn, option_attn)
        return final

    def forward_crosspath(self, context, option):
        context, c_h = self.rnn(self.emb_drop(context))
        c_h = torch.cat([i for i in c_h], dim=-1)
        option, o_h = self.rnn(self.emb_drop(option))
        o_h = torch.cat([i for i in o_h], dim=-1)
        return context, c_h.squeeze(), option, o_h.squeeze()

    def forward_attn(self, output, hidden):
        max_len = output.size(1)
        b_size = output.size(0)

        hidden = hidden.squeeze(0).unsqueeze(2)
        attn = self.attn(output.contiguous().view(b_size*max_len, -1))
        attn = attn.view(b_size, max_len, -1)
        attn_energies = (attn.bmm(hidden).transpose(1,2))
        alpha = F.softmax(attn_energies.squeeze(1), dim=-1)
        alpha = alpha.unsqueeze(1)
        weighted_attn = alpha.bmm(output)

        return weighted_attn.squeeze()

    def forward_fc(self, context, option):
        out = torch.mm(context, self.M).unsqueeze(1)
        out = torch.bmm(out, option.unsqueeze(2))
        out = out + self.b
        return out

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

