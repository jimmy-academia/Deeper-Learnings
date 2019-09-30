import torch
from torch.autograd import Variable

class LinearNet(torch.nn.Module):
    def __init__(self, dim_embeddings):
        super(LinearNet, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_embeddings, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256)
        )

    def forward(self, context, options):
        context = self.mlp(context).max(1)[0]
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option = self.mlp(option).max(1)[0]
            logit = ((context - option) ** 2).sum(-1)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

class GruNet(torch.nn.Module):
    def __init__(self, dim_embeddings):
        super(GruNet, self).__init__()
        self.C_process = torch.nn.GRU(256, 256, 1, batch_first=True)
        self.context_word = torch.nn.GRU(300, 256, 1, batch_first=True)
        self.P_process = torch.nn.GRU(300, 256, 1, batch_first=True)
        self.W = torch.nn.Parameter(data=torch.rand(256, 256), requires_grad=True)

    def forward(self, context, options):

        ## context 10, 14, 30, 300
        context_vec = []
        for utterance in context.transpose(1,0):  # 10, 30, 300
            utter_vec, __ = self.context_word(utterance) # 10, 30, 256
            utter_vec = utter_vec.max(1)[0]         # 10, 256
            context_vec.append(utter_vec)           
        context_vec = torch.stack(context_vec, 1)   # 10, 14, 256
        context, __ = self.C_process(context_vec)   # 10, 14, 256

        ## context 10, 30, 300; options 10, 100, 50, 300
        # context, __ = self.C_process(context) ## 10,30,256
        context = context.max(1)[0] # 10,256
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):  # 100, 10, 50, 300
            option, __ = self.P_process(option) #10,50,300 -> 10,50, 256
            option = option.max(1)[0]   #10, 256
            logit = context.matmul(self.W).matmul(option.transpose(1,0)).sum(-1)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

class LastNet(torch.nn.Module):
    def __init__(self, dim_embeddings):
        super(LastNet, self).__init__()
        self.C_process = torch.nn.LSTM(300, 256, 1, batch_first=True)
        # self.context_word = torch.nn.LSTM(300, 256, 1, batch_first=True)
        self.P_process = torch.nn.LSTM(300, 256, 1, batch_first=True)
        self.W = torch.nn.Parameter(data=torch.rand(256, 256), requires_grad=True)

    def forward(self, context, options):

        ## context 10, 30, 300; options 10, 100, 50, 300
        context, __ = self.C_process(context) ## 10,30,256
        context = context.max(1)[0] # 10,256
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):  # 100, 10, 50, 300
            option, __ = self.P_process(option) #10,50,300 -> 10,50, 256
            option = option.max(1)[0]   #10, 256
            logit = context.matmul(self.W).matmul(option.transpose(1,0)).sum(-1)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

class RnnNet(torch.nn.Module):
    def __init__(self, dim_embeddings):
        super(RnnNet, self).__init__()
        self.C_process = torch.nn.LSTM(256, 256, 1, batch_first=True)
        self.context_word = torch.nn.LSTM(300, 256, 1, batch_first=True)
        self.P_process = torch.nn.LSTM(300, 256, 1, batch_first=True)
        self.W = torch.nn.Parameter(data=torch.rand(256, 256), requires_grad=True)

    def forward(self, context, options):

        ## context 10, 14, 30, 300
        context_vec = []
        for utterance in context.transpose(1,0):  # 10, 30, 300
            utter_vec, __ = self.context_word(utterance) # 10, 30, 256
            utter_vec = utter_vec.max(1)[0]         # 10, 256
            context_vec.append(utter_vec)           
        context_vec = torch.stack(context_vec, 1)   # 10, 14, 256
        context, __ = self.C_process(context_vec)   # 10, 14, 256

        ## context 10, 30, 300; options 10, 100, 50, 300
        # context, __ = self.C_process(context) ## 10,30,256
        context = context.max(1)[0] # 10,256
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):  # 100, 10, 50, 300
            option, __ = self.P_process(option) #10,50,300 -> 10,50, 256
            option = option.max(1)[0]   #10, 256
            logit = context.matmul(self.W).matmul(option.transpose(1,0)).sum(-1)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

class DecAttn(torch.nn.Module):
    """docstring for DecAttn"""
    def __init__(self):
        super(DecAttn, self).__init__()
        self.lstm = torch.nn.LSTM(300, 256, 1, batch_first=True)
        self.softmax = torch.nn.Softmax(dim=1)
        self.attn = torch.nn.Linear(256, 256)
        self.final = torch.nn.LSTM(256+256, 256, 1, batch_first=True)

    def forward(self, context, option):
        """
        context: 10,30, 256
        option: 10, 50, 256
        attn_weight: 10, 50, 30
        attn_context_sum: 10, 50, 256
        """
        option, __ = self.lstm(option)
        attn_context = self.attn(context)
        attn_weight = self.softmax(option.bmm(attn_context.transpose(1,2)))
        attn_context_sum = attn_weight.bmm(context)

        complete = torch.cat([option, attn_context_sum], 2)
        final, __ = self.final(complete)
        return final

# class RnnAttentionNet(torch.nn.Module):
#     def __init__(self, dim_embeddings):
#         super(RnnAttentionNet, self).__init__()
#         self.C_process = torch.nn.LSTM(300, 256, 2, batch_first=True)
#         self.P_process = DecAttn()
#         self.W = torch.nn.Parameter(torch.rand(256, 256), requires_grad=True)

#     def forward(self, context, options):
#         ## context 10, 30, 300; options 10, 100, 50, 300
#         context_raw, hidden = self.C_process(context) ## 10,30,256
#         context = context_raw.max(1)[0] # 10,256
#         logits = []
#         for i, option in enumerate(options.transpose(1, 0)):  # 100, 10, 50, 300
#             option = self.P_process(context_raw, option) #10,50,300 -> 10,50, 256
#             option = option.max(1)[0]   #10, 256
#             logit = context.matmul(self.W).matmul(option.transpose(1,0)).sum(-1)
#             logits.append(logit)
#         logits = torch.stack(logits, 1)
#         return logits

#     def save(self, filepath):
#         torch.save(self.state_dict(), filepath)
        
class RnnAttentionNet(torch.nn.Module):
    def __init__(self, dim_embeddings):
        super(RnnAttentionNet, self).__init__()
        self.C_process = torch.nn.LSTM(256, 256, 1, batch_first=True)
        self.context_word = torch.nn.LSTM(300, 256, 1, batch_first=True)
        self.P_process = DecAttn()
        self.W = torch.nn.Parameter(data=torch.rand(256, 256), requires_grad=True)

    def forward(self, context, options):

        ## context 10, 14, 30, 300
        context_vec = []
        for utterance in context.transpose(1,0):  # 10, 30, 300
            utter_vec, __ = self.context_word(utterance) # 10, 30, 256
            utter_vec = utter_vec.max(1)[0]         # 10, 256
            context_vec.append(utter_vec)           
        context_vec = torch.stack(context_vec, 1)   # 10, 14, 256
        context_raw, __ = self.C_process(context_vec)   # 10, 14, 256

        ## context 10, 30, 300; options 10, 100, 50, 300
        # context, __ = self.C_process(context) ## 10,30,256
        context = context_raw.max(1)[0] # 10,256
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):  # 100, 10, 50, 300
            option = self.P_process(context_raw, option) #10,50,300 -> 10,50, 256
            option = option.max(1)[0]   #10, 256
            logit = context.matmul(self.W).matmul(option.transpose(1,0)).sum(-1)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)