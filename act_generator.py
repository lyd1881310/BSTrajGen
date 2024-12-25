import torch
import torch.nn as nn
import torch.nn.functional as F


class ActTrajGen(nn.Module):
    def __init__(self, cfg):
        """
        不预测空间距离和时间, 直接建模活动序列的概率分布
        """
        super().__init__()
        self.device = cfg['device']
        self.usr_type_num = cfg['usr_type_num']
        self.usr_emb_dim = cfg['usr_emb_dim']
        self.usr_feature_dim = cfg['usr_feature_dim']
        self.act_num = cfg['act_num']
        self.act_emb_dim = cfg['act_emb_dim']
        self.week_emb_dim = cfg['week_emb_dim']

        self.usr_emb = nn.Embedding(self.usr_type_num, self.usr_emb_dim)
        self.act_emb = nn.Embedding(self.act_num + 1, self.act_emb_dim)
        self.week_emb = nn.Embedding(7 + 1, self.week_emb_dim)

        self.input_size = self.usr_emb_dim + self.usr_feature_dim + self.act_emb_dim + self.week_emb_dim
        self.hidden_size = cfg['hidden_size']
        self.rnn_layers = cfg['rnn_layers']
        self.dropout = cfg['dropout']
        self.seq_model = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.rnn_layers,
            dropout=self.dropout, batch_first=True
        )
        self.act_clf = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.act_num),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, batch):
        usr_type = batch['usr_type'].to(self.device)
        usr_feat = batch['usr_feature'].to(self.device)
        weekday = batch['weekday'].to(self.device)
        x_act = batch['x_act'].to(self.device)
        seq_len = x_act.shape[1]

        usr_type = self.usr_emb(usr_type).unsqueeze(1).repeat(1, seq_len, 1)
        weekday = self.week_emb(weekday).unsqueeze(1).repeat(1, seq_len, 1)
        usr_feat = usr_feat.unsqueeze(1).repeat(1, seq_len, 1)

        x_act = self.act_emb(x_act)
        input_emb = torch.cat([usr_type, usr_feat, x_act, weekday], dim=-1)
        output_emb, (h, c) = self.seq_model(input_emb)
        act_log_prob = self.act_clf(output_emb)
        return act_log_prob

    def masked_act_loss(self, act_log_prob, target, mask):
        mask = ~mask.view(-1)
        act_log_prob = act_log_prob.view(-1, self.act_num)[mask]
        target = target.view(-1)[mask]
        return F.nll_loss(act_log_prob, target)

    def calc_loss(self, batch):
        mask = batch['mask'].to(self.device)
        y_act = batch['y_act'].to(self.device)
        act_log_prob = self.forward(batch)
        loss = self.masked_act_loss(act_log_prob=act_log_prob, target=y_act, mask=mask)
        return loss

    def generate(self, batch):
        """
        :param batch:
            'seq_len',
            Tensors 'usr_type' (bs,), 'usr_feature' (bs, feat_dim),
                'x_act' (bs, 1), 'weekday' (bs, 1)
        """
        for key in ['usr_type', 'usr_feature', 'x_act', 'weekday']:
            batch[key] = batch[key].to(self.device)
        gen_act = [batch['x_act'].squeeze(1)]
        for i in range(batch['seq_len'] - 1):
            act_log_prob = self.forward(batch)
            act_prob = torch.exp(act_log_prob[:, -1, :])
            act = torch.multinomial(act_prob, num_samples=1)
            gen_act.append(act.squeeze(1))
            batch['x_act'] = torch.cat([batch['x_act'], act], dim=-1)
        return torch.stack(gen_act, dim=-1)

