import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loaders import Scaler


class SeqGen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg['device']
        self.usr_type_num = cfg['usr_type_num']
        self.usr_emb_dim = cfg['usr_emb_dim']
        self.usr_feature_dim = cfg['usr_feature_dim']
        self.category_num = cfg['category_num']
        self.category_emb_dim = cfg['category_emb_dim']
        self.hour_emb_dim = cfg['hour_emb_dim']
        self.week_emb_dim = cfg['week_emb_dim']
        self.dur_weight = cfg['dur_weight']
        self.dist_weight = cfg['dist_weight']

        # 生成阶段 de-normalize
        self.scaler = Scaler(cfg['dataset'])

        self.usr_emb = nn.Embedding(self.usr_type_num, self.usr_emb_dim)
        self.category_emb = nn.Embedding(self.category_num + 1, self.category_emb_dim)
        self.hour_emb = nn.Embedding(24, self.hour_emb_dim)
        self.week_emb = nn.Embedding(7, self.week_emb_dim)

        self.input_size = (self.usr_emb_dim + self.usr_feature_dim
                           + self.category_emb_dim + self.hour_emb_dim + self.week_emb_dim)
        self.hidden_size = cfg['hidden_size']
        self.rnn_layers = cfg['rnn_layers']
        self.dropout = cfg['dropout']
        self.seq_model = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.rnn_layers,
            dropout=self.dropout, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.category_num),
            # nn.Softmax(dim=-1)
            nn.LogSoftmax(dim=-1)
        )
        self.dur_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, 1),
            # nn.Softplus()
            nn.Sigmoid()
        )
        self.dist_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, 1),
            # nn.Softplus()
            nn.Sigmoid()
        )

    def forward(self, batch):
        usr_type = batch['usr_type'].to(self.device)
        usr_feat = batch['usr_feature'].to(self.device)
        category = batch['x_event'].to(self.device)
        hour = batch['hour'].to(self.device)
        week = batch['weekday'].to(self.device)
        seq_len = category.shape[1]

        usr_type = self.usr_emb(usr_type).unsqueeze(1).repeat(1, seq_len, 1)
        usr_feat = usr_feat.unsqueeze(1).repeat(1, seq_len, 1)
        category = self.category_emb(category)
        hour = self.hour_emb(hour)
        week = self.week_emb(week)
        input_emb = torch.cat([usr_type, usr_feat, category, hour, week], dim=-1)
        output_emb, (h, c) = self.seq_model(input_emb)
        log_prob = self.classifier(output_emb)
        dur_pred = self.dur_mlp(output_emb)
        dist_pred = self.dist_mlp(output_emb)
        return log_prob, dur_pred, dist_pred

    def masked_cross_entropy(self, log_prob, target, mask):
        mask = ~mask.view(-1)
        # log_prob = torch.log(log_prob).view(-1, self.category_num)[mask]
        log_prob = log_prob.view(-1, self.category_num)[mask]
        target = target.view(-1)[mask]
        return F.nll_loss(log_prob, target)

    @staticmethod
    def masked_mse(pred, target, mask):
        mask = ~mask.view(-1)
        pred = pred.view(-1)[mask]
        target = target.view(-1)[mask]
        return F.mse_loss(pred, target)

    def calc_loss(self, batch):
        mask = batch['mask'].to(self.device)
        category_target = batch['y_event'].to(self.device)
        dur_target = batch['dur'].to(self.device)
        dist_target = batch['dist'].to(self.device)
        category_logit, dur_pred, dist_pred = self.forward(batch)
        clf_loss = self.masked_cross_entropy(category_logit, category_target, mask)
        dur_loss = self.masked_mse(dur_pred, dur_target, mask)
        dist_loss = self.masked_mse(dist_pred, dist_target, mask)
        return {
            'clf_loss': clf_loss,
            'dur_loss': dur_loss,
            'dist_loss': dist_loss,
            'total_loss': clf_loss + self.dur_weight * dur_loss + self.dist_weight * dist_loss,
        }

    def generate(self, batch):
        """
        :param batch:
            'seq_len',
            Tensors 'time' (bs,), 'usr_type' (bs,), 'usr_feature' (bs, feat_dim),
                'x_event' (bs, 1), 'hour' (bs, 1), 'weekday' (bs, 1)
        """
        for key in ['time', 'usr_type', 'usr_feature', 'x_event', 'hour', 'weekday']:
            batch[key] = batch[key].to(self.device)
        bs = batch['x_event'].shape[0]
        result = {
            'event': [batch['x_event'].squeeze(1)],
            'time': [batch['time']],
            'dist': [torch.zeros(bs).to(self.device)]
        }
        for i in range(batch['seq_len'] - 1):
            log_prob, dur, dist = self.forward(batch)
            step_prob = torch.exp(log_prob[:, -1, :])
            step_dur = dur[:, -1, :]
            step_dist = dist[:, -1, :]

            # 反归一化
            step_dur = self.scaler.denormalize_dur(step_dur)
            step_dist = self.scaler.denormalize_dist(step_dist)

            event = torch.multinomial(step_prob, num_samples=1)
            result['event'].append(event.squeeze(1))
            batch['x_event'] = torch.cat([batch['x_event'], event], dim=-1)

            result['dist'].append(step_dist.squeeze(1))

            prev_time = result['time'][-1]
            cur_time = prev_time + step_dur.squeeze(1)
            result['time'].append(cur_time)

            cur_hour = (cur_time % 24).long()
            prev_weekday = batch['weekday'][:, -1]
            cur_day = torch.div(cur_time, 24, rounding_mode='floor').long()
            prev_day = torch.div(prev_time, 24, rounding_mode='floor').long()
            cur_weekday = (prev_weekday + (cur_day - prev_day)) % 7

            batch['hour'] = torch.cat([batch['hour'], cur_hour.unsqueeze(-1)], dim=-1)
            batch['weekday'] = torch.cat([batch['weekday'], cur_weekday.unsqueeze(-1)], dim=-1)

        result['event'] = torch.stack(result['event'], dim=1)
        result['time'] = torch.stack(result['time'], dim=1)
        result['dist'] = torch.stack(result['dist'], dim=1)
        return result


class ActTrajGen(nn.Module):
    def __init__(self, cfg):
        """
        不预测空间距离, 直接生成活动序列
        """
        super().__init__()
        self.device = cfg['device']
        self.usr_type_num = cfg['usr_type_num']
        self.usr_emb_dim = cfg['usr_emb_dim']
        self.usr_feature_dim = cfg['usr_feature_dim']
        self.act_num = cfg['act_num']
        self.act_emb_dim = cfg['act_emb_dim']
        self.time_slot_num = cfg['time_slot_num']
        self.time_emb_dim = cfg['time_emb_dim']
        self.week_emb_dim = cfg['week_emb_dim']
        self.time_weight = cfg['time_weight']

        self.usr_emb = nn.Embedding(self.usr_type_num, self.usr_emb_dim)
        self.act_emb = nn.Embedding(self.act_num + 1, self.act_emb_dim)
        self.time_emb = nn.Embedding(self.time_slot_num + 1, self.time_emb_dim)
        self.week_emb = nn.Embedding(7 + 1, self.week_emb_dim)

        self.input_size = (self.usr_emb_dim + self.usr_feature_dim
                           + self.act_emb_dim + self.time_emb_dim + self.week_emb_dim)
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
        self.time_clf = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.time_slot_num),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, batch):
        usr_type = batch['usr_type'].to(self.device)
        weekday = batch['weekday'].to(self.device)
        usr_feat = batch['usr_feature'].to(self.device)
        x_act = batch['x_act'].to(self.device)
        time_slot = batch['x_time'].to(self.device)
        seq_len = x_act.shape[1]

        usr_type = self.usr_emb(usr_type).unsqueeze(1).repeat(1, seq_len, 1)
        weekday = self.week_emb(weekday).unsqueeze(1).repeat(1, seq_len, 1)
        usr_feat = usr_feat.unsqueeze(1).repeat(1, seq_len, 1)

        x_act = self.act_emb(x_act)
        time_slot = self.time_emb(time_slot)
        input_emb = torch.cat([usr_type, usr_feat, x_act, time_slot, weekday], dim=-1)
        output_emb, (h, c) = self.seq_model(input_emb)
        act_log_prob = self.act_clf(output_emb)
        time_log_prob = self.time_clf(output_emb)
        return act_log_prob, time_log_prob

    def masked_act_loss(self, act_log_prob, target, mask):
        mask = ~mask.view(-1)
        act_log_prob = act_log_prob.view(-1, self.act_num)[mask]
        target = target.view(-1)[mask]
        return F.nll_loss(act_log_prob, target)

    def masked_time_loss(self, time_log_prob, target, mask):
        mask = ~mask.view(-1)
        time_log_prob = time_log_prob.view(-1, self.time_slot_num)[mask]
        target = target.view(-1)[mask]
        return F.nll_loss(time_log_prob, target)

    def calc_loss(self, batch):
        mask = batch['mask'].to(self.device)
        y_act = batch['y_act'].to(self.device)
        y_time = batch['y_time'].to(self.device)
        act_log_prob, time_log_prob = self.forward(batch)
        act_loss = self.masked_act_loss(act_log_prob=act_log_prob, target=y_act, mask=mask)
        time_loss = self.masked_time_loss(time_log_prob=time_log_prob, target=y_time, mask=mask)
        return {
            'act_loss': act_loss,
            'time_loss': time_loss,
            'total_loss': act_loss + self.time_weight * time_loss
        }

    def sample_time_slot(self, prev_time, time_prob):
        """
        :param prev_time: (batch_size,)
        :param time_prob: (batch_size, time_slot_num)
        :return:
        """
        eps = 1e-8
        batch_time = []
        for batch_idx, prev_slot in enumerate(prev_time.detach().cpu().tolist()):
            # 生成的时候约束下一跳的时间片大于当前点, 对未来时间片的采样概率做归一化
            start_idx = min(self.time_slot_num - 1, prev_slot + 1)
            future_prob = time_prob[batch_idx][start_idx:].detach() + eps
            future_prob = future_prob / future_prob.sum()
            cur_slot = torch.multinomial(future_prob, num_samples=1).cpu().tolist()[0]
            cur_slot += start_idx
            batch_time.append(cur_slot)
        cur_time = torch.LongTensor(batch_time).unsqueeze(1).to(self.device)
        return cur_time

    def generate(self, batch):
        """
        :param batch:
            'seq_len',
            Tensors 'usr_type' (bs,), 'usr_feature' (bs, feat_dim),
                'x_act' (bs, 1), 'x_time' (bs, 1), 'weekday' (bs, 1)
        """
        for key in ['usr_type', 'usr_feature', 'x_act', 'x_time', 'weekday']:
            batch[key] = batch[key].to(self.device)
        gen_act = [batch['x_act'].squeeze(1)]
        gen_time = [batch['x_time'].squeeze(1)]
        for i in range(batch['seq_len'] - 1):
            act_log_prob, time_log_prob = self.forward(batch)
            act_prob = torch.exp(act_log_prob[:, -1, :])
            time_prob = torch.exp(time_log_prob[:, -1, :])
            act = torch.multinomial(act_prob, num_samples=1)
            time_slot = self.sample_time_slot(prev_time=gen_time[-1], time_prob=time_prob)
            gen_act.append(act.squeeze(1))
            gen_time.append(time_slot.squeeze(1))
            batch['x_act'] = torch.cat([batch['x_act'], act], dim=-1)
            batch['x_time'] = torch.cat([batch['x_time'], time_slot], dim=-1)
        return {
            'act': torch.stack(gen_act, dim=-1),
            'time': torch.stack(gen_time, dim=-1)
        }

