import torch

class SP_Distill(torch.nn.Module):
    def __init__(self, bit_list=[16,32,64]):
        super(SP_Distill, self).__init__()
        self.bit_list = bit_list

    def forward(self, X_list):
        losses = []
        for idx in range(len(X_list)-1):
            distill_loss = self.similarity_loss(X_list[idx], X_list[idx+1].detach())
            losses.append(distill_loss)
        losses = torch.stack(losses)
        return losses
    
    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        G_s = torch.mm(f_s, torch.t(f_s))
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        G_t = torch.nn.functional.normalize(G_t)
        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss[0]

class Ours_Distill_all(torch.nn.Module):
    def __init__(self, bit_list=[8,16,32,64,128]):
        super(Ours_Distill_all, self).__init__()
        self.bit_list = bit_list

    def forward(self, outputs):
        distill_losses = [] 
        preserve_losses = []
        # short_c = outputs[1]
        # short_l = self.bit_list[1]
        # long_c = outputs[2]
        for idx, short_bit in enumerate(self.bit_list):
            if idx == len(self.bit_list)-1:
                break
            short_c = outputs[idx]
            long_c = outputs[idx+1]
            distill_loss = self.similarity_loss(short_c, long_c.detach())
            preserve_loss = self.similarity_loss(torch.cat((short_c, long_c[:,short_bit:].detach()), dim=1), long_c.detach())
            distill_losses.append(distill_loss)
            preserve_losses.append(preserve_loss)
        distill_losses = torch.stack(distill_losses).sum()
        preserve_losses = torch.stack(preserve_losses).sum()
        return distill_losses, preserve_losses
    
    def similarity_loss(self, f_s, f_t):
        # 这里采用的是SP方法
        bsz = f_s.shape[0]
        G_s = torch.mm(f_s, torch.t(f_s))
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        G_t = torch.nn.functional.normalize(G_t)
        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss[0]
    
class Ours_Distill(torch.nn.Module):
    def __init__(self, bit_list=[8,16,32,64,128]):
        super(Ours_Distill, self).__init__()
        self.bit_list = bit_list

    def forward(self, outputs):
        distill_losses = [] 
        preserve_losses = []
        # short_c = outputs[1]
        # short_l = self.bit_list[1]
        # long_c = outputs[2]
        for idx, short_bit in enumerate(self.bit_list):
            if idx == len(self.bit_list)-1:
                break
            short_c = outputs[idx]
            long_c = outputs[idx+1]
            distill_loss = self.similarity_loss(short_c, long_c.detach())
            preserve_loss = self.similarity_loss(torch.cat((short_c, long_c[:,short_bit:].detach()), dim=1), long_c.detach())
            distill_losses.append(distill_loss)
            preserve_losses.append(preserve_loss)
        return distill_losses
    
    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        G_s = torch.mm(f_s, torch.t(f_s))
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        G_t = torch.nn.functional.normalize(G_t)
        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss[0]
    
class Ours_Distill_per_one(torch.nn.Module):
    def __init__(self):
        super(Ours_Distill_per_one, self).__init__()

    def forward(self, outputs):
        # short_c = outputs[1]
        # short_l = self.bit_list[1]
        # long_c = outputs[2]
        (short, long) = outputs
        distill_loss = self.similarity_loss(short, long.detach())
        return distill_loss
    
    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        G_s = torch.mm(f_s, torch.t(f_s))
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        G_t = torch.nn.functional.normalize(G_t)
        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss[0]