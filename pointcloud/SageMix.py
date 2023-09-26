import torch
from emd_ import emd_module

class SageMix:
    def __init__(self, args, num_class=40):
        self.num_class = num_class
        self.EMD = emd_module.emdModule()
        self.sigma = args.sigma
        self.beta = torch.distributions.beta.Beta(torch.tensor([args.theta]), torch.tensor([args.theta]))

    
    def mix(self, xyz, label, saliency=None):
        """
        Args:
            xyz (B,N,3)
            label (B)
            saliency (B,N): Defaults to None.
        """        
        B, N, _ = xyz.shape
        idxs = torch.randperm(B)#(B)
        # print("xyz:",xyz.shape)#(32,1024,3)
        # print("label:",label.shape)
        # print("saliency:",saliency.shape)

        
        #Optimal assignment in Eq.(3)s
        perm = xyz[idxs]
        # print("perm:",perm.shape)#(32,1024,3)
        # print("saliency:",saliency.shape)
        # print("sasa:",saliency[idxs].shape)#(32,1024)
        
        _, ass = self.EMD(xyz, perm, 0.005, 500) # mapping
        ass = ass.long()#(32,1024)
        perm_new = torch.zeros_like(perm).cuda()#(32,1024,3)
        perm_saliency = torch.zeros_like(saliency).cuda()#(32,1024)
        # print("sss:",saliency[6][ass[6]].shape)#(1024)
        
        for i in range(B):
            perm_new[i] = perm[i][ass[i]]#(1024,3) 存放的是混合样本中各个点的重新排列方式。
            perm_saliency[i] = saliency[idxs][i][ass[i]]#(1024)存放的是混合样本中各个点的显著性值，用于计算样本的权重
        
        #####
        # Saliency-guided sequential sampling
        #####
        #Eq.(4) in the main paper
        saliency = saliency/saliency.sum(-1, keepdim=True)
        anc_idx = torch.multinomial(saliency, 1, replacement=True)#(32,1)
        anchor_ori = xyz[torch.arange(B), anc_idx[:,0]]#(32,3)
        
        #cal distance and reweighting saliency map for Eq.(5) in the main paper
        sub = perm_new - anchor_ori[:,None,:]#(32,1024,3)
        dist = ((sub) ** 2).sum(2).sqrt()#(32,1024)
        perm_saliency = perm_saliency * dist
        perm_saliency = perm_saliency/perm_saliency.sum(-1, keepdim=True)
        
        #Eq.(5) in the main paper
        anc_idx2 = torch.multinomial(perm_saliency, 1, replacement=True)#(32,1)
        anchor_perm = perm_new[torch.arange(B),anc_idx2[:,0]]#(32,3)
                
                
        #####
        # Shape-preserving continuous Mixup
        #####
        alpha = self.beta.sample((B,)).cuda()#(32,1)
        sub_ori = xyz - anchor_ori[:,None,:]#(32,1024,3)
        sub_ori = ((sub_ori) ** 2).sum(2).sqrt()
        #Eq.(6) for first sample
        ker_weight_ori = torch.exp(-0.5 * (sub_ori ** 2) / (self.sigma ** 2))  #(M,N)(32,1024)
        sub_perm = perm_new - anchor_perm[:,None,:]#(32,1024,3)
        sub_perm = ((sub_perm) ** 2).sum(2).sqrt()
        #Eq.(6) for second sample
        ker_weight_perm = torch.exp(-0.5 * (sub_perm ** 2) / (self.sigma ** 2))  #(M,N)(32,1024)
        #Eq.(9)
        weight_ori = ker_weight_ori * alpha#(32,1024)
        weight_perm = ker_weight_perm * (1-alpha)#(32,1024)
        weight = (torch.cat([weight_ori[...,None],weight_perm[...,None]],-1)) + 1e-16#(32,1024,2)
        weight = weight/weight.sum(-1)[...,None]
        #Eq.(8) for new sample
        x = weight[:,:,0:1] * xyz + weight[:,:,1:] * perm_new
        
        #Eq.(8) for new label
        target = weight.sum(1)
        target = target / target.sum(-1, keepdim=True)
        label_onehot = torch.zeros(B, self.num_class).cuda().scatter(1, label.view(-1, 1), 1)
        label_perm_onehot = label_onehot[idxs]
        label = target[:, 0, None] * label_onehot + target[:, 1, None] * label_perm_onehot 
        
        return x, label

if __name__ == "__main__":
    import argparse
    import torch
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    
    parser.add_argument('--sigma', type=float, default=-1) 
    parser.add_argument('--theta', type=float, default=0.2) 
    args = parser.parse_args()

    device = torch.device("cuda")
    data = torch.Tensor(3,128,3).to(device)
    print("data:",type(data))
    label = torch.Tensor(3).to(device).squeeze()
    saliency=torch.Tensor(3,128).to(device)
    
    sagemix = SageMix(args, 4)
    data, label = sagemix.mix(data, label, saliency)
    print("data:",data.shape)
    