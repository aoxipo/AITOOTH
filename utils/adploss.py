
class AdpLoss():
    def __init__(self, k_of_BNM = 0.5, pre = 0.5, p = 0.5):
        self.k_of_BNM = k_of_BNM
        self.pre = pre
        self.p = p
        self.k_list = []
        self.p_list = []
        print("start paramete k:{}, pre:{}, p:{}".format(self.k_of_BNM, self.pre, self.p))
    
    def update(self, loss):
        self.k_of_BNM = (self.p+0.018)/(self.p+0.018+loss.item())
        self.pre = self.p
        self.p = (1-self.k_of_BNM)*self.pre
        
        self.k_list.append(self.k_of_BNM)
        self.p_list.append(self.p)

    def cal_score(self, loss_other):
        target_softmax = loss_other.view(loss_other.shape[2],loss_other.shape[3])
        transfer_loss = - torch.norm(target_softmax,'nuc')/loss_other.shape[0]
        return transfer_loss
        
    def __call__(self, loss_main, loss_other):
        if(abs(loss_main) <= 0.1):
            transfer_loss = 0
        else:
            transfer_loss = loss_other # cal_score(loss_other)

        loss = (1-self.k_of_BNM)*loss_main + self.k_of_BNM * transfer_loss
        if(loss <= 0):
            loss = loss_main 
        self.update(loss)
        return loss