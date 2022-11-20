import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
def generate(N,M,L):
    PWM = np.random.dirichlet(np.ones(4), size=(M,))
    PWM = torch.from_numpy(PWM)
    bg = torch.ones(L-M, 4) / 4
    positions = torch.randint(low=0, high=L-M+1, size=(N,))[:,None]
    positions_aug = torch.cat([positions + i for i in range(M)], dim=-1)

    prob = torch.ones(size=(N, L, 4), dtype=torch.float64) / 4.0
    prob = prob.scatter(dim=1, index=positions_aug.unsqueeze(-1).repeat(1,1,4), src=PWM.unsqueeze(0).repeat(N,1,1)).contiguous()
    seq = torch.multinomial(prob.reshape(shape=(N*L, 4)), num_samples=1).reshape(shape=(N,L)) ##the DNA sequence
    return seq, PWM, positions

def MEME(N,M,L,seq,maxiter=200,num_candidate=10):
    Zs = torch.gather(seq[:,None,:].repeat(1,L-M+1,1), dim=2, index=torch.cat([torch.arange(start=i, end=L-M+i+1)[:,None] for i in range(M)], dim=-1).unsqueeze(0).repeat(N,1,1))
    onehot_Zs = torch.eye(4, dtype=torch.float64)[Zs]

    # model = np.random.dirichlet(np.ones(4), size=(M,))
    # model = torch.from_numpy(model)
    models, logp = [], []
    for i in range(num_candidate):
        model = torch.rand(size=(M,4), dtype=torch.float64)
        model = model / torch.sum(model, dim=-1, keepdim=True)
        p = torch.diagonal(onehot_Zs @ model.T, dim1=-2, dim2=-1).prod(-1)
        logp.append(p.sum(-1).log().sum().item())
        models.append(model)
    idx = np.argmax(logp)
    model = models[idx]

    LLs = []
    for i in range(maxiter):
        p = torch.diagonal(onehot_Zs @ model.T, dim1=-2, dim2=-1).prod(-1)
        LLs.append(p.sum(-1).log().sum().item())
        p = p / torch.sum(p, dim=-1, keepdim=True)
        model = (p[:,:,None,None].repeat(1,1,M,4) * onehot_Zs).mean(dim=(0,1))
        model = model / torch.sum(model, dim=-1,keepdim=True)
    return model, LLs

def SMEME(N,M,L,seq,maxiter=200,num_candidate=10, vr = True, alpha=0.001):
    emb = torch.eye(4, dtype=torch.float64)
    Zs = torch.gather(seq[:,None,:].repeat(1,L-M+1,1), dim=2, index=torch.cat([torch.arange(start=i, end=L-M+i+1)[:,None] for i in range(M)], dim=-1).unsqueeze(0).repeat(N,1,1))
    onehot_Zs = emb[Zs]
    
    # model = np.random.dirichlet(np.ones(4), size=(M,))
    # model = torch.from_numpy(model)
    models, logp = [], []
    for i in range(num_candidate):
        model = torch.rand(size=(M,4), dtype=torch.float64)
        model = model / torch.sum(model, dim=-1, keepdim=True)
        p = torch.diagonal(onehot_Zs @ model.T, dim1=-2, dim2=-1).prod(-1)
        logp.append(p.sum(-1).log().sum().item())
        models.append(model)
    idx = np.argmax(logp)
    model = models[idx]

    LLs = []
    p = torch.diagonal(onehot_Zs @ model.T, dim1=-2, dim2=-1).prod(-1)
    LLs.append(p.sum(-1).log().sum().item())
    p = p / torch.sum(p, dim=-1, keepdim=True)
    model = (p[:,:,None,None].repeat(1,1,M,4) * onehot_Zs).mean(dim=(0,1))
    norm_model = model / torch.sum(model, dim=-1,keepdim=True)

    for i in range(maxiter * N):
        if i % N == 0:
            p = torch.diagonal(onehot_Zs @ norm_model.T, dim1=-2, dim2=-1).prod(-1)
            LLs.append(p.sum(-1).log().sum().item())
        if vr and i % N == 0:
            norm_model_c = norm_model.clone()
            p = torch.diagonal(onehot_Zs @ norm_model_c.T, dim1=-2, dim2=-1).prod(-1)
            p = p / torch.sum(p, dim=-1, keepdim=True)
            model_c = (p[:,:,None,None].repeat(1,1,M,4) * onehot_Zs).mean(dim=(0,1))
        idx = torch.randint(low=0, high=N, size=(1,))[0].item()
        neo_Zs = Zs[idx]
        neo_onehot_Zs = emb[neo_Zs]
        p = torch.diagonal(neo_onehot_Zs @ norm_model.T, dim1=-2, dim2=-1).prod(-1)
        p /= p.sum()
        neo_model = (p[:,None,None].repeat(1,M,4) * neo_onehot_Zs).mean(dim=0)
        if vr:
            p = torch.diagonal(neo_onehot_Zs @ norm_model_c.T, dim1=-2, dim2=-1).prod(-1)
            p /= p.sum()
            neo_model_c = (p[:,None,None].repeat(1,M,4) * neo_onehot_Zs).mean(dim=0)
            model = (model * (1-alpha) + (neo_model-neo_model_c+model_c) * alpha).clamp(1e-8)
        else:
            model = model * (1-alpha) + neo_model * alpha
        norm_model = model / torch.sum(model, dim=-1,keepdim=True)
    return norm_model, LLs

torch.manual_seed(2022)
np.random.seed(2022)
for N in [600, 6000, 60000]:
    for M, L in [(8,24), (12, 36), (16, 48)]:
        ##number of samples, length of motif, length of DNA sequence
        seq, PWM, positions = generate(N,M,L)
        for method in ['MEME', 'SMEME', 'SMEME-VR']:
            if method == 'MEME':
                model, LLs = MEME(N,M,L,seq)
            elif method == 'SMEME':
                model, LLs = SMEME(N,M,L,seq, vr=False)
            elif method == 'SMEME-VR':
                model, LLs = SMEME(N,M,L,seq,vr=True)
            with open('results/{}/LLs_{}_{}_{}.pkl'.format(method, N,M,L), 'wb') as f:
                pickle.dump(LLs, f)
            torch.save(model, 'results/{}/PWM_{}_{}_{}.pt'.format(method, N,M,L))