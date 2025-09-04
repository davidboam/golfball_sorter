#!/usr/bin/env python3
import os, json, time, glob, argparse
from pathlib import Path
from typing import List
import numpy as np, pandas as pd, cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
import timm

GRADE_ORDER = ["Pearl","A","B","C","D"]
GRADE_TO_IDX = {g:i for i,g in enumerate(GRADE_ORDER)}

def imread_rgb(path: str, target_size: int = 224):
    img = cv2.imread(path, cv2.IMREAD_COLOR); 
    if img is None: raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w = img.shape[:2]; m=min(h,w); y0=(h-m)//2; x0=(w-m)//2
    img = img[y0:y0+m, x0:x0+m]; 
    if target_size and (img.shape[0]!=target_size or img.shape[1]!=target_size):
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return img

def to_tensor(img: np.ndarray) -> torch.Tensor:
    img = img.astype(np.float32)/255.0; img = img.transpose(2,0,1)
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)[:,None,None]
    std  = np.array([0.229,0.224,0.225], dtype=np.float32)[:,None,None]
    img = (img-mean)/std; return torch.from_numpy(img)

def circle_mask(h,w,cx,cy,r):
    yy,xx = np.ogrid[:h,:w]; dist=(yy-cy)**2+(xx-cx)**2; return dist<=r*r

def estimate_ball_lab_features(rgb: np.ndarray):
    h,w = rgb.shape[:2]; r=0.45*min(h,w); cy, cx = h/2.0, w/2.0
    mask = circle_mask(h,w,cx,cy,r)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR); lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[...,0].astype(np.float32)*(100.0/255.0); b = lab[...,2].astype(np.float32)-128.0
    return float(np.median(L[mask])), float(np.median(b[mask]))

class MultiViewBallDataset(Dataset):
    def __init__(self, root: str, split: str, n_views:int=4, img_size:int=224, label_csv:str="labels.csv", allowed_brands:List[str]=None):
        self.root = Path(root); self.images_root = self.root / "images"
        df = pd.read_csv(self.root / label_csv); self.df = df[df["split"].astype(str)==str(split)].copy()
        brands = sorted([str(b) for b in self.df["brand"].fillna("Other").unique().tolist()])
        if allowed_brands is not None: brands = allowed_brands
        if "Other" not in brands: brands.append("Other")
        self.brands = brands; self.brand_to_idx = {b:i for i,b in enumerate(brands)}
        self.samples = []
        for _,row in self.df.iterrows():
            ball_id = str(row["ball_id"]); grade=str(row["grade"]); brand=str(row.get("brand","Other") or "Other")
            folder = self.images_root / ball_id
            views = sorted(glob.glob(str(folder/"*.jpg")))+sorted(glob.glob(str(folder/"*.png")))
            if len(views)==0: continue
            self.samples.append({"ball_id":ball_id,"views":views,"grade":grade,"brand":brand if brand in self.brand_to_idx else "Other"})
        self.split = split; self.n_views=n_views; self.img_size=img_size
        if len(self.samples)==0: raise ValueError(f"No image folders under {self.images_root}")

    def __len__(self): return len(self.samples)
    def _select_views(self, paths):
        if len(paths)>=self.n_views: return paths[:self.n_views]
        reps=[]; i=0
        while len(reps)<self.n_views: reps.append(paths[i%len(paths)]); i+=1
        return reps

    def __getitem__(self, idx:int):
        s=self.samples[idx]; paths=self._select_views(s["views"]); imgs=[imread_rgb(p,self.img_size) for p in paths]
        Ls,bs=[],[]
        for rgb in imgs:
            Lm,bm=estimate_ball_lab_features(rgb); Ls.append(Lm); bs.append(bm)
        aux=np.array([np.mean(Ls,dtype=np.float32), np.mean(bs,dtype=np.float32)], dtype=np.float32)
        views_t = torch.stack([to_tensor(im) for im in imgs], dim=0)
        grade_idx = GRADE_TO_IDX[s["grade"]]; brand_idx = self.brand_to_idx.get(s["brand"], self.brand_to_idx["Other"])
        return views_t, torch.from_numpy(aux), {"grade":torch.tensor(grade_idx), "brand":torch.tensor(brand_idx), "ball_id":s["ball_id"]}

class AttentionPool(nn.Module):
    def __init__(self, emb_dim:int, att_dim:int=128):
        super().__init__(); self.att_w=nn.Linear(emb_dim, att_dim); self.att_v=nn.Linear(att_dim,1)
    def forward(self, feats):
        a=torch.tanh(self.att_w(feats)); a=self.att_v(a).squeeze(-1); w=torch.softmax(a,dim=1)
        fused=(feats*w.unsqueeze(-1)).sum(dim=1); return fused,w

class MultiViewBallNet(nn.Module):
    def __init__(self, backbone_name="mobilenetv3_small_100", n_grades=5, n_brands=11, emb_dim=256, aux_dim=2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool='avg')
        with torch.no_grad(): fdim=self.backbone(torch.zeros(1,3,224,224)).shape[-1]
        self.proj=nn.Linear(fdim, emb_dim); self.pool=AttentionPool(emb_dim,128)
        self.grade_head=nn.Sequential(nn.Linear(emb_dim+aux_dim,128), nn.ReLU(inplace=True), nn.Linear(128,n_grades-1))
        self.brand_head=nn.Linear(emb_dim, n_brands); self.n_grades=n_grades

    def encode_views(self, views):
        B,N,C,H,W=views.shape; x=views.view(B*N,C,H,W); f=self.proj(self.backbone(x)); f=f.view(B,N,-1)
        fused, w = self.pool(f); return fused, w
    def forward(self, views, aux):
        emb, attw = self.encode_views(views); glog=self.grade_head(torch.cat([emb,aux], dim=1)); blog=self.brand_head(emb)
        return glog, blog, attw

def grade_to_ordinal_targets(y, n_grades): 
    B=y.shape[0]; K=n_grades; t=torch.zeros(B,K-1, dtype=torch.float32, device=y.device)
    for k in range(K-1): t[:,k]=(y>k).float(); return t

def ordinal_logits_to_probs(logits):
    sig=torch.sigmoid(logits); B,K1=sig.shape; K=K1+1; p=torch.zeros(B,K,device=logits.device,dtype=logits.dtype)
    p[:,0]=1.0-sig[:,0]
    for k in range(1,K-1): p[:,k]=sig[:,k-1]-sig[:,k]
    p[:,K-1]=sig[:,K-2]; return p

class OrdinalLoss(nn.Module):
    def __init__(self, n_grades:int, pos_weight:float=1.0):
        super().__init__(); self.n_grades=n_grades; self.bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    def forward(self, logits, y): return self.bce(logits, grade_to_ordinal_targets(y, self.n_grades))

def main():
    print("This trimmed trainer keeps the same model API. For full training loop, use the earlier extended script.")

if __name__ == "__main__":
    main()
