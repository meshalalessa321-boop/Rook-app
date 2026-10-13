#!/usr/bin/env python3
"""Rook Next: random-init byte Transformer. Dataset: OpenAssistant/oasst1, CC BY 4.0."""
import os,json,random,time,argparse
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
V=259; PAD=256; BOS=257; EOS=258
def data(seed):
 from datasets import load_dataset
 ds=load_dataset("OpenAssistant/oasst1",split="train"); rows={}
 for r in ds:
  if r.get("deleted") or not r.get("text") or r.get("role") not in ("prompter","assistant"): continue
  rows[r["message_id"]]={"id":r["message_id"],"parent":r.get("parent_id"),"role":"User" if r["role"]=="prompter" else "Assistant","text":r["text"].strip()}
 kids=defaultdict(list)
 for k,r in rows.items():
  if r["parent"] in rows:kids[r["parent"]].append(k)
 roots=[k for k,r in rows.items() if not r["parent"] or r["parent"] not in rows]; ex=[]
 for root in roots:
  stack=[(root,[])]
  while stack:
   node,path=stack.pop(); r=rows[node]
   if r["role"]=="Assistant" and path and path[-1]["role"]=="User":
    p="".join(f'{m["role"]}: {m["text"]}\n' for m in path)+"Assistant:"
    a=" "+r["text"]+"\n"
    if len(p.encode())<12000 and 1<len(a.encode())<5000:ex.append({"root":root,"prompt":p,"answer":a})
   nxt=path+[r]
   for c in reversed(kids.get(node,[])):stack.append((c,nxt))
 rng=random.Random(seed); roots=list({e["root"] for e in ex});rng.shuffle(roots);n=len(roots)
 tr=set(roots[:int(.9*n)]);va=set(roots[int(.9*n):int(.95*n)]);te=set(roots[int(.95*n):])
 split=lambda s:[e for e in ex if e["root"] in s]
 return split(tr),split(va),split(te),{"dataset":"OpenAssistant/oasst1","license":"CC BY 4.0","messages":len(rows),"roots":n,"examples":len(ex),"train":sum(e["root"] in tr for e in ex),"validation":sum(e["root"] in va for e in ex),"test":sum(e["root"] in te for e in ex),"split_unit":"conversation root"}
def pack(e,ctx):
 full=[BOS]+list(e["prompt"].encode())+list(e["answer"].encode())+[EOS];off=max(0,len(full)-ctx);full=full[-ctx:]
 x=torch.tensor(full[:-1]);y=torch.tensor(full[1:]); mask=max(0,1+len(e["prompt"].encode())-off-1);y[:mask]=-100
 return x,y
class Block(nn.Module):
 def __init__(self,d,h):
  super().__init__();self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d);self.a=nn.MultiheadAttention(d,h,batch_first=True);self.f=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
 def forward(self,x):
  t=x.size(1);m=torch.triu(torch.ones(t,t,device=x.device,dtype=torch.bool),1);z=self.n1(x);a,_=self.a(z,z,z,attn_mask=m,need_weights=False);x=x+a;return x+self.f(self.n2(x))
class Rook(nn.Module):
 def __init__(self,d=384,l=8,h=6,c=512):
  super().__init__();self.c=c;self.t=nn.Embedding(V,d);self.p=nn.Embedding(c,d);self.b=nn.ModuleList([Block(d,h) for _ in range(l)]);self.n=nn.LayerNorm(d);self.o=nn.Linear(d,V,bias=False);self.o.weight=self.t.weight
 def forward(self,x):
  x=x[:,-self.c:];z=self.t(x)+self.p(torch.arange(x.size(1),device=x.device))[None]
  for b in self.b:z=b(z)
  return self.o(self.n(z))
def ev(m,ex,dev,c,limit=500):
 m.eval();s=n=0
 with torch.no_grad():
  for e in ex[:limit]:
   x,y=pack(e,c);x=x[None].to(dev);y=y[None].to(dev);z=m(x);s+=F.cross_entropy(z.flatten(0,1),y.flatten(),ignore_index=-100,reduction="sum").item();n+=y.ne(-100).sum().item()
 return s/max(1,n)
def main():
 p=argparse.ArgumentParser();p.add_argument("--steps",type=int,default=5000);p.add_argument("--d",type=int,default=384);p.add_argument("--layers",type=int,default=8);p.add_argument("--heads",type=int,default=6);p.add_argument("--ctx",type=int,default=512);p.add_argument("--batch",type=int,default=8);p.add_argument("--smoke",action="store_true");a=p.parse_args()
 random.seed(1701);torch.manual_seed(1701);dev="cuda" if torch.cuda.is_available() else "cpu";torch.set_num_threads(8)
 tr,va,te,meta=data(1701);out=Path("artifacts/rook-next");out.mkdir(parents=True,exist_ok=True);(out/"dataset_manifest.json").write_text(json.dumps(meta,indent=2))
 if a.smoke:a.d,a.layers,a.heads,a.ctx,a.batch=192,4,6,128,2
 m=Rook(a.d,a.layers,a.heads,a.ctx).to(dev);params=sum(x.numel() for x in m.parameters());print(json.dumps({"device":dev,"parameters":params,"data":meta}),flush=True)
 assert (params>=10_000_000 or a.smoke) and len(tr)>1000 and len(va)>20 and len(te)>20
 opt=torch.optim.AdamW(m.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1);best=1e9;hist=[];start=time.time()
 if a.smoke:a.steps=2
 for step in range(1,a.steps+1):
  m.train();xs=[];ys=[]
  for _ in range(a.batch):
   x,y=pack(random.choice(tr),a.ctx);xs.append(F.pad(x,(a.ctx-len(x),0),value=PAD));ys.append(F.pad(y,(a.ctx-len(y),0),value=-100))
  x=torch.stack(xs).to(dev);y=torch.stack(ys).to(dev);opt.zero_grad(set_to_none=True)
  with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=dev=="cuda" and torch.cuda.is_bf16_supported()):z=m(x);loss=F.cross_entropy(z.flatten(0,1),y.flatten(),ignore_index=-100)
  if not torch.isfinite(loss):raise RuntimeError("non-finite training loss")
  loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
  if step==1 or step%100==0 or step==a.steps:
   vl=ev(m,va,dev,a.ctx);r={"step":step,"train_loss":loss.item(),"val_loss":vl,"seconds":round(time.time()-start,1)};hist.append(r);print(json.dumps(r),flush=True)
   if vl<best:best=vl;torch.save({"model":m.state_dict(),"config":{"d":a.d,"layers":a.layers,"heads":a.heads,"ctx":a.ctx,"vocab":V},"step":step,"val_loss":vl},out/"best.pt")
   torch.save({"model":m.state_dict(),"optimizer":opt.state_dict(),"step":step,"history":hist},out/"resume.pt")
 ck=torch.load(out/"best.pt",map_location=dev,weights_only=False);m.load_state_dict(ck["model"]);report={"architecture":"random-init byte Transformer","parameters":params,"pretrained_weights":False,"external_inference_api":False,"data":meta,"best_step":ck["step"],"best_validation_loss":ck["val_loss"],"heldout_test_loss":ev(m,te,dev,a.ctx),"history":hist,"elapsed_seconds":round(time.time()-start,1),"device":dev}
 (out/"report.json").write_text(json.dumps(report,indent=2));print("FINAL_REPORT "+json.dumps(report),flush=True)
if __name__=="__main__":main()
