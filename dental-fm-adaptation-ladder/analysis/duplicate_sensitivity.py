# Sensitivity check: recompute ADCD test macro AUROC with crops from duplicated radiograph files removed.
# Usage: python duplicate_sensitivity.py <results_folder>   (writes <results_folder>/dupsens.json)
import numpy as np, json, glob, os, re, sys
from sklearn.metrics import roc_auc_score
R_ = sys.argv[1]
G=np.load(f'{R_}/adcd_groups_v2.npy',allow_pickle=True); E=np.load(f'{R_}/adcd_external_v2.npy')
R=json.load(open(f'{R_}/adcd_records_v2.json'))
def macro(p,y):
    a=[]
    for j in range(3):
        if y[:,j].sum()<5 or (1-y[:,j]).sum()<5: return np.nan
        a.append(roc_auc_score(y[:,j],p[:,j]))
    return np.mean(a)
keep_cache={}
out={}
for f in sorted(glob.glob(f'{R_}/per_tooth/*.npz')):
    n=os.path.basename(f)[:-4]
    if n.endswith('_ext'): continue
    z=np.load(f); g=z['groups']
    key=tuple(g[:20].tolist())+(len(g),)
    if key not in keep_cache:
        tg=set(g.tolist()); idx=[i for i in range(len(G)) if not E[i] and G[i] in tg]
        assert len(idx)==len(g)
        first={}
        for i in idx: first.setdefault(G[i],sorted({R[k]['path'] for k in idx if G[k]==G[i]})[0])
        keep_cache[key]=np.array([R[i]['path']==first[G[i]] for i in idx])
    m=keep_cache[key]
    out[n]=(macro(z['probs'],z['labels']), macro(z['probs'][m],z['labels'][m]), int((~m).sum()))
d=np.array([v[1]-v[0] for v in out.values() if v[0]==v[0] and v[1]==v[1]])
print('runs',len(out),'dropped crops per seed', sorted({v[2] for v in out.values()}))
print('macro change: mean %.4f  max|.| %.4f'%(d.mean(), np.abs(d).max()))
json.dump({k:list(v) for k,v in out.items()}, open(os.path.join(R_,'dupsens.json'),'w'))
