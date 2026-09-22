"""Separate V4 reproduction from V5's bias-excluded output definition."""
import torch
import pandas as pd
from transformers import AutoModelForCausalLM,AutoTokenizer
import v5_experiment as v5
import v4_experiment as v4
rows=[]
for mid in ['Qwen/Qwen2.5-0.5B','facebook/opt-125m','EleutherAI/pythia-70m']:
    tok=AutoTokenizer.from_pretrained(mid)
    model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32,attn_implementation='eager').cuda().eval()
    for name,w,xs,bias in v5.capture(model,tok,mid):
        v4.PREP_CACHE.clear()
        with torch.no_grad():
            row,h,_=v4.run_config(w,xs[0],xs[1],bias,name,v4.saliency_scores(w,xs[0]),'activation_rms',.05,8,'none',32,1.5,'contiguous','fp8',4,'mse')
            ea,eb,ec=v5.errors(w,h,xs)
        rows.append(dict(layer=name,bpw=row['bpw'],bias_included_b=row['test_output_error'],bias_excluded_b=eb))
    del model; torch.cuda.empty_cache()
out=pd.DataFrame(rows); out.to_csv(v5.OUT/'csv/baseline_audit.csv',index=False)
print(out[['bpw','bias_included_b','bias_excluded_b']].mean().to_string())
