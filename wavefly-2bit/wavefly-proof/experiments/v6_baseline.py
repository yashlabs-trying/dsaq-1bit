"""Re-run V5 without overwriting its historical measurements."""
import v5_experiment as v5
v5.OUT=v5.ROOT/'results/v6/baseline_reproduction'
for d in ['csv','json','plots']: (v5.OUT/d).mkdir(parents=True,exist_ok=True)
v5.main()
