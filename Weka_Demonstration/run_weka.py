"""Rerun the saved Weka experiments. Requires Java and a local Weka installation."""
from pathlib import Path
import os,json,subprocess,re,csv,time
root=Path(__file__).resolve().parent
home=Path(os.environ.get('WEKA_HOME','D:/Weka/Weka-3-8-7'))
java=home/'jre/jre-25.0.2-full/bin/java.exe'
java=str(java) if java.exists() else 'java'
assert (home/'weka.jar').exists(),'Set WEKA_HOME to your Weka installation directory'
jobs=json.loads((root/'results/commands.json').read_text(encoding='utf8'))
rows=[]
for job in jobs:
    command=[str(v).replace('$DEMO_DIR',str(root)).replace('$WEKA_JAR',str(home/'weka.jar')).replace('$JAVA',java) for v in job['command']]
    start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,encoding='utf8',errors='replace')
    ds,model,protocol=job['dataset'],job['model'],job['protocol']
    (root/f'results/{ds}_{model}_{protocol}.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf8')
    assert r.returncode==0 and '=== Confusion Matrix ===' in r.stdout,r.stderr
    marker='=== Stratified cross-validation ===' if protocol=='cv10' else '=== Error on test data ==='
    output=r.stdout.split(marker)[-1]
    acc=re.search(r'Correctly Classified Instances\s+\d+\s+([\d.]+)',output)
    label='poisonous' if ds=='Mushroom' else 'bad'
    metric=next(line.split() for line in output.splitlines() if line.strip().endswith(label) and len(line.split())==9)
    rows.append(dict(dataset=ds,model=model,protocol=protocol,accuracy=float(acc.group(1))/100,precision=metric[2],recall=metric[3],f1=metric[4],auc=metric[6],seconds=round(time.perf_counter()-start,3)))
    print(rows[-1],flush=True)
with (root/'results/weka_comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
