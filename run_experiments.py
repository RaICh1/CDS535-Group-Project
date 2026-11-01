"""Reproduce the CDS535 comparison. Run from any working directory."""
from pathlib import Path
import json, time, hashlib, platform, sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, confusion_matrix, roc_curve,
    silhouette_score, adjusted_rand_score)
from sklearn.base import clone
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
for d in ["results", "figures"]:
    (ROOT/d).mkdir(exist_ok=True)
SEED = 535
MUSH = ["class","cap-shape","cap-surface","cap-color","bruises","odor",
    "gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape",
    "stalk-root","stalk-surface-above-ring","stalk-surface-below-ring",
    "stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color",
    "ring-number","ring-type","spore-print-color","population","habitat"]
MODELS = {
    "Decision tree": DecisionTreeClassifier(min_samples_leaf=2, random_state=SEED),
    "Random forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=1,
                                            random_state=SEED, n_jobs=1),
    "Logistic regression": LogisticRegression(C=1, max_iter=3000, solver="lbfgs"),
    "RBF SVM": SVC(C=1, gamma="scale"),
}

def prep(name, variant="base"):
    if name == "Mushroom":
        imp = SimpleImputer(strategy="constant", fill_value="unknown") if variant == "unknown" else SimpleImputer(strategy="most_frequent")
        return Pipeline([("impute",imp),("encode",OneHotEncoder(handle_unknown="ignore",sparse_output=False)),
                         ("constant",VarianceThreshold())])
    return Pipeline([("impute",SimpleImputer(strategy="median")),("constant",VarianceThreshold()),
                     ("scale",StandardScaler() if variant != "unscaled" else "passthrough")])

def scores(y, pred, score):
    return dict(accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),
                recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),
                auc=roc_auc_score(y,score),kappa=cohen_kappa_score(y,pred))

def getscore(model, x):
    return model.decision_function(x) if hasattr(model,"decision_function") else model.predict_proba(x)[:,1]

def load():
    m = pd.read_csv(ROOT/"data/mushroom/agaricus-lepiota.data",header=None,names=MUSH,na_values="?")
    i = pd.read_csv(ROOT/"data/ionosphere/ionosphere.data",header=None,
                    names=[f"x{j:02d}" for j in range(1,35)]+["class"])
    # Exact duplicate records are removed before any random split. Raw files remain unchanged.
    i = i.drop_duplicates().reset_index(drop=True)
    return {"Mushroom":(m.drop(columns="class"),(m["class"]=="p").astype(int)),
            "Ionosphere":(i.drop(columns="class"),(i["class"]=="b").astype(int))}

def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "savefig.dpi":180})
    data=load(); rows=[]; cvrows=[]; ablations=[]; clusters=[]; metadata={}
    fig, axs=plt.subplots(1,2,figsize=(9,3))
    for ax,(name,(x,y)) in zip(axs,data.items()):
        counts=y.value_counts().sort_index()
        ax.bar(["negative","positive"],counts,color=["#1c728e","#dc8541"])
        for j,v in enumerate(counts): ax.text(j,v+max(counts)*.02,str(v),ha="center")
        ax.set_title(name); ax.set_ylim(0,max(counts)*1.17); ax.set_ylabel("Records")
    fig.tight_layout();fig.savefig(ROOT/"figures/01_class_balance.png");plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.3))
    m,my=data["Mushroom"];i,iy=data["Ionosphere"]
    pd.crosstab(m["odor"],my).plot.bar(stacked=True,ax=axs[0],color=["#1c728e","#dc8541"])
    axs[0].set_title("Mushroom: odor by class");axs[0].set_ylabel("Records")
    axs[0].legend(["edible","poisonous"],fontsize=8)
    for label in [0,1]: axs[1].hist(i.loc[iy==label,"x03"],bins=15,alpha=.6,label=["good","bad"][label])
    axs[1].set_title("Ionosphere: x03");axs[1].set_xlabel("Signal feature value")
    axs[1].legend(); fig.tight_layout();fig.savefig(ROOT/"figures/02_distributions.png");plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    im=ax.imshow(i.drop(columns="x02").corr(),cmap="RdBu_r",vmin=-1,vmax=1)
    ax.set_title("Ionosphere Pearson correlation (constant x02 excluded)")
    ticks=list(range(0,33,4)); ax.set_xticks(ticks,i.drop(columns="x02").columns[ticks],rotation=45)
    ax.set_yticks(ticks,i.drop(columns="x02").columns[ticks]);fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(ROOT/"figures/03_correlations.png");plt.close(fig)
    for name,(x,y) in data.items():
        print("DATASET",name,flush=True)
        clean=x.copy();clean["target"]=y;clean.to_csv(ROOT/f"data/{name.lower()}.csv",index=False)
        train,test=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=SEED)
        xd,yd=x.iloc[train],y.iloc[train];xt,yt=x.iloc[test],y.iloc[test]
        splits=list(StratifiedKFold(10,shuffle=True,random_state=SEED).split(xd,yd))
        foldid=np.full(len(y),-1)
        for fold,(_,val) in enumerate(splits,1):foldid[train[val]]=fold
        pd.DataFrame({"row":np.arange(len(y)),"development_fold":foldid,"target":y}).to_csv(ROOT/f"results/{name}_splits.csv",index=False)
        metadata[name]={"raw_n":8124 if name=="Mushroom" else 351,"n":len(y),"features":x.shape[1],"class_counts":y.value_counts().to_dict(),
                        "missing_cells":int(x.isna().sum().sum()),"missing_rows":int(x.isna().any(axis=1).sum()),
                        "constant_columns":list(x.columns[x.nunique(dropna=False)==1]),
                        "duplicate_predictors":int(x.duplicated().sum()),"development_n":len(train),"test_n":len(test)}
        pd.DataFrame({"attribute":x.columns,"type":[str(t) for t in x.dtypes],"missing":x.isna().sum().values,
                      "unique_observed":x.nunique().values}).to_csv(ROOT/f"data/{name}_schema.csv",index=False)
        fig,ax=plt.subplots(figsize=(6.5,4))
        fitted={}
        for modelname,est in MODELS.items():
            start=time.perf_counter()
            for fold,(tr,va) in enumerate(splits,1):
                pipe=Pipeline([("prep",prep(name)),("model",clone(est))])
                pipe.fit(xd.iloc[tr],yd.iloc[tr]);pr=pipe.predict(xd.iloc[va]);sc=getscore(pipe,xd.iloc[va])
                cvrows.append(dict(dataset=name,model=modelname,fold=fold,**scores(yd.iloc[va],pr,sc)))
            elapsed=time.perf_counter()-start
            pipe=Pipeline([("prep",prep(name)),("model",clone(est))]).fit(xd,yd)
            pr=pipe.predict(xt);sc=getscore(pipe,xt); met=scores(yt,pr,sc)
            subset=pd.DataFrame([r for r in cvrows if r["dataset"]==name and r["model"]==modelname])
            row=dict(dataset=name,model=modelname,cv_seconds=elapsed,**{f"test_{k}":v for k,v in met.items()})
            for k in met:row[f"cv_{k}"]=subset[k].mean();row[f"sd_{k}"]=subset[k].std()
            rows.append(row);fitted[modelname]=pipe
            pd.DataFrame({"row":test,"actual":yt.to_numpy(),"predicted":pr,"score":sc}).to_csv(ROOT/f"results/{name}_{modelname.replace(' ','_')}_test_predictions.csv",index=False)
            np.savetxt(ROOT/f"results/{name}_{modelname.replace(' ','_')}_confusion.csv",confusion_matrix(yt,pr,labels=[0,1]),fmt="%d",delimiter=",")
            fpr,tpr,_=roc_curve(yt,sc);ax.plot(fpr,tpr,label=f"{modelname}: {met['auc']:.3f}")
            print(modelname, "CV F1",round(row["cv_f1"],4),"test",round(met["f1"],4),flush=True)
        ax.plot([0,1],[0,1],"--",color="gray");ax.set(xlabel="False positive rate",ylabel="True positive rate",title=f"{name}: untouched 20% test ROC")
        ax.legend(fontsize=8,loc="lower right");fig.tight_layout();fig.savefig(ROOT/f"figures/04_ROC_{name}.png");plt.close(fig)
        variant="unknown" if name=="Mushroom" else "unscaled"
        for modelname in ["Logistic regression","RBF SVM"]:
            for fold,(tr,va) in enumerate(splits,1):
                pipe=Pipeline([("prep",prep(name,variant)),("model",clone(MODELS[modelname]))]).fit(xd.iloc[tr],yd.iloc[tr])
                ablations.append(dict(dataset=name,model=modelname,variant=variant,fold=fold,
                    **scores(yd.iloc[va],pipe.predict(xd.iloc[va]),getscore(pipe,xd.iloc[va]))))
        # Unsupervised exploration uses development predictors only. Labels are used only after fitting.
        z=prep(name).fit_transform(xd)
        for k in range(2,7):
            km=KMeans(n_clusters=k,n_init=20,random_state=SEED).fit(z)
            sil=silhouette_score(z,km.labels_,sample_size=min(1500,len(z)),random_state=SEED)
            ct=pd.crosstab(km.labels_,yd.to_numpy())
            clusters.append(dict(dataset=name,k=k,inertia=km.inertia_,silhouette=sil,
                purity=ct.max(axis=1).sum()/len(yd),ari=adjusted_rand_score(yd,km.labels_)))
            ct.to_csv(ROOT/f"results/{name}_clusters_k{k}.csv")
    pd.DataFrame(rows).to_csv(ROOT/"results/model_comparison.csv",index=False)
    pd.DataFrame(cvrows).to_csv(ROOT/"results/fold_metrics.csv",index=False)
    pd.DataFrame(ablations).to_csv(ROOT/"results/ablation_folds.csv",index=False)
    pd.DataFrame(clusters).to_csv(ROOT/"results/clustering.csv",index=False)
    (ROOT/"results/metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf8")
    versions={"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,
              "scikit-learn":sklearn.__version__,"matplotlib":matplotlib.__version__,"seed":SEED}
    (ROOT/"results/environment.json").write_text(json.dumps(versions,indent=2),encoding="utf8")
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"data").rglob("*") if p.is_file()}
    (ROOT/"results/data_sha256.json").write_text(json.dumps(hashes,indent=2),encoding="utf8")
    c=pd.DataFrame(clusters);fig,axs=plt.subplots(1,2,figsize=(9,3.2))
    for name in data:
        sub=c[c.dataset==name]
        axs[0].plot(sub.k,sub.inertia/sub.inertia.iloc[0],marker="o",label=name)
        axs[1].plot(sub.k,sub.silhouette,marker="o",label=name)
    axs[0].set(xlabel="k",ylabel="SSE / SSE at k=2",title="Relative within-cluster dispersion")
    axs[1].set(xlabel="k",ylabel="Silhouette",title="Separation in transformed feature space")
    for ax in axs:ax.legend()
    fig.tight_layout();fig.savefig(ROOT/"figures/05_clustering.png");plt.close(fig)
    print("COMPLETE",flush=True)

if __name__=="__main__":
    with threadpool_limits(limits=1): main()
