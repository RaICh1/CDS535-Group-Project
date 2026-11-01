"""Short presentation demo: rebuild the radar SVM using the saved split."""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from run_experiments import ROOT, load, prep, MODELS, getscore, scores

x,y=load()["Ionosphere"]
split=pd.read_csv(ROOT/"results/Ionosphere_splits.csv")
train=split.loc[split.development_fold!=-1,"row"].to_numpy()
test=split.loc[split.development_fold==-1,"row"].to_numpy()
model=Pipeline([("prep",prep("Ionosphere")),("model",clone(MODELS["RBF SVM"]))])
model.fit(x.iloc[train],y.iloc[train])
pred=model.predict(x.iloc[test])
print("Ionosphere: training =",len(train),"test =",len(test))
print(model)
print("\nPositive label: bad radar return")
for name,value in scores(y.iloc[test],pred,getscore(model,x.iloc[test])).items():
    print(f"{name}: {value:.4f}")
print("\nConfusion matrix: true rows / predicted columns, [good, bad]")
print(confusion_matrix(y.iloc[test],pred))
