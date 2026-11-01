# Actual Weka experiments

Weka 3.8.7 runs J48, RandomForest, Logistic and SMO on UCI Mushroom and Ionosphere. The ARFF development/test records match the Python splits. Weka creates its own CV folds; its pooled CV metrics are reported separately from Python fold means. See results/weka_comparison.csv for 16 evaluations and configuration.json for settings.

Set WEKA_HOME to your Weka installation directory, then run `python run_weka.py` to reproduce the experiments. Requires Python 3 and Java. On Windows, `Open_Weka_Demo.cmd` opens the real Weka Explorer with Ionosphere development data, FilteredClassifier/J48, 10 folds and seed 535. Click Start to execute. The included helper class targets Java 17 or later. Weka itself is not redistributed here.

Learned filters are fitted inside each CV fold. Positive labels are poisonous (Mushroom) and bad (Ionosphere). The 20% test partition remains separate. Saved models are fitted on the development partition.
