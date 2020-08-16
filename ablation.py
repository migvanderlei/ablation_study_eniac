import pandas as pd
from json import load
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion
from features import *
import argparse

# args
parser=argparse.ArgumentParser()
parser.add_argument('--task', help='Task 1 ou 2')
parser.add_argument('--ds', help='Dataset [books, restaurants, tweets, apps]')
parser.add_argument('--jobs', help="Numero de jobs executados. Default = 4")
parser.add_argument('--cv', help="Numero de folds. Default = 10")
parser.add_argument('--clf', help="Classifier. 0 = SVM; 1 = GBT. Default = 0")
args=parser.parse_args()

#fazendo parsing dos argumentos
jobs = 1
task = '1'
dataset='books'
classifier=0
cv = 10

if args.jobs is not None:
    jobs = int(args.jobs)
if args.cv is not None:
    cv = int(args.cv)
if args.task is not None:
    task = args.task
if args.ds is not None:
    if args.ds == 'apps' and args.task == '1':
        raise ValueError("Dataset apps não disponível para a tarefa 1")
    else:
        dataset = args.ds
if args.clf is not None:
    classifier = int(args.clf)

#lendo os dados
csv_file_name = './data/task{}/{}.tsv'.format(task, dataset)

print("Task: {}\nDataset: {}\nClassifier: {}\nCross Validation: {} folds\nJobs: {}"
      .format("Subjectivity (1)" if task=="1" else "Polarity (2)", dataset, "Support Vector Machine" if classifier == 0 else "Gradient Bosting Trees", cv, jobs))
print("-- Iniciando --")

data = pd.read_csv(csv_file_name, sep="\t")

corpus = data.sentence
if task == '1':
    targets = data.has_opinion
else:
    targets = data.polarity



def get_classifier(task, classifier, dataset):
    with open("./data/parameters/{}-{}-{}.json".format(task, "svm" if classifier == 0 else "gbt", dataset)) as f:
        parameters = load(f)
    if classifier == 0:
        return SVC(kernel=parameters['kernel'],
                   C=parameters['C'],
                   class_weight= None if parameters['class_weight'] == 'None' else parameters['class_weight'],
                   gamma=parameters['gamma'])
    else:
        return GradientBoostingClassifier(
            learning_rate=parameters['learning_rate'],
            max_depth=parameters['max_depth'],
            min_samples_leaf=parameters['min_samples_leaf'],
            n_estimators=parameters['n_estimators']
        )


features_list = [('qtAdjectives', CountAdjectives()),
                  ('qtComparatives', CountComparatives()),
                  ('qtSuperlatives', CountSuperlatives()),
                  ('qtAdverbs', CountAdverbs()),
                  ('qtNouns', CountNouns()),
                  ('qtAmod', CountAmod())
              ]

def ablate_feature(index):
    if index < 5:
        removed_feature = features_list[index]
        return [feature for feature in features_list if feature != removed_feature]
    else:
        return features_list

def eval(features, corpus, targets, folds, removed_feature):
    if removed_feature == 'words':
        pipeline = Pipeline(steps=[
            ('features', FeatureUnion(
                transformer_list=[
                    ('tree', Pipeline([
                        ('spacy', SpacyTransformer()),
                        ('tree_features', FeatureUnion(
                            transformer_list=features))
                    ]))
                ],
            )),
            ('clf', get_classifier(task, classifier, dataset)),
        ])

    else:
        pipeline = Pipeline(steps=[
            ('features', FeatureUnion(
                transformer_list=[
                    ('qtWords', CountWords()),
                    ('tree', Pipeline([
                      ('spacy', SpacyTransformer()),
                      ('tree_features', FeatureUnion(
                          transformer_list=features))
                  ]))
                ],
            )),
            ('clf', get_classifier(task, classifier, dataset)),
        ])

    scoring = {'p': 'precision',
               'r': 'recall',
               'f1': 'f1',
               'a': 'accuracy'}
    #print("-- Pipeline preparada --")
    scores = cross_validate(pipeline, X=corpus, y=targets, scoring=scoring, cv=folds)
    #print("-- Scores obtidos --")
    return removed_feature+","+\
        "%.3f," % scores['test_a'].mean()+\
        "%.3f," % scores['test_p'].mean()+\
        "%.3f," % scores['test_r'].mean()+\
        "%.3f" % scores['test_f1'].mean()+'\n'

removed_features = ['adj', 'comp', 'super', 'adv', 'noun', 'amod', 'words']
content = ['removed_feature, accuracy, precision, recall, f1\n']
for index, feature in zip(range(7), removed_features):
    print("removendo %s" % feature, index)
    content.append(eval(ablate_feature(index), corpus, targets, cv, feature))

out_file_name = './out/ablation_study-{}-{}-{}.csv'.format(task, dataset, "SVM" if classifier == 0 else "GBT")
with open(out_file_name, 'w+') as f:
    f.writelines(content)
print("-- Arquivo {} gerado --".format(out_file_name))
print("-- Finalizado --")

