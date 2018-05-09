import pandas as pd
import sqlite3
from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset

# Create Storage
conn = sqlite3.connect('retro.db')
db = conn.cursor()

table = pd.read_sql_query("SELECT * from game_stats", conn)

# Setup Auto_ML
df_train = table.sample(frac=.5)
df_test = table.sample(frac=.5)

rew_descriptions = {
    'acts1': 'output',
    'cur_action': 'categorical',
    'prev_action':'categorical'
}

ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=rew_descriptions)
ml_predictor.train(df_train, model_names=['DeepLearningClassifier'],ml_for_analytics=True)
# ml_predictor.score(df_test, df_test.acts1)
ml_predictor.predict(table[-1:])
ml_predictor.save(file_name='reward.ml',verbose=True)


etable = pd.read_sql_query("SELECT * from sarsa", conn)

df_etrain = etable.sample(frac=.5)
df_etest = etable.sample(frac=.5)

esteem_descriptions = {
    'esteem': 'output',
    'cluster': 'categorical',
}

ml_predictor2 = Predictor(type_of_estimator='regressor', column_descriptions=esteem_descriptions)
ml_predictor2.train(df_etrain,ml_for_analytics=True)
# ml_predictor2.score(df_test, df_test.acts1)
ml_predictor2.predict(etable[-1:])
ml_predictor2.save(file_name='esteem.ml',verbose=True)