

from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import preprocessing
import lightgbm as lgb
import numpy as np
import pandas as pd

data = pd.read_csv('./input/train_sample5k.csv')


print("----------类别特征---------")
categorical = ["user_id", "region", "city", "parent_category_name", "category_name", "user_type",
               "image_top_1", "param_1", "param_2", "param_3"]
print("Encoding :", categorical)
lbl = preprocessing.LabelEncoder()
for col in categorical:
    data[col] = lbl.fit_transform(data[col].astype(str))

cols_to_drop = ["item_id", "user_id", "title", "description","activation_date", "image"]
data = data.drop(cols_to_drop, axis=1)


train_y = data['deal_probability']
train_X = data.drop('deal_probability', axis=1)


NFold = 5

skf = KFold(n_splits=NFold, shuffle=False, random_state=218)

rmse_state = []

for n_round in range(100,1001,100):
    for learning_rate in [0.1, 0.05, 0.01]:
        temp = 0.0
        params = {
                "objective": "regression",
                "boosting_type": "gbdt",
                "learning_rate": learning_rate,
                "num_leaves": int(15),
                "feature_fraction": 0.5,
                "seed": 218,
                "drop_rate": 0.1,
                "max_drop": 50,
                "min_child_samples": 10,
                "min_child_weight": 150,
        }

        kf = skf.split(train_X, train_y)
        for train_index, valid_index in kf:
            X_train_x, X_train_y, valid_x, valid_y = train_X.iloc[train_index], train_y.iloc[train_index], train_X.iloc[valid_index], train_y.iloc[valid_index]

            train_data = lgb.Dataset(X_train_x, X_train_y)
            valid_data = lgb.Dataset(valid_x, valid_y, reference=train_data)

            bst = lgb.train(params, train_data, num_boost_round=n_round, valid_sets=valid_data, verbose_eval=1, early_stopping_rounds=50)

            print("Model Evaluation Stage")
            pred_y = bst.predict(valid_x)

            curr_score = np.sqrt(metrics.mean_squared_error(valid_y, pred_y))
            #print( 'RMSE Score of the single stage:', curr_score,'[learnig_rate]', learning_rate,'[n_round]',n_round)
            temp += curr_score

        print('RMSE score:', temp/NFold, '[learnig_rate]', learning_rate,'[n_round]',n_round)
        rmse_state.append([temp/NFold, learning_rate, n_round])

best = min(rmse_state, key=lambda x: x[0])
print('The best score is', best[0], '[learning_rate]', best[1], '[n_round]', best[2])