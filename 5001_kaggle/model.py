import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, learning_curve


class Model():
    dataset = None
    train_features = None
    train_labels = None
    test_features = None

    def loadData(self,labelName):
        self.dataset = pd.read_csv('data/train.csv', header = 0)
        self.test_features = pd.read_csv('data/test.csv', header = 0)

        self.train_labels = self.dataset[labelName].to_frame()
        colLen = len(self.dataset.columns)
        self.train_features = self.dataset.iloc[:,:(colLen-1)]

    def anaData(self):
        print(self.dataset.info())


    def preproData(self):
        combine = {'train': self.train_features, 'test': self.test_features}
        for key in combine.keys():

            # Encode data
            # penalty_mapping = {
            #     'none': 1,
            #     'l2': 2,
            #     'elasticnet': 3,
            #     'l1': 4,
            # }
            # combine[key]['penalty'] = combine[key]['penalty'].map(penalty_mapping)


            dum = pd.get_dummies(combine[key]['penalty'], prefix='pena')
            combine[key] = combine[key].drop('penalty', axis=1)
            combine[key] = pd.concat([combine[key], dum], axis=1)

            combine[key]['pena_l1'] = combine[key].apply(lambda x: x['l1_ratio'] if x['pena_elasticnet']==1 else x['pena_l1'], axis=1)
            combine[key]['pena_l2'] = combine[key].apply(lambda x: 1-x['pena_l1'], axis=1)

            combine[key]['n_jobs'] = combine[key]['n_jobs'].apply(lambda v: 16 if v == -1 else v)
            combine[key]['n_cluster'] = combine[key]['n_classes']*combine[key]['n_clusters_per_class']
            combine[key]['n_num'] = combine[key]['n_samples']*combine[key]['n_features']

            # Delete attributes
            drop_attr = ['id','pena_elasticnet','l1_ratio']
            combine[key] = combine[key].drop(drop_attr, axis=1)

        self.train_features = combine['train']
        self.test_features = combine['test']

        print(self.train_features.head())
        # print(self.train_features.info())
        self.train_features.to_csv(r'feature.csv', index=False)

    def modelTest(self,alg, X, Y, performCV=True, printFeatureImportance=True, cv_folds=10):
        # Perform cross-validation:
        if performCV:
            cv_score = cross_val_score(alg, X, Y, cv=cv_folds, scoring='mean_squared_error')

        # Print model report:
        print("\nModel Report")
        #print("Accuracy : %.4g" % metrics.accuracy_score(Y.values, dtrain_predictions))
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (
            np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

        # Print Feature Importance:
        if printFeatureImportance:
            feat_imp = pd.Series(alg.feature_importances_).sort_values(ascending=False)
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Feature Importance Score')
            plt.show()

    def model_predict(self,alg):
        X = self.test_features.values
        data = alg.predict(X)
        ind = range(len(data))
        result = pd.DataFrame({'id': ind, 'time': data})
        result.to_csv(r'result.csv', float_format='%.16f', index=False)
        result['time'] = result['time'].apply(lambda t: 0 if t < 0 else t)
        result.to_csv(r'result2.csv', float_format='%.16f', index=False)

    def learning_plot(self,alg):
        x_train = self.train_features.values
        y_train = self.train_labels.values.ravel()
        train_sizelist = [1, 50, 100, 150, 200, 250, 300]
        train_sizes, train_scores, val_scores = learning_curve(estimator=alg,
                                                               X=x_train,
                                                               y=y_train,
                                                               train_sizes=train_sizelist,
                                                               cv=5,
                                                               shuffle=True,
                                                               scoring='neg_mean_squared_error')

        train_scores_mean = -train_scores.mean(axis=1)
        val_scores_mean = -val_scores.mean(axis=1)

        print('Training scores:\n\n', pd.Series(train_scores_mean, index=train_sizes))
        print('\n', '-' * 70)  # separator to make the output easy to read
        print('\nValidation scores:\n\n', pd.Series(val_scores_mean, index=train_sizes))
        print('\n', '-' * 70)  # separator to make the output easy to read
        print('\nsub:\n\n', pd.Series(val_scores_mean-train_scores_mean, index=train_sizes))


        plt.style.use('seaborn')
        plt.plot(train_sizes, train_scores_mean, label='Training error')
        plt.plot(train_sizes, val_scores_mean, label='Validation error')
        plt.ylabel('MSE', fontsize=14)
        plt.xlabel('Training set size', fontsize=14)
        plt.title('Learning curves', fontsize=18, y=1.03)
        plt.legend()
        plt.ylim(0, 40)
        plt.show()

    def trainModel(self):
        x_train = self.train_features.values
        y_train = self.train_labels.values.ravel()

        # param_test1 = {'learning_rate':np.arange(0.05,0.06,0.01)}
        # gsearch1 = GridSearchCV(estimator=gbClf,
        #     # param_grid=param_test1,
        #     scoring='mean_squared_error', cv=5)
        # gsearch1.fit(x_train, y_train)
        # print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)


        gbClf = GradientBoostingRegressor(
            learning_rate=0.01,
            n_estimators=1000,
            max_depth=7,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt', subsample=0.65, random_state=10
        )

        self.learning_plot(gbClf)

        # gbClf.fit(x_train,y_train)
        # score = cross_val_score(gbClf,x_train,y_train,scoring='mean_squared_error',cv=5)
        # print(score)
        # print(np.mean(score))

        # self.modelTest(gbClf,x_train,y_train)
        # self.model_predict(gbClf)


        # gbClf = GradientBoostingRegressor()
        # self.modelTest(gbClf,x_train,y_train)


m = Model()
print('---------------LOAD DATA---------------')
m.loadData('time')
m.anaData()
print('---------------PREPROCESSING DATA---------------')
# m.preproData()
print('---------------TRAINING---------------')
# m.trainModel()
m.learning_plot()