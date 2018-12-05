import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import models, layers, optimizers, regularizers
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold, learning_curve, GridSearchCV, cross_val_score
import os
# np.set_printoptions(precision=4, threshold=20)
# pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', None)

class Model():
    dataset = None
    train_features = None
    train_labels = None
    test_features = None

    def loadData(self,labelName):
        ds = pd.read_csv('data/train.csv', header = 0)
        self.dataset = ds.sample(frac=1)
        self.test_features = pd.read_csv('data/test.csv', header = 0)

        self.train_labels = self.dataset[labelName].to_frame()
        colLen = len(self.dataset.columns)
        self.train_features = self.dataset.iloc[:,:(colLen-1)]

    def anaData(self):
        df = self.train_features
        # print(self.train_features.isnull().sum())
        # print(self.test_features.isnull().sum())

        # print(self.train_features.describe())
        print(df.head(10))

        print(self.train_features[['n_features','n_informative','n_classes','n_clusters_per_class']].head(20))


    def preproData(self):
        combine = {'train': self.train_features, 'test': self.test_features}
        self.train_features.to_csv(r'feature1.csv', index=False)
        for key in combine.keys():
            combine[key].loc[combine[key]['penalty'] == 'none', 'l1_ratio'] = 0

            dum = pd.get_dummies(combine[key]['penalty'], prefix='pena')
            combine[key] = combine[key].drop('penalty', axis=1)
            combine[key] = pd.concat([combine[key], dum], axis=1)

            combine[key]['pena_l1'] = combine[key].apply(lambda x: x['l1_ratio'] if x['pena_elasticnet']==1 else x['pena_l1'], axis=1)
            combine[key]['pena_l2'] = combine[key].apply(lambda x: 1-x['pena_l1'], axis=1)

            combine[key]['n_jobs'] = combine[key]['n_jobs'].apply(lambda v: 16 if v == -1 else v)
            # combine[key]['n_cluster'] = combine[key]['n_classes']*combine[key]['n_clusters_per_class']
            # combine[key]['n_num'] = combine[key]['n_samples']*combine[key]['n_features']

            # Delete attributes
            drop_attr = ['id','pena_elasticnet','l1_ratio', 'random_state']
            combine[key] = combine[key].drop(drop_attr, axis=1)

            # Data normalization(有三种：零均值规格化，最大最小规格化-此处用的这种，基数变换规格化)
            for c in combine[key].columns:
                mi = np.min(combine[key][c])
                ma = np.max(combine[key][c])
                ratio = (1-0)/(ma-mi)
                combine[key][c] = combine[key][c].apply(lambda x: (x-mi)*ratio)

        self.train_features = combine['train']
        self.test_features = combine['test']

        # print(self.train_features.head())
        # print(self.train_features.info())
        # self.train_features.to_csv(r'feature2.csv', index=False)

    def learning_plot(self,alg):
        x_train = self.train_features.values
        y_train = self.train_labels.values.ravel()
        # train_sizelist = [50, 100, 150, 200, 250, 300]
        train_sizelist = [320]
        train_sizes, train_scores, val_scores = learning_curve(estimator=alg,
                                                               X=x_train,
                                                               y=y_train,
                                                               train_sizes=train_sizelist,
                                                               cv=5,
                                                               shuffle=True,
                                                               scoring='neg_mean_squared_error')

        train_scores_mean = -train_scores.mean(axis=1)
        val_scores_mean = -val_scores.mean(axis=1)

        print('Training scores:\n', pd.Series(train_scores_mean, index=train_sizes))
        # print('\n', '-' * 70)  # separator to make the output easy to read
        print('\nValidation scores:\n\n', pd.Series(val_scores_mean, index=train_sizes))
        # print('\n', '-' * 70)  # separator to make the output easy to read
        print('\nsub:\n', pd.Series(val_scores_mean-train_scores_mean, index=train_sizes))


        plt.style.use('seaborn')
        plt.plot(train_sizes, train_scores_mean, label='Training error')
        plt.plot(train_sizes, val_scores_mean, label='Validation error')
        plt.ylabel('MSE', fontsize=14)
        plt.xlabel('Training set size', fontsize=14)
        plt.title('Learning curves', fontsize=18, y=1.03)
        plt.legend()
        plt.ylim(0, 40)
        plt.show()

    def createModel(self,optimizer='Adamax',reg_w = 0.001,lr=0.005, momentum=0.9):
        colNum = self.train_features.shape[1]
        model = models.Sequential()
        model.add(layers.Dense(1000, activation='relu',input_shape=(colNum,)))
        model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(reg_w)))
        model.add(layers.Dense(32, activation='relu',kernel_regularizer=regularizers.l2(reg_w)))
        # model.add(layers.Dense(32, activation='relu',kernel_regularizer=regularizers.l2(reg_w)))
        # model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(16, activation='relu',kernel_regularizer=regularizers.l2(reg_w)))

        # model.add(layers.Dense(16, activation='relu'))
        # model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(8, activation='relu',kernel_regularizer=regularizers.l2(reg_w)))

        # model.add(layers.Dense(4, activation='relu',kernel_regularizer=regularizers.l2(reg_w)))
        # model.add(layers.Dense(2, activation='relu',kernel_regularizer=regularizers.l2(reg_w)))
        model.add(layers.Dense(1))

        # sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=momentum, nesterov=True)
        # model.compile(optimizer=sgd, loss='mse', metrics=['mae'])  # Adamax
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])  # Adamax
        return model

    def trainModel(self,optimazer,batchsize,epoch,reg_w):
        X = self.train_features.values
        Y = self.train_labels.values.ravel()

        weight_path = "{}_weights.best.hdf5".format('5001_kaggle_model')

        checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min', save_weights_only=True)

        # reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10,
        #                                    verbose=0, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
        #

        early = EarlyStopping(monitor="val_loss",
                              mode="min",
                              patience=10)

        # callbacks_list = [checkpoint, early, reduceLROnPlat]
        callbacks_list = [checkpoint]

        model = self.createModel(optimazer,reg_w)
        # if os.path.exists('5001_kaggle_model_weights.best.hdf5'):
        #     model.load_weights(weight_path)

        score = model.fit(X,Y,
                  validation_split=0.1,
                  shuffle=True,
                  batch_size=batchsize,
                  epochs=epoch,
                  callbacks=callbacks_list
        )

        print(score.history)
        print(score.history.keys())

        # summarize history for loss
        plt.plot(score.history['loss'])
        plt.plot(score.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        return model

    def valModel(self,batchsize,epoch,reg_w,lr, momentum):
        X = self.train_features.values
        Y = self.train_labels.values.ravel()
        weight_path = "{}_weights.best.hdf5".format('5001_kaggle_model')
        checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=0,
                                     save_best_only=True, mode='min', save_weights_only=True)

        reduceLROnPlat = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10,
                                           verbose=0, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
        early = EarlyStopping(monitor="loss",
                              mode="min",
                              patience=10)

        callbacks_list = [checkpoint, early, reduceLROnPlat]


        trains = []
        tests = []
        kf = KFold(n_splits=5, shuffle=True, random_state=10)
        for trainIndes, testIndes in kf.split(X,Y):
            model = self.createModel(reg_w,lr,momentum)
            model.fit(X[trainIndes], Y[trainIndes], batch_size=batchsize, epochs=epoch, callbacks=callbacks_list)

            scores1 = model.evaluate(X[trainIndes], Y[trainIndes],verbose=0)
            scores2 = model.evaluate(X[testIndes], Y[testIndes],verbose=0)
            trains.append(scores1[0])
            tests.append(scores2[0])

        print('trainScore: ', trains)
        print('testScore: ', tests)
        print('mean-trainScore: ', np.mean(trains))
        print('mean-testScore: ', np.mean(tests))


    def model_test(self):
        X = self.train_features.values
        Y = self.train_labels.values.ravel()

        model = KerasRegressor(build_fn=self.createModel,batch_size=32, epochs=600, verbose=0)
        # model = KerasRegressor(build_fn=self.createModel, verbose=0)
        param_grid = dict(reg_w=[0.0004,0.0005,0.0006])
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=2, scoring='neg_mean_squared_error', cv=5)
        grid_result = grid.fit(X, Y)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        print(grid_result.grid_scores_)

        # score = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=5)
        # print(score)
        # print(np.mean(score))

        # self.learning_plot(model)

    def conTrain(self,model,epoch,batchsize):

        l = len(m.train_features)
        x = m.train_features[int(0.9 * l):l]
        y = m.train_labels[int(0.9 * l):l]

        model.fit(x,y,epochs=epoch,batch_size=batchsize)
        return model

    def model_predict(self,model):
        test = self.test_features.values

        data = model.predict(test)
        l = len(data)
        ind = range(l)
        data.resize(1, l)
        result = pd.DataFrame({'id': ind, 'time': data[0]})
        result.to_csv(r'result.csv', float_format='%.16f', index=False)


m = Model()
print('---------------LOAD DATA---------------')
m.loadData('time')
# m.anaData()
print('---------------PREPROCESSING DATA---------------')
m.preproData()
print('---------------TRAINING---------------')
# m.trainModel(128,200,0)
# m.valModel()
# m.lalala()
# m.model_predict()
# m.model_test()
# m.learning_plot()



