import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
#rcParams['figure.figsize']=100,5
#plt.rcParams['agg.path.chunksize'] = 10000
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# 하루당 총 방문수를 시계열로 예측

# 평가 지표를 모아놓은 함수
def regression_results(y_true, y_pred):
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred)
    mse=metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))


def get_users_data():
    return pd.read_csv('df_users.csv')

def get_user_visit_data():
    return pd.read_csv('df_user_visits.csv')



if __name__ == '__main__':
    def main():
        group_data = get_users_data()
        visit_data = get_user_visit_data()

        merged_df = pd.merge(group_data, visit_data, on ='user_uuid')

        merged_df = merged_df[['date_joined', 'visits']]

        merged_df.dropna()

        merged_df['date_joined'] = pd.to_datetime(merged_df.date_joined, format="%Y/%m/%d")

        # 히스토그램으로 각 월당 데이터가 얼마나 많은지 확인

        counts, bins = np.histogram(pd.DatetimeIndex(merged_df['date_joined']).month)
        plt.hist(bins[:-1], bins, weights=counts)
        plt.show()


        # 2월달 이전의 데이터가 많지 않고 또한 가입일자와 총 방문 수의 라인 그래프를 그려본 결과 2월 이전의 그래프가 패턴이 없어
        # 모델 퍼포먼스를 올리기 위해 2월 이전의 데이터를 빼기로 함
        df_age_negative = merged_df[merged_df['date_joined'] < datetime.strptime('2030/02/15', "%Y/%m/%d") ]
        merged_df = merged_df.drop(df_age_negative.index, axis=0)


        merged_df = merged_df.groupby(merged_df['date_joined']).sum()

        merged_df['visits'] =  np.log(1+merged_df['visits'])


        # 없는 날짜들이 있으므로 전날의 평균 방문수로 forward filling

        daily_visit1 = merged_df.asfreq('D',method = 'ffill')


        # 시계열 데이터를 담을 데이터 프래임 설정
        daily_visit2 = daily_visit1[['visits']]
        # 하루 전의 방문수를 담을 열
        daily_visit2.loc[:, 'Yesterday'] = daily_visit2.loc[:, 'visits'].shift()
        # 하루 전과 이틀 전의 방문수 차이를 담을 열
        daily_visit2.loc[:, 'Yesterday_Diff'] = daily_visit2.loc[:, 'Yesterday'].diff()

        daily_visit2 = daily_visit2.dropna()

        X_train = daily_visit2[:datetime.strptime('2030/04/10', "%Y/%m/%d")].drop(['visits'], axis=1)
        y_train = daily_visit2.loc[:datetime.strptime('2030/04/10', "%Y/%m/%d"), 'visits']
        X_test = daily_visit2[datetime.strptime('2030/04/11', "%Y/%m/%d"):].drop(['visits'], axis=1)
        y_test = daily_visit2.loc[datetime.strptime('2030/04/11', "%Y/%m/%d"):, 'visits']


        models = []
        models.append(('LR', LinearRegression()))
        models.append(('NN', MLPRegressor(solver='sgd',max_iter=2000)))
        models.append(('KNN', KNeighborsRegressor()))
        models.append(('RF', RandomForestRegressor(n_estimators=20)))
        models.append(('SVR', SVR(gamma='auto')))


        # 크로스 발리데이션으로 각각의 모델을 비교
        results = []
        names = []
        print("#####################Cross validation comparison between models############################")
        for name, model in models:
            # 시계열 크로스 발리데이션
            tscv = TimeSeriesSplit(n_splits=4)

            cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
            results.append(cv_results)
            names.append(name)
            print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

        # 모델들을 박스 플롯으로 비교
        plt.boxplot(results, labels=names)
        plt.title('Algorithm Comparison')
        plt.show()


        # 크로스 발리데이션으로 선정한 가장 좋은 모델로 핏
        reg = LinearRegression().fit(X_train, y_train)

        # 테스트 데이터로 예측하고 평가 지표 계산
        print("###############Liner regression prediction score###################")
        y_true = y_test.values
        y_pred = reg.predict(X_test)
        regression_results(y_true, y_pred)





if __name__ == '__main__':
    main()