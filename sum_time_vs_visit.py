import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler


def get_user_visit_data():
    return pd.read_csv('df_user_visits.csv')

# 여러 디그리의 폴리노미얼 리그레션을 비교하기 위한 함수
def score_polyfit(n,X_train,y_train,X_valid,y_valid):
    model = make_pipeline(
        PolynomialFeatures(degree=n, include_bias=True),
        LinearRegression(fit_intercept=False)
    )
    model.fit(X_train, y_train)
    print('n=%i: score=%.5g' % (n, model.score(X_valid, y_valid)))



if __name__ == '__main__':
    def main():
        visit_data = get_user_visit_data()

        visit_data = visit_data[visit_data['sum_time_to_interactive'].notna()]
        visit_data = visit_data[visit_data['avg_time_to_interactive'].notna()]

        # 총 로딩 시간과 방문 수 사이에 대략 어떤 관계가 있는지 확인하기 위해 스케터 플롯

        #ax = sns.regplot(x=visit_data['sum_time_to_interactive'], y=visit_data['visits'], marker="+")
        #plt.show()

        # 대략 리니어의 관계가 보임.
        # 데이터가 많이 분산되어있고 skewed 되어있어 preprocessing으로 QuantileTransformer 선택
        # 방문 수 데이터는 많이 skewed 되어 있으므로 로그 스케일로 전환

        quantile_transformer = preprocessing.QuantileTransformer(output_distribution = 'normal', random_state = 0)
        X_train, X_valid, y_train, y_valid = train_test_split(
            quantile_transformer.fit_transform(np.stack([np.log(visit_data['sum_time_to_interactive'].to_numpy())], axis=1)),
            np.log(visit_data['visits'].to_numpy()))
        print(X_train.shape, X_valid.shape)
        print(y_train.shape, y_valid.shape)

        for i in range(1,10):
            score_polyfit(i,X_train,y_train,X_valid,y_valid)



if __name__ == '__main__':
    main()