
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot


# 두 그룹간의 방문수에 차이가 있는지 통계적으로 검증
# T test 가 필요함
# alpha = 0.05
# null hypothesis: 그룹1과 그룹2 의 population mean 이 동일하다
# alternative hypothesis: 그룹1과 그룹2의 population mean 이 동일하지 않다
# Assumptions:


def get_users_data():
    return pd.read_csv('df_users.csv')

def get_user_visit_data():
    return pd.read_csv('df_user_visits.csv')


if __name__ == '__main__':
    def main():
        group_data = get_users_data()
        visit_data = get_user_visit_data()

        #데이터의 형태를 살피기

        print("Shape of group data:",group_data.shape)
        print("Shape of visit data",visit_data.shape)
        print("Number of unique carrier",group_data['carrier'].nunique())
        print("Number of Nan in visits column: ",visit_data['visits'].isnull().sum())

        # 두 그룹간의 방문수 차이를 비교하기 위해 두 데이터 프래임을 유저 아이디를 기준으로 합침
        merged_df = pd.merge(group_data, visit_data, on ='user_uuid')

        # 한글이 플롯에 인식되지 않아 영어로 변환
        merged_df.loc[merged_df['carrier'] == "초록", 'carrier'] = "green"
        merged_df.loc[merged_df['carrier'] == "노랑", 'carrier'] = "yellow"
        merged_df.loc[merged_df['carrier'] == "보라", 'carrier'] = "purple"

        # 필요한 열만 선택
        group_visit = merged_df[['carrier','group','visits']]

        # 방문 수의 히스토그램을 그려본 결과 heavily right skewed 돼있음
        # 그룹1과 2의 데이터를 분리하여 각각 로그 스케일로 전환하여 skewedness 를 줄임

        # 그룹1의 데이터
        group_1_visits = group_visit[group_visit['group'] == "group_1"].copy()
        print("Size of group_1:",group_1_visits.shape)

        group_1_visits['positive'] = (1 + group_1_visits['visits']) / 2
        group_1_visits['logvisits1'] = np.log(group_1_visits['positive'])

        # 그룹2의 데이터
        group_2_visits = group_visit[group_visit['group'] == "group_2"].copy()
        print("Size of group_2:",group_2_visits.shape)

        group_2_visits['positive'] = (1+group_2_visits['visits'])/2
        group_2_visits['logvisits2']=np.log(group_2_visits['positive'])


        #hist_group_2 = group_2_visits['logvisits2'].plot.hist(bins=12, alpha=0.5)

        #fig1 = hist_group_2.get_figure()

        #fig1.savefig('/Users/yourearl82/PycharmProjects/RainistTest/figure2.pdf')

        # 로그 스케일된 방문수의 히스토그램을 그룹1과 그룹2를 겹쳐 나타냄

        result_visits = pd.concat([group_1_visits['logvisits1'],group_2_visits['logvisits2']],axis=1)


        hist_visits = result_visits.plot.hist(bins = 12, alpha=0.5)
        plt.show()

        # Normality test: null hypothesis: Population is normally distributed.
        # Apply appropriate test.
        print("############### t test for group 1 and group 2######################")
        print("P val for normality test of group_2 : ",stats.normaltest(group_2_visits['logvisits2'].values).pvalue)
        # 로그 스케일로 변환한 후에도 normal distribution이 아님. 하지만 샘플 숫자가 많으므로 central limit theorem에 의해 문제 없음.
        # 두 그룹의 variance가 같은지 테스트
        # Null hypothesis: Two populations have the same variance
        print("Equal variance test for group_1 VS group_2 : ",stats.levene(group_1_visits['logvisits1'],group_2_visits['logvisits2']).pvalue)
        # P value is bigger than 0.05 : fail to reject null (two populations probably have equal variance)
        # However since the sample size of group_1 is approx double of the size of group_2, do Welch's t test
        # Null hypothesis: The mean of the population for group_1 and group_2 are the same
        print("Welch's t test for group_1 VS group_2 :",stats.mstats.ttest_ind(group_1_visits['logvisits1'], group_2_visits['logvisits2'],equal_var=False).pvalue)

        # Not knowing what the true distributions look like for group_1 and group_2, we settle for mannwhitney test

        print("Mann Whitney U test for group_1 VS group_2 :",stats.mannwhitneyu(group_1_visits['logvisits1'], group_2_visits['logvisits2']).pvalue)

        # 각각의 테스트에서 p value가 alpha 보다 크므로 fail to reject null hypotheis.
        # t test결과 두 그룹의 population mean 이 같을 확률이 크다.

        print("######## Compare the difference in the number of visits in group_1 and group_2 in each of G,Y,V carriers")

        # We want to test if the mean of the populations of all possible combination of 2 independent variables
        # each with level 2 and level 3 are the same. So we are considering 6 distributions.

        # To select an appropriate test, we want to check the sample size in each group.

        print("Sample size of Group_1 초록: ",group_visit[(group_visit['group'] == "group_1") & (group_visit['carrier'] == "green")].shape)
        print("Sample size of Group_2 초록: ",group_visit[(group_visit['group'] == "group_2") & (group_visit['carrier'] == "green")].shape)
        print("Sample size of Group_1 노랑: ",group_visit[(group_visit['group'] == "group_1") & (group_visit['carrier'] == "yellow")].shape)
        print("Sample size of Group_2 노랑: ",group_visit[(group_visit['group'] == "group_2") & (group_visit['carrier'] == "yellow")].shape)
        print("Sample size of Group_1 보라: ",group_visit[(group_visit['group'] == "group_1") & (group_visit['carrier'] == "purple")].shape)
        print("Sample size of Group_2 보라: ",group_visit[(group_visit['group'] == "group_2") & (group_visit['carrier'] == "purple")].shape)

        # We want to test if the effect of the group on the mean number of visits is independent from the effect of carrier and vice versa

        sns_plot=sns.boxplot(x="group", y="visits", hue="carrier", data=group_visit, palette="Set3")
        plt.show()

        print("############ Unbalanced two way anova test################ ")
        # Referring to recent paper by Oyvind, two-way anova test. Use type 2 for unbalanced data.
        model = ols('visits ~ C(group) + C(carrier) + C(group):C(carrier)', data=group_visit).fit()
        anova_table = sm.stats.anova_lm(model, typ=3)
        print(anova_table)

        fig = interaction_plot(x=group_visit['carrier'], trace=group_visit['group'], response=group_visit['visits'],
                               colors=['#4c061d', '#d17a22'])

        plt.show()

        # Try balancing sample size for each group by selecting the same number of points for each group.
        # To minimize the waste of data points, select 26060(smallest group size) points from each group.

        balanced = pd.concat([group_visit[(group_visit['group'] == "group_1") & (group_visit['carrier'] == "green")].sample(n=26060, random_state=1),
                              group_visit[(group_visit['group'] == "group_2") & (group_visit['carrier'] == "green")].sample(n=26060, random_state=1),
                              group_visit[(group_visit['group'] == "group_1") & (group_visit['carrier'] == "yellow")].sample(n=26060, random_state=1),
                              group_visit[(group_visit['group'] == "group_2") & (group_visit['carrier'] == "yellow")].sample(n=26060, random_state=1),
                              group_visit[(group_visit['group'] == "group_1") & (group_visit['carrier'] == "purple")].sample(n=26060, random_state=1),
                              group_visit[(group_visit['group'] == "group_2") & (group_visit['carrier'] == "purple")].sample(n=26060, random_state=1)
                              ])
        print("############ Balanced two way anova test################ ")
        # Two way anova for balanced data
        model2 = ols('visits ~ C(group) + C(carrier) + C(group):C(carrier)', data=balanced).fit()
        anova_table2 = sm.stats.anova_lm(model2, typ=2)
        print(anova_table2)

        # since the p value is not significant, we fail to reject the null hypothesis
        # No need to to post hoc pairwise comparison.


if __name__ == '__main__':
    main()