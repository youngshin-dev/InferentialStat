
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot


# statistically test if there is any difference in the number of clicks
# We first consider two tailed two sample t test  with alpha = 0.05 with the following hypothesis
# null hypothesis: population mean in group 1 and grou2 are equal
# alternative hypothesis: population mean in group 1 and group 2 are not the same
# Assumptions: To do t test we need that the two populations are normally distributed and
# they have the equal variance. We will test these assumptions and apply appropriate hypothesis testing.



def get_users_data():
    return pd.read_csv('df_users.csv')

def get_user_visit_data():
    return pd.read_csv('df_user_visits.csv')


if __name__ == '__main__':
    def main():
        group_data = get_users_data()
        visit_data = get_user_visit_data()

        #Explore the shape of the data

        print("Shape of group data:",group_data.shape)
        print("Shape of visit data",visit_data.shape)
        print("Number of unique carrier",group_data['carrier'].nunique())
        print("Number of Nan in visits column: ",visit_data['visits'].isnull().sum())

        # Merge two data frame by user id
        merged_df = pd.merge(group_data, visit_data, on ='user_uuid')

        # 한글이 플롯에 인식되지 않아 영어로 변환
        merged_df.loc[merged_df['carrier'] == "초록", 'carrier'] = "green"
        merged_df.loc[merged_df['carrier'] == "노랑", 'carrier'] = "yellow"
        merged_df.loc[merged_df['carrier'] == "보라", 'carrier'] = "purple"

        # Select only the necessary columns
        group_visit = merged_df[['carrier','group','visits']]

        # Histogram of the number of clicks shows heavily right skewed
        # Separate group1 and group2 data and apply log to reduce skewness

        # Group 1
        group_1_visits = group_visit[group_visit['group'] == "group_1"].copy()
        print("Size of group_1:",group_1_visits.shape)

        group_1_visits['positive'] = (1 + group_1_visits['visits']) / 2
        group_1_visits['logvisits1'] = np.log(group_1_visits['positive'])

        # Group2
        group_2_visits = group_visit[group_visit['group'] == "group_2"].copy()
        print("Size of group_2:",group_2_visits.shape)

        group_2_visits['positive'] = (1+group_2_visits['visits'])/2
        group_2_visits['logvisits2']=np.log(group_2_visits['positive'])


        #hist_group_2 = group_2_visits['logvisits2'].plot.hist(bins=12, alpha=0.5)

        #fig1 = hist_group_2.get_figure()

        # Overlayed histogram of log number of clicks for group 1 and group 2

        result_visits = pd.concat([group_1_visits['logvisits1'],group_2_visits['logvisits2']],axis=1)


        hist_visits = result_visits.plot.hist(bins = 12, alpha=0.5)
        plt.show()

        # Normality test: null hypothesis: Population is normally distributed.
        # Apply appropriate test.
        print("############### t test for group 1 and group 2######################")
        print("P val for normality test of group_2 : ",stats.normaltest(group_2_visits['logvisits2'].values).pvalue)
        # Not looking normal distribution even after transformation.
        # Equal variance test
        # Null hypothesis: Two populations have the same variance
        print("Equal variance test for group_1 VS group_2 : ",stats.levene(group_1_visits['logvisits1'],group_2_visits['logvisits2']).pvalue)
        # P value is bigger than 0.05 : fail to reject null (two populations probably have equal variance)
        # However since the sample size of group_1 is approx double of the size of group_2, do Welch's t test
        # Null hypothesis: The mean of the population for group_1 and group_2 are the same
        print("Welch's t test for group_1 VS group_2 :",stats.mstats.ttest_ind(group_1_visits['logvisits1'], group_2_visits['logvisits2'],equal_var=False).pvalue)

        # Not knowing what the true distributions look like for group_1 and group_2, we settle for mannwhitney test

        print("Mann Whitney U test for group_1 VS group_2 :",stats.mannwhitneyu(group_1_visits['logvisits1'], group_2_visits['logvisits2']).pvalue)

        # In each test, p value greater than alpha:fail to reject null hypotheis.

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