import statsmodels.formula.api as smf
import thinkplot
import thinkstats2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelBinarizer

sns.set_theme(style="ticks", color_codes=True)
# read dataset
project = pd.read_csv("BankChurners.csv", delimiter=',')
# rename the column names for simplicity
old_names = project.columns
new_names = ['Clientnum', 'Attrition', 'Age', 'Gender', 'Dependent_count', 'Education', 'Marital_Status', 'Income', 
             'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive', 'Contacts_Count', 
             'Credit_Limit', 'Total_Revolving_Bal','Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio','Naive_Bayes_1','Naive_Bayes_2']
project.rename(columns=dict(zip(old_names, new_names)), inplace=True)
# DATA CLEANING AND VARIABLE SECTION (BULLET 1)

# drop variables
data1 = project.drop(['Naive_Bayes_1', 'Naive_Bayes_2', 'Clientnum'], axis=1)
# Check for duplicate values or null values
data1.duplicated().any()
data1.isnull().any()

# Checking for any multicolinearity
X = data1.copy()
X.drop(columns=['Attrition'], inplace=True)
y = data1['Attrition']
labelBinarizer = LabelBinarizer()
y = labelBinarizer.fit_transform(y)
y = np.reshape(y, -1)
y = pd.Series(y)
analysisData = X.copy()
analysisData['Attrition'] = y
correlation = analysisData.corr()
f, ax = plt.subplots(figsize=(14,12))
plt.title('Correlation of numerical attributes', size=16)
sns.heatmap(correlation)
plt.show()
# Delete some variables with multicolinearity
data2 = data1.drop(['Age', 'Months_on_book', 'Total_Revolving_Bal','Avg_Utilization_Ratio', 'Total_Trans_Amt', 'Total_Trans_Ct'], axis=1)

# plots for demographic variables that are left
# Gender vs Attrition
sns.catplot(y="Attrition", hue = 'Gender', kind="count", palette="pastel", edgecolor=".6", data=data2)
# Dependents vs Attrition
sns.catplot(x="Dependent_count", y="Attrition", hue="Attrition", kind="box", dodge=False, data=data1)
Non_Customer = data2[data2['Attrition'] == "Attrited Customer"] #Not A Customer
Customer = data2[data2['Attrition'] == "Existing Customer"] # Customer
def plot_compare(column, category_name):
    Noncust = len(Non_Customer[column].unique())
    Cust = len(Customer[column].unique())
    
    noncustCounts = Non_Customer[column].value_counts().sort_index()
    custCounts = Customer[column].value_counts().sort_index()
    
    inoncust = np.arange(Noncust)    
    icust = np.arange(Cust)    
    width = 1  
    figs, axs = plt.subplots(1,2, figsize=(10,5))
    axs[1].bar(inoncust, noncustCounts, width, color='b')
    axs[1].set_title('Non_Customer ' + category_name, fontsize=20)
    axs[1].set_xticks(inoncust)
    axs[1].set_xticklabels(noncustCounts.index.tolist(), rotation=45)
    
    axs[0].bar(icust, custCounts, width, color='r')
    axs[0].set_title('Customer ' + category_name, fontsize=20)
    axs[0].set_ylabel('Amount of People')
    axs[0].set_xticks(icust)
    axs[0].set_xticklabels(custCounts.index.tolist(), rotation=45)
    plt.show()     
# Eduction vs Attrition
plot_compare('Education','Education')
# Marital Status vs Attrition
plot_compare('Marital_Status', 'Marital_Status')
# Income vs Attrition
plot_compare('Income','Income')
#Drop demographics
data3 = data2.drop(['Gender', 'Dependent_count', 'Education', 'Marital_Status', 'Income'], axis=1)
# Change categorial to numerical 
ord_enc = OrdinalEncoder()
data3["Attrition1"] = ord_enc.fit_transform(data3[["Attrition"]])
data3[["Attrition", "Attrition1"]].head(10126)

data3["Card_Category1"] = ord_enc.fit_transform(data3[["Card_Category"]])
data3[["Card_Category", "Card_Category1"]].head(10126)

# Drop the category variables that were converted.
data4 = data3.drop(['Attrition',  'Card_Category'], axis=1)

# HISTOGRAMS OF REMAINING VARIABLES (BULLET 3)

# Histograms for variables
data4.hist(column = 'Attrition1')
data4.hist(column = 'Total_Relationship_Count')
data4.hist(column = 'Months_Inactive')
data4.hist(column = 'Contacts_Count')
data4.hist(column = 'Credit_Limit')
data4.hist(column = 'Avg_Open_To_Buy')
data4.hist(column = 'Total_Amt_Chng_Q4_Q1')
data4.hist(column = 'Total_Ct_Chng_Q4_Q1')
data4.hist(column = 'Card_Category1')

# DESCRIPTIVE STATISTICS (BULLET 4)

data4.agg({"Total_Relationship_Count": ["min", "max", "median", "skew"],
            "Months_Inactive": ["min", "max", "median", "mean"],})
data4.agg({"Contacts_Count": ["min", "max", "median", "mean"],
            "Credit_Limit": ["min", "max", "median", "mean"],})
data4.agg({"Avg_Open_To_Buy": ["min", "max", "median", "mean"],
            "Total_Amt_Chng_Q4_Q1": ["min", "max", "median", "mean"],})
data4.agg({"Total_Ct_Chng_Q4_Q1": ["min", "max", "median", "mean"],
            "Card_Category1": ["min", "max", "median", "mean"],
            "Attrition1": ["min", "max", "median", "mean"],})
        
# PMF (BULLET 5)
less_5000 = data2[data2["Credit_Limit"] < 5000]
more_5000 = data2[data2["Credit_Limit"] > 5000]
thinkplot.PrePlot(2, cols=2)
first_pmf = thinkstats2.Pmf(less_5000.Total_Relationship_Count, label= "< 5000 Credit Limit" )
second_pmf = thinkstats2.Pmf(more_5000.Total_Relationship_Count, label= " > 5000 Credit Limit")
thinkplot.Hist(first_pmf, width=1)
thinkplot.Hist(second_pmf, width=1)
thinkplot.Config(xlabel="Amount of Products by customer", ylabel="Probability")

thinkplot.PrePlot(2)
thinkplot.SubPlot(2)
thinkplot.Pmfs([first_pmf,second_pmf])
thinkplot.Show(xlabel="Amount of Products by customer")

# CDF (BULEET 6)
cdf = thinkstats2.Cdf(data4.Credit_Limit)
thinkplot.Cdf(cdf)

# Analytical distribution (BULLET 7)

mu, var = thinkstats2.TrimmedMeanVar(data4.Credit_Limit, p=0.01)
print('Mean, Var', mu, var)
sigma = np.sqrt(var)
print('Sigma', sigma)
xs, ps = thinkstats2.RenderNormalCdf(mu, sigma, low=0, high=12.5)

thinkplot.Plot(xs, ps, label='model', color='0.6')
cdf = thinkstats2.Cdf(data4.Credit_Limit, label='data')

thinkplot.PrePlot(1)
thinkplot.Cdf(cdf) 
thinkplot.Config(title='Credit Limit',
                 xlabel='Credit Limit (Dollars)',
                 ylabel='CDF')
# ScatterPlots (BULLET 8)
thinkplot.scatter(data1.Total_Revolving_Bal,data1.Avg_Utilization_Ratio )
thinkstats2.Corr(data1.Total_Revolving_Bal, data1.Avg_Utilization_Ratio)
thinkstats2.SpearmanCorr(data1.Total_Revolving_Bal,data1.Avg_Utilization_Ratio)

# Hypothesis Testing (BULLET 9)
Male = data2[data2["Gender"] == "M"]
Female = data2[data2["Gender"] == "F"]
data = Male.Total_Relationship_Count.values, Female.Total_Relationship_Count.values
class DiffMeansPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat

    def MakeModel(self):
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data
ht = DiffMeansPermute(data)
pvalue = ht.PValue()
pvalue
ht.PlotCdf()
thinkplot.Show(xlabel='test statistic', ylabel='CDF')

# Linear regression (BULLET 9)
formula = 'Attrition1 ~ Total_Relationship_Count + Months_Inactive + Contacts_Count + Credit_Limit + Avg_Open_To_Buy + Total_Ct_Chng_Q4_Q1 + Total_Ct_Chng_Q4_Q1 + Card_Category1'
model = smf.ols(formula, data=data4)
results = model.fit() 

inter = results.params['Intercept']
slope = results.params['Total_Relationship_Count']
def SummarizeResults(results):

    for name, param in results.params.items():
        pvalue = results.pvalues[name]
        print('%s   %0.3g   (%.3g)' % (name, param, pvalue))

    try:
        print('R^2 %.4g' % results.rsquared)
        ys = results.model.endog
        print('Std(ys) %.4g' % ys.std())
        print('Std(res) %.4g' % results.resid.std())
    except AttributeError:
        print('R^2 %.4g' % results.prsquared)
SummarizeResults(results)
#  Correlation between variables
thinkstats2.Corr(data4.Attrition1,data4.Total_Relationship_Count)
thinkstats2.Corr(data4.Attrition1,data4.Months_Inactive)
thinkstats2.Corr(data4.Attrition1,data4.Contacts_Count)
thinkstats2.Corr(data4.Attrition1,data4.Credit_Limit)
thinkstats2.Corr(data4.Attrition1,data4.Avg_Open_To_Buy)
thinkstats2.Corr(data4.Attrition1,data4.Total_Amt_Chng_Q4_Q1)
thinkstats2.Corr(data4.Attrition1,data4.Total_Ct_Chng_Q4_Q1)
thinkstats2.Corr(data4.Attrition1,data4.Card_Category1)

# EDA accuaracy moving to R
data4.to_csv(r'C:\Users\camer\OneDrive\Documents\ThinkStats2\code\data4.csv', index = False)


