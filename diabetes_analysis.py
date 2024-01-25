import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
import plotly.express as px
from scipy.stats import norm, f, ttest_ind

#about dataset
# Data is comparing clinical variables between those with and without diabetes
# Outcome: 0 is non-diabetic, 1 is diabetic

#importing dataset
data = pd.read_csv('diabetes copy.csv')
data.head()

#DATA CLEANING
datacopy = data

#checking for missing values
datacopy.isnull().sum()
#no null values

#finding the sum of the number of zeroes in each column 
for col in datacopy.columns:
    count = (datacopy[col]== 0).sum()
    print(f"{col}: {count}")
#There should not be any zero values in glucose, blood pressure, skin thickness, insulin, or BMI

#removing rows with 0 values in those columns
clean_data = datacopy.loc[(datacopy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] !=0).all(axis=1)]

#clean dataset
clean_data.head()
clean_data.tail()

#Summary
clean_data.describe()


#EXPLORATORY DATA ANALYSIS
#create boxplots to see distribution of values
#stratify by diabetes status

#diabetes dataset
db_data = clean_data.loc[clean_data["Outcome"] == 1]
db_data.head()
db_data.tail()

#non diabetic dataset
ndb_data = clean_data.loc[clean_data["Outcome"] == 0]
ndb_data.head()
ndb_data.tail()

def make_boxplot(data, variables, columns, color):
    ncols = columns
    nrows = int(np.ceil(len(variables)/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols = ncols)
    sns.set_style("dark")
    
    for i,var in enumerate(variables):
        row = int(i//ncols)
        col = i%ncols
        a = sns.boxplot(x=data[var], color = color, ax= axes[row,col])
        a.set_xlabel(var)

    plt.suptitle('Boxplots of Variables')
    plt.tight_layout()
    plt.show()

variables = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']

make_boxplot(db_data, variables, 3, "lightpink")
#there appears to be many outliers in the insulin variable

make_boxplot(ndb_data, variables, 3, "lightblue")
#there appears to be many outliers in the insulin and age variables


#number of diabetics vs non-diabetics in dataset
len(db_data) # 130 diabetics
len(ndb_data) # 262 non-diabetics

#side by side comparison of values based on "Outcome" using facetgrid

#initiate facetgrid object and split into columns based on outcome
def make_facetgrid(data, variables, col_to_split):
    for var in variables:
        g = sns.FacetGrid(data, col= col_to_split)
        g.map_dataframe(sns.boxplot, x= var, color = "lightblue")
        g.tight_layout()
        plt.show()

make_facetgrid(clean_data, variables, "Outcome")

#It appears that the group with diabetes appears to have a higher median number of pregnancies, higher blood glucose, diabetes pedigree function, skin thickness, insulin, BMI, and age
#Will need to do tests after to see if these differences are significant.


#histograms to look for normal distribution of continuous variables

def make_histogram(data, variables, bins, columns, color):
    ncols = columns
    nrows = int(np.ceil(len(variables)/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols = ncols)
    sns.set_style("dark")
    
    for i,var in enumerate(variables):
        row = int(i//ncols)
        col = i%ncols
        a = sns.histplot(x=data[var], color = color, bins=bins, ax= axes[row,col])
        a.set_xlabel(var)

    plt.suptitle('Histogram of Variables')
    plt.tight_layout()
    plt.show()

make_histogram(db_data, variables, 10, 3, 'lightpink')
make_histogram(ndb_data, variables, 10, 3, 'lightblue')

# it appears that age, insulin, and diabetes pedigree function are not normally distributed
# number of pregnancies is a discrete quantitative variable 

#log transformation of age, insulin, diabetes
variables_log = ["Age", "Insulin", "DiabetesPedigreeFunction"]

for var in variables_log:
    log_data = np.log(clean_data[var])
    sns.histplot(clean_data, x=log_data, color="lightblue", bins=10)
    plt.show()

#update dataset to have log form of insulin and diabetes pedigree function
log_insulin = np.log(clean_data["Insulin"])
clean_data["Insulin"] = log_insulin

log_dpf = np.log(clean_data["DiabetesPedigreeFunction"])
clean_data["DiabetesPedigreeFunction"] = log_dpf

log_age = np.log(clean_data["Age"])
clean_data["Age"] = log_age

#update diabetes and non-diabetic datasets
db_data = clean_data.loc[clean_data["Outcome"] == 1]
ndb_data = clean_data.loc[clean_data["Outcome"] == 0]

#histogram with transformed values
make_histogram(clean_data, variables, 10, 3, "blue")

make_histogram(db_data, variables, 10, 3, "lightpink")
make_histogram(ndb_data, variables, 10, 3, "lightblue")


#Comparing means between groups
#tests to see if differences between variables are significant

#F-test 
#to compare variances between groups
#This ensures that the groups have similar variances before performing a t-test

def f_test(group_1, group_2):
    #converting variables to numpy arrays
    group_1 = np.array(group_1)
    group_2 = np.array(group_2)
    
    #calculating F test statistic
    #get the variances of each group and divide them by eachother to get f test stat
    f_stat = np.var(group_1, ddof=1)/np.var(group_2, ddof=1) 
    
    #Finding P value of F test stat
    dfn = group_1.size-1 #degrees of freedom numerator 
    dfd = group_2.size-1 #degrees of freedom denominator 
    p = 1-f.cdf(f_stat, dfn, dfd) #finds p-value
    return f_stat, p 


f_variables = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Pregnancies']

#f test 
for var in f_variables:
    f_stat, p = f_test(db_data[var], ndb_data[var])
    print(f"{var}: f-test statistic,{f_stat}; p-value {p}\n")

#variables with a p value of less than 0.5 from the f-test
for var in f_variables:
    f_stat, p = f_test(db_data[var], ndb_data[var])
    if p < 0.05:
        print(f"{var} has a significantly different variance between groups: {p} \n")

#variances with signficant differences: glucose
#this will be accounted for during the t-test

#t test
#to compare the means of variables between the diabetic and non-diabetic groups
#assumptions: variables are continuous, follow a normal distribution, randomnly sampled, similar variances between groups(confirmed with f-test)
t_variables = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Pregnancies']

for var in t_variables:
    if var == "Glucose" or var == "Pregnancies":
        t, p = ttest_ind(db_data[var], ndb_data[var], equal_var=False)
        print(f"{var}: t-stat is {t}; p-value is {p}\n")
    else:
        t, p = ttest_ind(db_data[var], ndb_data[var])
        print(f"{var}: t-stat is {t}; p-value is {p}\n")

#variables that are significantly different between those with and without diabetes:
for var in t_variables:
    if var == "Glucose" or var == "Pregnancies":
        t, p = ttest_ind(db_data[var], ndb_data[var], equal_var=False)
        if p < 0.05:
            print(f"{var}")
    else:
        t, p = ttest_ind(db_data[var], ndb_data[var])
        if p < 0.05:
            print(f"{var}")
    
# It appears that there is a significant difference in means of glucose, blood pressure, skin thickness, insuloin, BMI, diabetes pedigree function, and age between those with and without diabetes.
db_data.mean()
ndb_data.mean()
#Those with diabetes have higher: pregnancies, glucose, blood pressure,  skin thickness, insulin, bmi, diabetes pedigree function, and age


#scatterplots to visualize variables, use hue to show differences based on gdm status
scatter_variables = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Pregnancies']

#want to look at differences in each variables based off glucose levels
#use hue for outcome to see impact of glucose level and outcome on variables
def make_scatter(data, variables, columns):
    ncols = columns
    nrows = int(np.ceil(len(variables)/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols = ncols)
    sns.set_style("dark")
    
    for i,var in enumerate(variables):
        row = int(i//ncols)
        col = i%ncols
        a = sns.scatterplot(
            clean_data,
            x="Glucose",
            y=var, 
            hue = "Outcome",
            size = var,
            ax=axes[row,col], 
            legend=False
        )
    plt.suptitle(f"Scatterplots of Glucose vs.{var}")
    plt.show()

make_scatter(clean_data, scatter_variables, 4)

#correlation test
corr_data = clean_data.drop(["Outcome"], axis=1)
corr_matrix = corr_data.corr(method="pearson")

#create heatmap
fig = sns.heatmap(corr_matrix, annot=True, cmap = "mako")
fig.set_title("Correlation Heatmap", fontsize = 12)

plt.tight_layout()
plt.show()
#It appears that the highest correlation is between age and pregnancies (0.7), BMI and skin thickness (0.66), and glucose and insulin (0.62).


#save dataset
clean_data.to_csv('clean_data.csv', index=False)






