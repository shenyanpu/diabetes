from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

#loading the dataset
diabetes_data = pd.read_csv('diabetes.csv')

#print the first 5 rows of the dataframe
diabetes_data.head()

### Basic EDA and statistical analysis ###
## gives information about the data types,columns, null value counts, memory usage etc
## function reference : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html
diabetes_data.info(verbose=True)

'''
DataFrame.describe() method generates descriptive statistics that summarize the central 
tendency, dispersion and shape of a dataset’s distribution, excluding NaN values. 
This method tells us a lot of things about a dataset. One important thing is that the 
describe() method deals only with numeric values. It doesn't work with any categorical 
values. So if there are any categorical values in a column the describe() method will 
ignore it and display summary for the other columns unless parameter include="all" is passed.
'''
## basic statistic details about the data (note only numerical columns would be displayed here unless parameter include="all")
## for reference: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html#pandas.DataFrame.describe
diabetes_data.describe(include='all')

diabetes_data.describe().T

'''
The Question creeping out of this summary
Can minimum value of below listed columns be zero (0)?
On these columns, a value of zero does not make sense and thus indicates missing value.

Following columns or variables have an invalid zero value:

Glucose 葡萄糖
BloodPressure
SkinThickness
Insulin 胰岛素
BMI

It is better to replace zeros with nan since after that counting them would be easier and zeros need to be replaced with suitable values
'''

diabetes_data_copy=diabetes_data.copy(deep=True)
# When deep=True (default), a new object will be created with a copy of the calling object’s data and indices. 
# Modifications to the data or indices of the copy will not be reflected in the original object (see notes below).

diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(diabetes_data_copy.isnull().sum())

#To fill these Nan values the data distribution needs to be understood

p=diabetes_data.hist(figsize=(20,20))

#Aiming to impute nan values for the columns in accordance with their distribution
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace=True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)

# Plotting after Nan removal
p = diabetes_data_copy.hist(figsize = (20,20))

#Skewness
'''
A left-skewed distribution has a long left tail. Left-skewed distributions are also called negatively-skewed distributions. That’s because there is a long tail in the negative direction on the number line. The mean is also to the left of the peak.

A right-skewed distribution has a long right tail. Right-skewed distributions are also called positive-skew distributions. That’s because there is a long tail in the positive direction on the number line. The mean is also to the right of the peak.
'''

## observing the shape of the data
diabetes_data.shape

## data type analysis
plt.figure(figsize=(5,5))
sns.set(font_scale=2)
sns.countplot(y=diabetes_data.dtypes ,data=diabetes_data)
plt.xlabel("count of each data type")
plt.ylabel("data types")
plt.show()

## null count analysis
import missingno as msno
p=msno.bar(diabetes_data)

## checking the balance of the data by plotting the count of outcomes by their value
color_wheel = {1:"#0392cf", 2: "#7bc043"}
colors = diabetes_data["Outcome"].map(lambda x: color_wheel.get(x + 1))
print(diabetes_data.Outcome.value_counts())
p=diabetes_data.Outcome.value_counts().plot(kind="bar")

'''
The above graph shows that the data is biased towards datapoints having outcome value 
as 0 where it means that diabetes was not present actually. 
The number of non-diabetics is almost twice the number of diabetic patients
'''

### Scatter matrix of uncleaned data
from pandas.plotting import scatter_matrix
p=scatter_matrix(diabetes_data, figsize=(25,25))

'''
The pairs plot builds on two basic figures, the histogram and the scatter plot. 
The histogram on the diagonal allows us to see the distribution of a single 
variable while the scatter plots on the upper and lower triangles show the 
relationship (or lack thereof) between two variables
'''

p=sns.pairplot(diabetes_data_copy,hue='Outcome')

'''
Pearson's Correlation Coefficient: helps you find out the relationship between two quantities. 
It gives you the measure of the strength of association between two variables. 
The value of Pearson's Correlation Coefficient can be between -1 to +1. 
1 means that they are highly correlated and 0 means no correlation.
'''

### Heatmap for unclean data
'''
A heat map is a two-dimensional representation of information with the help of colors. 
Heat maps can help the user visualize simple or complex information.
'''
plt.figure(figsize=(25,20))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(diabetes_data.corr(),annot=True,cmap='RdYlGn')