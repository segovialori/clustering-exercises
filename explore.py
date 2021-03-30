import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            # stratify=df[target]
                                           )
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       # stratify=train_validate[target]
                                      )
    return train, validate, test

###################### ________________________________________
### Univariate

def explore_univariate(train, cat_vars, quant_vars):
    '''
    explore each individual categorical variably by: 
    taking in a dataframe and a categorical variable and returning
    a frequency table and barplot of the frequencies. 
    
    explore each individual quantitative variable by:
    takes in a dataframe and a quantitative variable and returns
    descriptive stats table, histogram, and boxplot of the distributions. 
    '''
    for var in cat_vars:
        explore_univariate_categorical(train, var)
        print('_________________________________________________________________')
    for col in quant_vars:
        p, descriptive_stats = explore_univariate_quant(train, col)
        plt.show(p)
        print(descriptive_stats)

def explore_univariate_categorical(train, cat_var):
    '''
    takes in a dataframe and a categorical variable and returns
    a frequency table and barplot of the frequencies. 
    '''
    frequency_table = freq_table(train, cat_var)
    plt.figure(figsize=(2,2))
    sns.barplot(x=cat_var, y='Count', data=frequency_table, color='lightseagreen')
    plt.title(cat_var)
    plt.show()
    print(frequency_table)

def explore_univariate_quant(train, quant_var):
    '''
    takes in a dataframe and a quantitative variable and returns
    descriptive stats table, histogram, and boxplot of the distributions. 
    '''
    descriptive_stats = train[quant_var].describe()
    plt.figure(figsize=(8,2))

    p = plt.subplot(1, 2, 1)
    p = plt.hist(train[quant_var], color='lightseagreen')
    p = plt.title(quant_var)

    # second plot: box plot
    p = plt.subplot(1, 2, 2)
    p = plt.boxplot(train[quant_var])
    p = plt.title(quant_var)
    return p, descriptive_stats
    
def freq_table(train, cat_var):
    '''
    for a given categorical variable, compute the frequency count and percent split
    and return a dataframe of those values along with the different classes. 
    '''
    class_labels = list(train[cat_var].unique())

    frequency_table = (
        pd.DataFrame({cat_var: class_labels,
                      'Count': train[cat_var].value_counts(normalize=False), 
                      'Percent': round(train[cat_var].value_counts(normalize=True)*100,2)}
                    )
    )
    return frequency_table

###################### ________________________________________
#### Bivariate


def explore_bivariate(train, categorical_target, continuous_target, binary_vars, quant_vars):
    '''
    This function makes use of explore_bivariate_categorical and explore_bivariate_quant functions. 
    Each of those take in a continuous target and a binned/cut version of the target to have a categorical target. 
    the categorical function takes in a binary independent variable and the quant function takes in a quantitative 
    independent variable. 
    '''
    
    for binary in binary_vars:
        explore_bivariate_categorical(train, categorical_target, continuous_target, binary)
    for quant in quant_vars:
        explore_bivariate_quant(train, categorical_target, continuous_target, quant)

###################### ________________________________________
## Bivariate Categorical

def explore_bivariate_categorical(train, categorical_target, continuous_target, binary_var):
    '''
    takes in binary categorical variable and binned/categorical target variable, 
    returns a crosstab of frequencies
    runs a chi-square test for the proportions
    and creates a barplot, adding a horizontal line of the overall rate of the binary categorical variable. 
    '''
    print(binary_var, "\n_____________________\n")
    ct = pd.crosstab(train[binary_var], train[categorical_target], margins=True)
    chi2_summary, observed, expected = run_chi2(train, binary_var, categorical_target)
    mannwhitney = compare_means(train, continuous_target, binary_var, alt_hyp='two-sided')
    p = plot_cat_by_target(train, categorical_target, binary_var)
    print("\nMann Whitney Test Comparing Means: ", mannwhitney)
    print(chi2_summary)
    print("\nobserved:\n", ct)
    print("\nexpected:\n", expected)
    plt.show(p)
    print("\n_____________________\n")
    

    
def run_chi2(train, binary_var, categorical_target):
    observed = pd.crosstab(train[binary_var], train[categorical_target])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    chi2_summary = pd.DataFrame({'chi2': [chi2], 'p-value': [p], 
                                 'degrees of freedom': [degf]})
    expected = pd.DataFrame(expected)
    return chi2_summary, observed, expected


def plot_cat_by_target(train, categorical_target, binary_var):
    p = plt.figure(figsize=(2,2))
    p = sns.barplot(categorical_target, binary_var, data=train, alpha=.8, color='lightseagreen')
    overall_rate = train[binary_var].mean()
    p = plt.axhline(overall_rate, ls='--', color='gray')
    return p

def compare_means(train, continuous_target, binary_var, alt_hyp='two-sided'):
    x = train[train[binary_var]==0][continuous_target]
    y = train[train[binary_var]==1][continuous_target]
    return stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)

###################### ________________________________________
## Bivariate Quant

def explore_bivariate_quant(train, categorical_target, continuous_target, quant_var):
    '''
    descriptive stats by each target class. 
    boxenplot of target x quant
    swarmplot of target x quant
    Scatterplot
    '''
    print(quant_var, "\n____________________\n")
    descriptive_stats = train.groupby(categorical_target)[quant_var].describe().T
    spearmans = compare_relationship(train, continuous_target, quant_var)
    plt.figure(figsize=(4,4))
    boxen = plot_boxen(train, categorical_target, quant_var)
    swarm = plot_swarm(train, categorical_target, quant_var)
    plt.show()
    scatter = plot_scatter(train, categorical_target, continuous_target, quant_var)
    plt.show()
    print(descriptive_stats, "\n")
    print("\nSpearman's Correlation Test:\n", spearmans)
    print("\n____________________\n")


def compare_relationship(train, continuous_target, quant_var):
    return stats.spearmanr(train[quant_var], train[continuous_target], axis=0)

def plot_swarm(train, categorical_target, quant_var):
    average = train[quant_var].mean()
    p = sns.swarmplot(data=train, x=categorical_target, y=quant_var, color='lightgray')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_boxen(train, categorical_target, quant_var):
    average = train[quant_var].mean()
    p = sns.boxenplot(data=train, x=categorical_target, y=quant_var, color='lightseagreen')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_scatter(train, categorical_target, continuous_target, quant_var):
    p = sns.scatterplot(x=quant_var, y=continuous_target, hue=categorical_target, data=train)
    p = plt.title(quant_var)
    return p


######################### ____________________________________

### Multivariate


def explore_multivariate(train, categorical_target, binary_var, quant_vars):
    '''
    '''
    plot_swarm_grid_with_color(train, categorical_target, binary_var, quant_vars)
    violin = plot_violin_grid_with_color(train, categorical_target, binary_var, quant_vars)
    plt.show()
    pair = sns.pairplot(data=train, vars=quant_vars, hue=categorical_target)
    plt.show()
    plot_all_continuous_vars(train, categorical_target, quant_vars)
    plt.show()    


def plot_all_continuous_vars(train, categorical_target, quant_vars):
    '''
    Melt the dataset to "long-form" representation
    boxenplot of measurement x value with color representing survived. 
    '''
    my_vars = [item for sublist in [quant_vars, [categorical_target]] for item in sublist]
    sns.set(style="whitegrid", palette="muted")
    melt = train[my_vars].melt(id_vars=categorical_target, var_name="measurement")
    plt.figure(figsize=(8,6))
    p = sns.boxenplot(x="measurement", y="value", hue=categorical_target, data=melt)
    p.set(yscale="log", xlabel='')    
    plt.show()
    
def plot_violin_grid_with_color(train, categorical_target, binary_var, quant_vars):
    for quant in quant_vars:
        sns.violinplot(x=categorical_target, y=quant, data=train, split=True, hue=binary_var, palette="Set2")
        plt.show()
        
def plot_swarm_grid_with_color(train, categorical_target, binary_var, quant_vars):
    for quant in quant_vars:
        sns.swarmplot(x=categorical_target, y=quant, data=train, split=True, hue=binary_var, palette="Set2")
        plt.show()
                