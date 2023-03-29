
import numpy as np
import pandas as pd
from scipy.stats import f

def sum_squares_between(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the sum of squares between groups. 
    
    Args:
        x (array): Array with categorical variable. 
            For example: If factor is gender, x-array can be:
            - np.ndarray([0,0,1,1,0,]) where 0 is for boys and 1 for girls or 
            - np.ndarray(['boys'boys', 'girls', 'girls', 'boys'])
        y (array): Array with values for the dependent variable.

    Returns:
        float: The sum of squares of the error between the different groups. 
    """
    if (not type(x) == np.ndarray) or (not type(y) == np.ndarray):
        raise TypeError('x and y must be of type array')

    y_bar = np.mean(y)
    SSB = 0
    for j in np.unique(x):
        
        # Get subset for each category
        filt = (x == j)
        y_vals = y[filt]
        
        # Calculate SSB for each group and add to total
        SSB_j = len(y_vals) * (np.mean(y_vals) - y_bar)**2
        SSB += SSB_j
        
    return SSB   

def sum_squares_within_one_way(
        x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the sum of squares between groups for one-way ANOVA.

    Args:
        x (array): Array with categorical variable.
        y (array): Array with values for the dependent variable
       

    Returns:
        float: The sum of squares of the error within the different groups. 
    """
    if not type(x) == np.ndarray or not type(y) == np.ndarray:
        raise TypeError('x and y must be of type array')

    SSW = 0
    for j in np.unique(x):
                   
        # Get subset for each combination between factors 1 and 2
        filt = (x == j)
        y_vals = y[filt]
        
        SSW_j = np.sum((y_vals - np.mean(y_vals))**2)
        SSW += SSW_j
    
    return SSW

def sum_squares_within_two_way(
        x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the sum of squares between groups for two-way ANOVA.

    Args:
        x1 (array): Array with categorical variable for factor 1.
        x2 (array): Array with categorical variable for factor 2.
        y (array): Array with values for the dependent variable
       

    Returns:
        float: The sum of squares of the error within the different groups. 
    """
    if (not type(x1) == np.ndarray or
        not type(x2) == np.ndarray or not type(y) == np.ndarray): 
        raise TypeError('x and y must be of type array')

    SSW = 0
    for j in np.unique(x1):
        for k in np.unique(x2):
            
            # Get subset for each combination between factors 1 and 2
            filt = (x1 == j) & (x2 == k)
            y_vals = y[filt]
            
            SSW_j_k = np.sum((y_vals - np.mean(y_vals))**2)
            SSW += SSW_j_k
    
    return SSW

def sum_squares_total(y: np.ndarray) -> float:
    """
    Computes the total sum of squares for a numeric array. 
    
    Args:
        y (array): Array with values for the dependent variable
       
    Returns:
        float: The sum of squares of the error within the different groups.       
    """
    if not type(y) == np.ndarray:
        raise TypeError('y must be of type array')

    y_bar = np.mean(y)
    return np.sum([(val - y_bar)**2 for val in y])


def get_anova_table(
        x1: np.ndarray, x2: np.ndarray, y: np.ndarray, alpha: float = 0.05,
        names: list[str] = ['Factor_A','Factor_B']) -> pd.DataFrame:

    """
        Args:
        x1 (array): Array with categorical variable for factor 1.
        x2 (array): Array with categorical variable for factor 2.
        y (array): Array with values for the dependent variable.
        alpha (float): It defines how strongly the sample evidence must
            contradict the null hypothesis before you can reject the null 
            hypothesis for the entire population. Defaults to 0.05 (i.e., 5%).
        names (list): List of strings with the names of each factor.
    """

    SSB_A = sum_squares_between(x1, y)
    SSB_B = sum_squares_between(x2, y)
    SSE = sum_squares_within_two_way(x1, x2, y)
    SST = sum_squares_total(y)
    SSB_AB = SST - SSB_A - SSB_B - SSE

    # Check computations
    # Recall SST = SSB_A + SSB_B + SSB_AB + SSE
    assert abs(SST - (SSB_A + SSB_B + SSB_AB + SSE)) < 1*10**-6

    N = len(y)
    a = len(np.unique(x1))
    b = len(np.unique(x2))

    #Create DataFrame
    anova_table = pd.DataFrame({
        'Source': [names[0], names[1], 'Interaction', 'Error', 'Total'],
        'SS': [SSB_A, SSB_B, SSB_AB, SSE, SST],
        'df': [(a-1),  (b-1), (a-1)*(b-1), (N-a*b), (N-1)]})
    anova_table.set_index('Source', inplace = True)

    # Add Mean Square and F-statistic columns
    anova_table['Mean Square'] = anova_table['SS'] / anova_table['df']
    anova_table['F_statistic'] = \
        anova_table['Mean Square'] / anova_table.loc['Error','Mean Square']

    # Aesthetic (i.e., we dont want these cells to have values)
    anova_table.loc['Total', ['Mean Square', 'F_statistic']] = ""
    anova_table.loc['Error',  'F_statistic'] = ""

    # Compute Critical F-statistic and also Reject/No reject Decisions
    f_rejection = []
    rejection = []
    p_val = []
    for idx, row in anova_table.iterrows():
        if idx not in ['Error', 'Total']:
            
            # Calculate p_value and F_critical
            p_value = f.sf(row['F_statistic'], row['df'], (N-a*b))
            f_reject = f.isf(alpha, row['df'], (N-a*b), loc=0, scale=1)

            p_val.append(p_value)
            f_rejection.append(f_reject)
            
            if row['F_statistic'] > f_reject:
                rejection.append("Reject Ho")
            else:
                rejection.append("Fail Reject Ho")
        else:
            f_rejection.append("")
            rejection.append("")
            p_val.append("")

    anova_table['F_critical'] = f_rejection
    anova_table['p_value'] = p_val
    anova_table['Decision'] = rejection

    return anova_table