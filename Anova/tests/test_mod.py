import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Internal Libraries
import anova_compute as anv

def test_sum_squares_between():
    # Test 1 Data
    gender_vals = np.array([0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1])
    score_vals = np.array([4,6,8,4,8,9,6,6,9,7,10,13,8,9,13,12,14,16])
    age_vals = np.array([10,10,10,10,10,10,11,11,11,11,11,11,12,12,12,12,12,12])

    SSB_gender = anv.sum_squares_between(gender_vals, score_vals)
    assert abs(SSB_gender - 32) < 1*10**-7

    SSB_age = anv.sum_squares_between(age_vals, score_vals)
    assert abs(SSB_age - 93) < 1*10**-7

def test_sum_squares_between_2():

    treatment_1 = [8,9,6,8,5]
    treatment_2 = [5,4,7,6,6]
    treatment_3 = [9,3,2,4]
    scores = np.array([*treatment_1, *treatment_2, *treatment_3])
    treatment_number = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3])
    SSB = anv.sum_squares_between(treatment_number, scores)
    assert abs(SSB - 16.71428571428) < 1*10**-7

def test_sum_squares_within_one_way():
    treatment_1 = [8,9,6,8,5]
    treatment_2 = [5,4,7,6,6]
    treatment_3 = [9,3,2,4]
    scores = np.array([*treatment_1, *treatment_2, *treatment_3])
    treatment_number = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3])
    SSW = anv.sum_squares_within_one_way(treatment_number, scores)
    assert abs(SSW - 45.0) < 1*10**-7

def test_sum_squares_within_two_way():
    gender_vals = np.array([0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1])
    score_vals = np.array([4,6,8,4,8,9,6,6,9,7,10,13,8,9,13,12,14,16])
    age_vals = np.array([10,10,10,10,10,10,11,11,11,11,11,
                         11,12,12,12,12,12,12])

    SSW = anv.sum_squares_within_two_way(
            x1 = gender_vals, x2 = age_vals,  y = score_vals)
    print(SSW)
    assert abs(SSW - 68.0) < 1*10**-7

    SSW = anv.sum_squares_within_two_way(
            x1 = age_vals, x2 = gender_vals,  y = score_vals)
    assert abs(SSW - 68.0) < 1*10**-7

def test_sum_squares_total():
    gender_vals = np.array([0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1])
    score_vals = np.array([4,6,8,4,8,9,6,6,9,7,10,13,8,9,13,12,14,16])
    age_vals = np.array([10,10,10,10,10,10,11,11,11,11,11,
                         11,12,12,12,12,12,12])

    SST = anv.sum_squares_total(y = score_vals)
    print(SST)
    assert abs(SST - 200.0) < 1*10**-7

def test_get_anova_table():

    filepath = os.path.join(os.path.dirname(__file__), 'test_data.csv')
    print(filepath)
    # Read Test Data
    df = pd.read_csv(filepath)
    y = df['grade'].to_numpy()
    x1 = df['program'].to_numpy()
    x2 = df['gender'].to_numpy()
 
    df = anv.get_anova_table(x1, x2, y, alpha =0.05, 
                    names = ['program', 'gender'])

    assert abs(df.loc['program', 'SS'] - 233.244) < 0.001
    assert abs(df.loc['Interaction', 'SS'] - 5317.06) < 0.01
    assert abs(df.loc['gender', 'Mean Square'] - 38.92) < 0.01
    assert abs(df.loc['Error', 'Mean Square'] - 18.67) < 0.01
    assert abs(df.loc['program', 'F_statistic'] - 12.49) < 0.01
    assert abs(df.loc['Interaction', 'F_statistic'] - 284.73) < 0.01
    assert abs(df.loc['program', 'p_value'] - 0.0007) < 0.0001
    assert abs(df.loc['gender', 'p_value'] - 0.153) < 0.001

def test_get_anova_table_2():

    # Test Data
    sales = np.array([47,43,46,40,62,68,67,71,41,39,42,46])
    height = np.array([1,1,1,1,2,2,2,2,3,3,3,3])
    width = np.array([1,1,2,2,1,1,2,2,1,1,2,2])

    df = anv.get_anova_table(height, width, sales, alpha =0.05, 
                    names = ['height', 'width'])
    print(df)
    assert abs(df.loc['height', 'SS'] - 1544.0) < 0.01
    assert abs(df.loc['width', 'SS'] - 12.0) < 0.01
    assert abs(df.loc['Interaction', 'SS'] - 24) < 0.01
    assert abs(df.loc['Error', 'SS'] - 62) < 0.01
    assert abs(df.loc['Total', 'SS'] - 1642) < 0.01
    assert abs(df.loc['height', 'Mean Square'] - 772) < 0.01
    assert abs(df.loc['width', 'Mean Square'] - 12) < 0.01
    assert abs(df.loc['Error', 'Mean Square'] - 10.33) < 0.01
    assert abs(df.loc['height', 'F_statistic'] - 74.71) < 0.01
    assert abs(df.loc['width', 'F_statistic'] - 1.16) < 0.01
    assert abs(df.loc['Interaction', 'F_statistic'] - 1.16) < 0.01
    assert abs(df.loc['height', 'p_value'] - 0.0001) < 0.0001
    assert abs(df.loc['width', 'p_value'] - 0.32) < 0.01
    assert abs(df.loc['Interaction', 'p_value'] - 0.37) < 0.01