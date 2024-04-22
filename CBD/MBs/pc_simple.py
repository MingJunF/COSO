# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/9 15:00
desc:
"""
import numpy as np
from CBD.MBs.common.condition_independence_test import cond_indep_test
from CBD.MBs.common.subsets import subsets

def pc_simple(data, target, alpha, isdiscrete):
    number, kVar = np.shape(data)
    ciTest = 0

    # Chose all variables except target itself
    PC = [i for i in range(kVar) if i != target]
    # Dictionary to store p-values for variables that remain after the test
    relevant_pvals = {}

    # Create the maximum condition set once for each variable and reuse it
    condition_sets = {x: [i for i in PC if i != x] for x in PC}

    PC_temp = PC.copy()
    for x in PC_temp:
        condition_set = condition_sets[x]
        # Since we're using the full set of other variables, there's no need for subsets
        pval, dep = cond_indep_test(data, x, target, condition_set, isdiscrete)
        ciTest += 1
        relevant_pvals[x] = pval
        if pval > alpha:
            PC.remove(x)
        #print('relevant_pvals', relevant_pvals)
    # Filter the relevant_pvals dictionary to include only those variables that are still in PC
    final_pvals = {var: pval for var, pval in relevant_pvals.items() if var in PC}

    return PC, ciTest, relevant_pvals




# use demo to test pc_simple
# data = pd.read_csv("C:\pythonProject\pyCausalFS\CBD\data\Child_s500_v1.csv")
# target=12
# alaph=0.01
# pc=pc_simple(data,target,alaph, True)
# print(pc)
