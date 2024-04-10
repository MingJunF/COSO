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
    k = 0

    # Chose all variables except target itself
    PC = [i for i in range(kVar) if i != target]
    # Dictionary to store p-values for variables that remain after the test
    relevant_pvals = {}

    while len(PC) > k:

        PC_temp = PC.copy()
        for x in PC_temp:
            condition_subsets = [i for i in PC_temp if i != x]
            print(x)
            if len(condition_subsets) >= k:
                css = subsets(condition_subsets, k)
                for s in css:
                    pval, dep = cond_indep_test(data, x, target, s, isdiscrete)
                    ciTest += 1
                    if pval > alpha:
                        PC.remove(x)
                        break  # If variable is independent, it's removed and we stop checking more subsets
                    else:
                        # For variables that are not removed, update their p-value
                        relevant_pvals[x] = pval
        k += 1

    # Filter the relevant_pvals dictionary to include only those variables that are still in PC
    final_pvals = {var: pval for var, pval in relevant_pvals.items() if var in PC}

    return PC, ciTest, final_pvals



# use demo to test pc_simple
# data = pd.read_csv("C:\pythonProject\pyCausalFS\CBD\data\Child_s500_v1.csv")
# target=12
# alaph=0.01
# pc=pc_simple(data,target,alaph, True)
# print(pc)
