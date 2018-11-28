import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

def getReduceMemDataFrame(gl):
    if isinstance(gl,pd.DataFrame):
    # downcast int64 to unsigned
        gl_int = gl.select_dtypes(include=['int64'])
        converted_int = gl_int.apply(pd.to_numeric,downcast='unsigned')

        # print(mem_usage(gl_int))
        # print(mem_usage(converted_int))
        # downcast float64 to float
        gl_float = gl.select_dtypes(include=['float64'])
        converted_float = gl_float.apply(pd.to_numeric,downcast='float')

        # print(mem_usage(gl_float))
        # print(mem_usage(converted_float))

        optimized_gl = gl.copy()
        optimized_gl[converted_int.columns] = converted_int
        optimized_gl[converted_float.columns] = converted_float

        # print(mem_usage(gl))
        # print(mem_usage(optimized_gl))

        # downcast object to category
        # optimized_gl = gl.copy()
        converted_obj = pd.DataFrame()
        gl_obj = gl.select_dtypes(include=['object']).copy()
        for col in gl_obj.columns:
            num_unique_values = len(gl_obj[col].unique())
            num_total_values = len(gl_obj[col])
            if num_unique_values / num_total_values < 0.5:
                converted_obj.loc[:,col] = gl_obj[col].astype('category')
            else:
                converted_obj.loc[:,col] = gl_obj[col]

        optimized_gl[converted_obj.columns] = converted_obj
        # print(mem_usage(optimized_gl))
        return optimized_gl
    else:
        return -1
if __name__ == '__main__':
    gl = pd.read_csv('traindata/operation_train_new.csv')
    # print(gl.head())
    # gl.info(memory_usage='deep')
    optimized_gl = getReduceMemDataFrame(gl)
    # print(mem_usage(optimized_gl))
    # print(optimized_gl['day'][950:1000])
    # print(gl['day'].head())

#   show scatter point day
    day_UID = optimized_gl.pivot_table(index='day', values='UID')
    day_UID.reset_index().plot.scatter('day','UID')
    plt.show()