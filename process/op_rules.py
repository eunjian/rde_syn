import pandas as pd

def operation(
    df:pd.DataFrame,
    target:str,
    equation:str,
):
    df[target] = df.eval(equation)
    return df