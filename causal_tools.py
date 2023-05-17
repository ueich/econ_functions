# this is a selection of usefule causal inference funcitons implemented in python


import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm



def did(df, var_outcome, var_treat, var_time, treat, post, vars_control = None, hc3_se=True):
    """
    Calculates the difference-in-differences (DiD) estimate for a time-series in a panel dataset.
    Can handle control variables and clustered standard errors.
    
    Parameters
    ----------
    df:            pandas.DataFrame
                        the panel dataset to analyse. NA values are dropped
    var_outcome:   str
                        the name of the column containing the outcome variable
    var_treat:     str
                        the name of the column containing the treatment assignment
    var_time:      str
                        the name of the column containing the time variable (variable must be a datetime object)
    treat:         str / int / bool
                        value of var_treat that identifies the treatment group
    post:          str of date
                        value of var_time of the policy change, after this value is the "POST" period
    vars_control:  list of str, optional
                        names of the columns containing the control variables. Default is None.
    hc3_se         bool, optional
                        bool indicating whether hc3_se (heteroskadasticity robust SE) are reported. Default is True.
    
    Returns
    -------
    statsmodels.regression.linear_model.OLSResults
        The linear DiD model estimated with `sm.OLS`.
    
    
    Example
    -------
    >>> did(df=df_act_up_agg, var_outcome="PRICE", var_treat="CATEGORY", var_time="START_TIME_UTC", treat="PRICE_TERT_W", post='2022-08-23 22:00:00')

        
    
    """
    df_did = df.dropna().copy()
    df_did["post"] = 0
    df_did.loc[df_did[var_time] >= post, "post"] = 1 # POST is after the policy change
    df_did["treatment"] = 0
    df_did.loc[df_did[var_treat] == treat, "treatment"] = 1
    df_did["post_treat"] = df_did["treatment"] * df_did["post"] # create interaction term
    X = ["treatment", "post", "post_treat"]
    if vars_control is not None:
        X = X+vars_control
        lm = sm.OLS(df_did[var_outcome], sm.add_constant(df_did[X])).fit()
    if hc3_se:
        lm = sm.OLS(df_did[var_outcome], sm.add_constant(df_did[X])).fit(cov_type="HC3")
        
    return(lm)



def event_study_q(df, var_outcome, var_treat, var_time, treat, baseline, vars_control = None, hc3_se=True):
    """
    Estimate a quarterly event-window regression model.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data.
    var_outcome : str
        The name of the dependent variable.
    var_treat : str
        The name of the treatment variable.
    var_time : str
        The name of the time variable. Must be a datetime variable with several quarters
    treat : str / int / bool
        The value of var_treat that identifies the treatment group
    baseline : str
        The baseline quarter.
    vars_control : list or None, optional
        A list of names of the control variables. Default is None.
    hc3_se : bool, optional
        Whether to use the HC3 standard errors. Default is True.

    Returns
    -------
    lm : statsmodels.regression.linear_model.RegressionResultsWrapper
        The results of the regression.
    quarterlist : array-like
        The list of quarters used in the regression.

    Examples
    --------
    >>> lmprice, quarterlist = event_study_q(df_act_up_agg.dropna(), var_outcome="PRICE", var_treat="CATEGORY", var_time="START_TIME_UTC", treat="PRICE_TERT_W", baseline="20222")

    """
    df_ew = df.copy()
    # Create regressors
    df_ew["month"] = df_ew[var_time].dt.strftime("%m")
    df_ew["quarter"] = np.where(df_ew["month"]<"04", "1", np.where(df_ew["month"]<"07", "2", np.where(df_ew["month"]<"10", "3", "4")))
    df_ew["quarter"] = df_ew[var_time].dt.strftime("%Y") + df_ew["quarter"]
    df_ew["treatment"] = 0
    df_ew.loc[df_ew[var_treat] == treat, "treatment"] = 1
    quarterlist = df_ew["quarter"].unique()
    df_ew = pd.get_dummies(data=df_ew, columns=["quarter"])
    # interactions
    for i in quarterlist:
        df_ew["inter_"+str(i)] = df_ew["treatment"]*df_ew["quarter_"+str(i)]
    df_ew = df_ew.drop(columns=["quarter_"+baseline,"inter_"+baseline]) # drop quarter as baseline 
    lm = sm.OLS(df_ew[var_outcome], sm.add_constant(df_ew.loc[:,"treatment":"inter_"+str(quarterlist[-1])])).fit()    
    if vars_control is not None:
        lm = sm.OLS(df_ew[var_outcome], sm.add_constant(df_ew[vars_control].join(df_ew.loc[:,"treatment":"inter_"+str(quarterlist[-1])]))).fit()
    if hc3_se:
        lm = sm.OLS(df_ew[var_outcome], sm.add_constant(df_ew.loc[:,"treatment":"inter_"+str(quarterlist[-1])])).fit(cov_type="HC3")
        if vars_control is not None:
            lm = sm.OLS(df_ew[var_outcome], sm.add_constant(df_ew[vars_control].join(df_ew.loc[:,"treatment":"inter_"+str(quarterlist[-1])]))).fit(cov_type="HC3")
    return(lm, quarterlist)






def coefplot(lm, xticklab="", xlab="", ylab="", start=""):
    """
    Plots the coefficients of a regression separately with the confidence interval defined in the model.

    Parameters
    ----------
    lm : statsmodels.regression.linear_model.OLSResults
        The linear model estimated with `sm.OLS`.
    xticklab : list of str, optional
        The desired labels on the x-axis ticks.
    xlab : str, optional
        The label for the x-axis.
    ylab : str, optional
        The label for the y-axis.
    start : string or tuple, optional
        Start string or tuple of several strings of the variables of the coefficients to plot. If left empty, all coefficients are plotted.

    Returns
    -------
    None.

    Examples
    --------
    >>> coefplot(lmprice, xticklab=['Tertiary', 'Tert*Post'], start=("treatment", "post_treat"), ylab="price CHF") --> this is an example with the output of the DiD funciton
    >>> coefplot(lmprice,start="inter", xticklab=[x for x in quarterlist if x != "20222"]) --> this is an example with the output of the event_study_q function
    """
    # Code for plotting the regression coefficients
    error = lm.params - lm.conf_int()[0]
    c_df = pd.DataFrame({'coef': lm.params.values[1:],
                            'err': error.values[1:],
                            'var': error.index.values[1:]
                           })
    c_df = c_df.loc[c_df["var"].str.startswith(start)]

    fig, ax = plt.subplots(figsize=(8, 5))
    c_df.plot(x='var', y='coef', kind='bar', 
                 ax=ax, color='none', 
                 yerr='err', legend=False)
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    ax.scatter(x=pd.np.arange(c_df.shape[0]), 
               marker='s', s=120, 
               y=c_df['coef'], color='black')
    ax.axhline(y=0, linestyle='--', color='black', linewidth=4)
    ax.xaxis.set_ticks_position('none')
    _ = ax.set_xticklabels(xticklab, 
                           rotation=0, fontsize=16)
    
    
