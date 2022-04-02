import io

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, pyplot

from mysite.settings import ML_DIR


def plot_fi(model, f, y, k=10, sort=False, ax=None, fig_show=True, save_dir=ML_DIR, fig_kwargs=None,
            bar_kwargs=None):
    """
    Plot feature importance

    This function plots the feature importance of an input <model> trained with the features
    <f> (feature names) and a label <y> (label name). It shows the first <k> features, which
    may or may not be sorted (according to <sort>). If <ax> is provided, new figure and axes
    are not created. The figure can be shown (according to <fig_show>) and stored locally
    (according to <save_as>). Additional figure settings are provided in <fig_kwargs>.
    The bar-graph settings are provided in <bar_kwargs>.

    For more information about the used bar-graph function, see:
    https://seaborn.pydata.org/generated/seaborn.barplot.html

    Parameters
    ----------

    model : obj
        tree-based object supporting <model>.feature_importance_ (e.g. XGBoost)

    f : list
        list with the feature names

    y : str
        str with name of the label (dependent variable: y)

    k : int, optional, default 10
        int value with number of important features to plot

    sort : bool, optional, default False
        boolean flag for plotting the sorted features

    ax : matplotlib.axes, optional, default None
        axes object

    fig_show : bool, optional, default True
        boolean flag for figure showing

    save_dir : str, optional, default {ML_DIR}
        str with the dir to store the figure into

    fig_kwargs : dict, optional, default None
        dict with additional figure settings

    bar_kwargs : dict, optional, default None
        dict with additional bar-graph settings

    Returns
    -------

    Tuple with axes object and the list of dicts with the first <k> features and importance
    """

    set_visual_styles()

    # Prepare the figure settings
    fig_kwargs = fig_kwargs if fig_kwargs else {
        "fig_size": (10, 10),
        "show_ticks": True,
        "ticks_rotation": 90,
        "x_label": "",
        "y_label": "",
        "title": f"feature importance: {y}"
    }

    # Prepare the bar-graph settings
    bar_kwargs = bar_kwargs if bar_kwargs else {"color": "#3b5b92", "edgecolor": "0.2"}

    # Get the feature importance(s)
    features = list(zip(model.feature_importances_, f))
    features = sorted(features, reverse=True) if sort else features

    if save_dir:
        # Get all features and store them
        all_features = [{"feature": feature, "importance": importance} for importance, feature in features]
        df_all_feat = pd.DataFrame(all_features)
        df_all_feat.to_excel(f'{save_dir}/features.xlsx', index=False)

    # Select limited features to plot
    features = features[:k if k < len(features) else len(features)] if k else features
    features = [{"feature": feature, "importance": importance} for importance, feature in features]

    # Create temporary DataFrame
    df_temp = pd.DataFrame(features)

    # Create figure if axes not inserted
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=fig_kwargs.get("fig_size"), sharex=False)

    # Create the barplot
    h = sns.barplot(x="feature", y="importance", data=df_temp, ax=ax, **bar_kwargs)

    # Set up the final adjustments
    h.set(xlabel=fig_kwargs.get("x_label"))
    h.set(ylabel=fig_kwargs.get("y_label"))
    h.set(title=fig_kwargs.get("title"))

    if fig_kwargs.get("show_ticks"):
        h.set_xticklabels(ax.get_xticklabels(), rotation=fig_kwargs.get("ticks_rotation"))
    else:
        h.set_xticklabels("")

        # Save the graph
    if save_dir:
        plt.savefig(f'{save_dir}/features.png', bbox_inches="tight")

    # Show the graph
    if fig_show:
        plt.show()
    else:
        plt.close()

    return ax, features


def set_visual_styles():
    # Matplotlib settings
    plt.style.use("classic")
    # Seaborn settings
    sns.set()
    sns.set(font_scale=1.0)
    sns.set_style({"font.family": "serif", "font.serif": ["Times New Roman"]})
    # Show seaborn settings
    sns.axes_style()


def df_into_to_sting(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def plot_logloss_and_error(model, name, save_dir=ML_DIR):
    set_visual_styles()

    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    # plot log loss
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Log Loss')
    pyplot.title(f'{name} - Log Loss')
    pyplot.savefig(f'{save_dir}/logloss.png', dpi=300)
    pyplot.show()

    # plot classification error
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Classification Error')
    pyplot.title(f'{name} - Classification Error')
    pyplot.savefig(f'{save_dir}/error.png', dpi=300)
    pyplot.show()

    if 'auc' in results['validation_0']:
        # plot auc
        fig, ax = pyplot.subplots()
        ax.plot(x_axis, results['validation_0']['auc'], label='Train')
        ax.plot(x_axis, results['validation_1']['auc'], label='Test')
        ax.legend()
        pyplot.xlabel('Iteration')
        pyplot.ylabel('Area under Curve')
        pyplot.title(f'{name} - Area under Curve')
        pyplot.savefig(f'{save_dir}/auc.png', dpi=300)
        pyplot.show()


def plot_cross_validation(results, name, save_dir=ML_DIR):
    set_visual_styles()

    # plot all metrics
    epochs = len(results['fit_time'])
    x_axis = range(0, epochs)
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['test_acc'], label='Accuracy')
    ax.plot(x_axis, results['test_sen'], label='Sensitivity')
    ax.plot(x_axis, results['test_spe'], label='Specificity')
    ax.plot(x_axis, results['test_f1'], label='F1 score')
    ax.plot(x_axis, results['test_mcc'], label='Matthews correlation coefficient')
    ax.legend(loc='best')
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Evaluation metrics')
    pyplot.title(f'{name} - Cross Validation Results')
    pyplot.savefig(f'{save_dir}/cross_validation_results.png', dpi=300)
    pyplot.show()
