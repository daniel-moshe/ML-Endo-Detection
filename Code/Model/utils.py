import pandas as pd
import logging
import pickle
import sys

sys.path.insert('..')

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]:[%(module)-15s:%(lineno)3d]:[%(levelname)-8s] - %(message)s",
)

# When importing utils.py from py script we cannot use sns and plt
try:
    get_ipython()
    import seaborn as sns
    import matplotlib.pyplot as plt
except:
    logging.warning("Running in terminal, cannot import sns and plt")


FEATURE_NAME_COL = "Feature Name"
CODE_NUMBER_COL  = "UKB Number"

utils_features_data = None

def init():
    global utils_features_data
    features_pickle_file = "Dataset/features_data.csv.pkl"

    with open(features_pickle_file, 'rb') as f:
        utils_features_data = pickle.load(f)
    logging.debug("Data initiated")

def code_to_feature(code):
    try:
        if not code.split('-')[0].isnumeric():
            return code
        code = int(code.split('-')[0])
        result = utils_features_data.loc[utils_features_data[CODE_NUMBER_COL] == code][FEATURE_NAME_COL].values[0]
        return result
    except Exception as e:
        logging.error(f"Code number does not exist - {e}")
        raise e

def feature_to_code(feature):
    try:
        feat_values = utils_features_data.loc[utils_features_data[FEATURE_NAME_COL].str.contains(feature.lower(), case=False)]
        result = f"{feat_values[CODE_NUMBER_COL].values[0]}-0.0"
        return result
    except Exception as e:
        logging.error("Feature name does not exist")
        raise e

def change_feature_name(old_name, new_name, is_value_code=False):
    try:
        if is_value_code:
            old_name = code_to_feature(old_name)
        if old_name not in utils_features_data[FEATURE_NAME_COL].values:
            logging.debug(utils_features_data[FEATURE_NAME_COL].values)
            raise Exception(f"Feature {old_name} does not exist")
        utils_features_data[FEATURE_NAME_COL] = utils_features_data[FEATURE_NAME_COL].replace({old_name: new_name})
        logging.info(f"Renamed feature: {old_name} -> {new_name}")
    except Exception as e:
        logging.error(f"Got an error - {e}")

def print_features(with_code=False):
    for _, row in utils_features_data.iterrows():
        code = f"{row[CODE_NUMBER_COL]} - " if with_code else ''
        logging.info(f"{code}{row[FEATURE_NAME_COL]} ")

def left_align(df):
    left_aligned_df = df.style.set_properties(**{'text-align': 'left'})
    left_aligned_df = left_aligned_df.set_table_styles(
        [dict(selector='th', props=[('text-align', 'left')])]
    )
    return left_aligned_df

def get_null_precentages(df):
    return pd.DataFrame(df.isnull().mean().round(4).mul(100).sort_values(ascending=False).rename(code_to_feature))

def plot_one_hot_columns(df, title, xlabel, ylabel):
    def calculate_stats(df):
        total_rows = len(df)
        stats = []
        for column in df.columns:
            zeros = (df[column] == 0).sum()
            ones = (df[column] == 1).sum()
            stats.append({
                'column': column,
                'zeros': zeros,
                'ones': ones,
                'zeros_pct': zeros / total_rows * 100,
                'ones_pct': ones / total_rows * 100
            })
        return pd.DataFrame(stats)

    # When importing utils.py from py script we cannot use sns and plt
    try:
        get_ipython()
    except:
        return

    # Calculate stats
    stats_df = calculate_stats(df)

    # Create the stacked bar plot
    plt.figure(figsize=(15, 8))
    sns.set(style="whitegrid")

    # Plot the stacked bars
    ax = sns.barplot(x='column', y='zeros', data=stats_df, color='skyblue', label='0')
    sns.barplot(x='column', y='ones', data=stats_df, color='lightgreen', label='1', bottom=stats_df['zeros'])

    # Customize the plot
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title='Value')
    plt.xticks(rotation=90)

    # Add percentage labels
    for i, column in enumerate(stats_df['column']):
        zeros_count = stats_df.loc[i, 'zeros']
        ones_count = stats_df.loc[i, 'ones']
        zeros_pct = stats_df.loc[i, 'zeros_pct']
        ones_pct = stats_df.loc[i, 'ones_pct']

        plt.text(i, zeros_count/2, f'{zeros_pct:.1f}%', ha='center', va='center')
        plt.text(i, zeros_count + ones_count/2, f'{ones_pct:.1f}%', ha='center', va='center')

    plt.tight_layout()
    plt.show()

init()