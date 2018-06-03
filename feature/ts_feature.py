import data.robot.loader as loader
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

data_path = 'data\\robot\\lp1.csv'

if __name__ == '__main__':
    x, y = loader.load_robot_execution()
    extracted_features = extract_features(x, column_id="id", column_sort="time")
    impute(extracted_features)
    features_filtered = select_features(extracted_features, y)
    print(features_filtered.shape)
