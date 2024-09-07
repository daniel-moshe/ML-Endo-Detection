import pandas as pd
import numpy as np
import warnings

from features_preprocess import preprocess_cat_features
from utils import *

warnings.filterwarnings('ignore')

class Cohort:
    def __init__(self):
        dataset_path = "Dataset/dataset_all.csv"
        self.cat_cols = ['had_pregnancy_complications', 'has_headache_syndromes', 'has_endocrine_disorder', 
            'has_anemia', 'has_gyno_conditions', 'has_gastro_conditions', 'had_cesarean_section', 
            'had_melanoma', 'had_cervical_cancer', 'had_breast_cancer', 'had_uterine_cancer', 'had_overian_cancer',
            'has_PCOS', 'has_excessive_menstruation', 'had_appendicitis', 'had_UTI', 'has_back_pain', 'has_menstrual_pain',
            'has_lupus', 'has_infertility', 'has_IBS', 'had_abotrion']
        self.X = None
        self.y = None
        try:
            self.df = pd.read_csv(dataset_path, index_col=0)
        except Exception as ee:
            logging.error(f"Cannot open CSV file - {ee}")

    def create_cohort(self):
        self.drop_male_patients()
        self.create_labels()
        self.sample_patients()
        self.preprocess_cat_features()
        self.add_new_cols()
        print('Finished creating cohort')

    def drop_male_patients(self):
        sex_col = feature_to_code('Sex')

        self.df = self.df[self.df[sex_col] != 1]
        logging.info(f"Dropping column {sex_col}")
        self.df.drop(columns=[sex_col], inplace=True)

    def create_labels(self):
        logging.info("Creating label column")
        endo_diag_col = '132123-0.0'
        endo_date_col = '132122-0.0'

        self.df['has_endo'] = self.df[endo_diag_col].isna().astype(int).apply(lambda x: 0 if x else 1)
        logging.info(f"Dropping columns {code_to_feature(endo_diag_col), code_to_feature(endo_date_col)}")
        self.df.drop(columns=[endo_diag_col, endo_date_col], inplace=True)

    def sample_patients(self):
        logging.info("Sampling patients")
        num_endo_patients  = self.df['has_endo'].value_counts()[1]
        y_has_endo = self.df[self.df['has_endo'] == 1]
        y_no_endo = self.df[self.df['has_endo'] == 0]

        y_no_endo = y_no_endo.sample(n=num_endo_patients)

        self.df = pd.concat([y_has_endo, y_no_endo])

    def preprocess_cat_features(self):
        logging.info("Preprocessing categorical features")
        preprocess_cat_features(self.df)

    def add_new_cols(self):
        self.add_estrogen_exposure_col()
        self.add_num_diagnoses_col()

    def add_estrogen_exposure_col(self):
        logging.info("Adding estrogen exposure column")
        menarche_age_code = feature_to_code("menarche")
        menopause_age_code = feature_to_code("Age at menopause")

        def calc_estrogen_exposure(row):
            answer_not_known = [-1, 3]
            if pd.isna(row[menarche_age_code]) or pd.isna(row[menopause_age_code]):
                return np.nan
            elif (row[menarche_age_code] in answer_not_known) or (row[menopause_age_code] in answer_not_known):
                return np.nan
            elif row[menopause_age_code] > row[menarche_age_code]:
                return row[menopause_age_code] - row[menarche_age_code]
            else:
                return np.nan  

        self.df['estrogen_exposure'] = self.df.apply(calc_estrogen_exposure, axis=1)

    def add_num_diagnoses_col(self):
        logging.info("Adding number of diagnoses column")
        diag_file = 'biobank/hesin_diag.txt'
        diag_df = pd.read_csv(diag_file, sep ='\t')

        code_counts = diag_df.groupby('eid')['diag_icd10'].count().reset_index(name='diag_count')
        self.df = self.df.merge(code_counts, on='eid', how='left')
        self.df['diag_count'] = self.df['diag_count'].fillna(0)
        self.df.drop(columns=['eid'], inplace=True)

    def drop_cols(self, cols):
        self.df.drop(columns=cols, inplace=True, errors='ignore')

    def split_x_y(self):
        y_column = "has_endo"
        self.X = self.df.drop(y_column, axis=1)
        self.y = self.df[y_column]

        self.X.columns = self.X.columns.to_series().apply(code_to_feature)