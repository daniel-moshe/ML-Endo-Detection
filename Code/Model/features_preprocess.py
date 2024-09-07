import pandas as pd
import numpy as np

from utils import feature_to_code
    
def encode_one_hot_by_nans(df, feature_list, new_feature_name):
    df[new_feature_name] = pd.DataFrame(df[feature_list]).any(axis=1).astype(int)
    df.drop(columns=feature_list, inplace=True)

def encode_one_hot_by_string_val(df, feature_code, str_match, new_feature_name):
    df[new_feature_name] = df[feature_code].str.startswith(str_match, na=False).astype(int)

def encode_single_feature(df, feature_code, new_feature_name):
    code = feature_to_code(feature_code)
    encode_one_hot_by_nans(df, code, new_feature_name)
    
def encode_pregnancy_comp_feature(df):
    pregnancy_complications_codes = [feature_to_code("O26"), feature_to_code("O44"), feature_to_code("O60"), feature_to_code("O00"), feature_to_code("O70")]
    encode_one_hot_by_nans(df, pregnancy_complications_codes, 'had_pregnancy_complications')

def encode_headache_feature(df):
    headache_syndromes_codes = [feature_to_code("G43"), feature_to_code("G44")]
    encode_one_hot_by_nans(df, headache_syndromes_codes, 'has_headache_syndromes')

def encode_endocrine_feature(df):
    hypothyroidism_code = feature_to_code("E02") # 130694
    general_hypothyroidism_code = feature_to_code("E03") # 130696
    general_endocrine_code = feature_to_code("E34") # 130746
    endocrine_codes = [hypothyroidism_code, general_hypothyroidism_code, general_endocrine_code]
    
    encode_one_hot_by_nans(df, endocrine_codes, 'has_endocrine_disorder')

def encode_anemia_feature(df):
    iron_def_anemia_code = feature_to_code("D50")
    b12_def_anemia_code = feature_to_code("D51")
    folate_def_anemia_code = feature_to_code("D52")
    aquired_anemia_code = feature_to_code("D59")

    anemia_codes = [iron_def_anemia_code, b12_def_anemia_code, folate_def_anemia_code, aquired_anemia_code]
    encode_one_hot_by_nans(df, anemia_codes, 'has_anemia')
    
def encode_genital_conditions_feature(df):
    genital_conditions_codes = [feature_to_code("N81"), feature_to_code("N84"), feature_to_code("N83"), feature_to_code("N70"), feature_to_code("N73")]
    encode_one_hot_by_nans(df, genital_conditions_codes, 'has_gyno_conditions')
    
def encode_gastro_conditions(df):
    gastro_conditions_codes = [feature_to_code("K52"), feature_to_code("K59"), feature_to_code("K50"), feature_to_code("K51")]
    encode_one_hot_by_nans(df, gastro_conditions_codes, 'has_gastro_conditions')

def encode_cesarean_feature(df):
    operation_code = feature_to_code("Operative procedures")
    cesarean_code = feature_to_code("O82") # 132280
    df['had_cesarean_section'] = np.where(df[operation_code].str.startswith(('R17', 'R18'), na=False) | df[cesarean_code].notna(), 1, 0)
    df.drop(columns=[operation_code, cesarean_code], inplace=True)

def encode_cancer_feature(df):
    cancer_code = feature_to_code("cancer")
    encode_one_hot_by_string_val(df, cancer_code, ('C56'), 'had_overian_cancer')
    encode_one_hot_by_string_val(df, cancer_code, ('C55'), 'had_uterine_cancer')
    encode_one_hot_by_string_val(df, cancer_code, ('C50'), 'had_breast_cancer')
    encode_one_hot_by_string_val(df, cancer_code, ('C53'), 'had_cervical_cancer')
    encode_one_hot_by_string_val(df, cancer_code, ('C44', 'C43'), 'had_melanoma')
    df.drop(columns=[cancer_code], inplace=True)

def preprocess_cat_features(df):
    encode_pregnancy_comp_feature(df)
    encode_headache_feature(df)
    encode_endocrine_feature(df)
    encode_anemia_feature(df)
    encode_cesarean_feature(df)
    encode_cancer_feature(df)
    encode_genital_conditions_feature(df)
    encode_gastro_conditions(df)
    encode_single_feature(df, "O03", 'had_abotrion')
    encode_single_feature(df, "K58", 'has_IBS')
    encode_single_feature(df, "N97", 'has_infertility')
    encode_single_feature(df, "M32", 'has_lupus')
    encode_single_feature(df, "N94", 'has_menstrual_pain')
    encode_single_feature(df, "M54", 'has_back_pain')
    encode_single_feature(df, "N39", 'had_UTI')
    encode_single_feature(df, "K35", 'had_appendicitis')
    encode_single_feature(df, "N92", 'has_excessive_menstruation')
    encode_single_feature(df, "E28", 'has_PCOS')