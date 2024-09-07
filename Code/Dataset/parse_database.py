import pandas as pd

features_path = 'features by category in biobank.xlsx'
headers = ['Feature Name', 'UKB Number', 'idk']
features_pickle_file = 'features_data.csv.pkl'

main_features_file = "biobank/fields672220.ukb"
second_features_file = "biobank/fields673316.ukb"

main_ukb_file = "biobank/ukb672220.csv"
second_ukb_file = "biobank/ukb673316.csv"
third_ukb_file = "biobank/ukb673540.csv"

class UKBDatasetCreator():
    df = None
    eids = []
    req_features = []
    fields = []
    second_fields = []
    third_fields = []
    features = []
    second_features = []
    third_features = []
    need_second_dataset = False
    need_third_dataset = False
    ukb_path = None
    num_rows = 0

    def __init__(self, features, num_rows=10000) -> None:
        self.ukb_path = main_ukb_file
        self.req_features = features
        self.num_rows = num_rows

    def sort_features(self):
        print("Sorting features by their dataset")
        with open(main_features_file, 'r') as mff: 
            main_feats = mff.read().split('\n')
        with open(second_features_file, 'r') as sff: 
            sec_feats = sff.read().split('\n')

        for feature in self.req_features:
            if feature in main_feats:
                self.features.append(feature)
            elif feature in sec_feats:
                self.second_features.append(feature)
            else:
                self.third_features.append(feature)

        if len(self.second_features) != 0:
            self.need_second_dataset = True
        if len(self.third_features) != 0:
            self.need_third_dataset = True

    def generate_fields(self) -> None:
        self.sort_features()
        self.fields = ["eid"] + [(str(feature) + '-0.0') for feature in self.features]
        if self.need_second_dataset:
            self.second_fields = ["eid"] + [(str(second_feature) + '-0.0') for second_feature in self.second_features]
        if self.need_third_dataset:
            self.third_fields = ["eid"] + [(str(third_feature) + '-0.0') for third_feature in self.third_features]
        print(f"Generated fields")

    def validate_fields(self):
        tmp_df = pd.read_csv(self.ukb_path, nrows=0)
        for i, field in enumerate(self.fields):
            if field not in tmp_df.columns:
                print(f"Field {field} was not found. Removing it from fields")
                self.fields.pop(i)

        if self.need_second_dataset:
            tmp_df = pd.read_csv(second_ukb_file, nrows=0)
            for i, field in enumerate(self.second_fields):
                if field not in tmp_df.columns:
                    print(f"Field {field} was not found. Removing it from second fields")
                    self.second_fields.pop(i)

        if self.need_third_dataset:
            tmp_df = pd.read_csv(third_ukb_file, nrows=0)
            for i, field in enumerate(self.third_fields):
                if field not in tmp_df.columns:
                    print(f"Field {field} was not found. Removing it from third fields")
                    self.third_fields.pop(i)
        print("Done validating the fields")

    def create_dataset(self):
        print("Creating dataset")
        self.generate_fields()
        assert self.fields != 1, "There are no fields to get"
        print("Validating all fields")
        self.validate_fields()
        print("Reading main csv")
        self.df = pd.read_csv(self.ukb_path, usecols=self.fields)
        self.eids = self.df["eid"].to_list()
        if self.need_second_dataset:
            print("Reading second csv")
            sec_df = pd.read_csv(second_ukb_file, usecols=self.second_fields)
            sec_df = sec_df[sec_df['eid'].isin(self.eids)]
            sec_df.to_csv("second.csv")
            print("Merging datasets by eid")
            self.df = pd.merge(self.df, sec_df, on='eid', how='inner')
        if self.need_third_dataset:
            print("Reading third csv")
            third_df = pd.read_csv(third_ukb_file, usecols=self.third_fields)
            third_df = third_df[third_df['eid'].isin(self.eids)]
            third_df.to_csv("third.csv")
            print("Merging datasets by eid")
            self.df = pd.merge(self.df, third_df, on='eid', how='inner')

    def save_dataset(self, db_path="dataset_all.csv"):
        print(f"Saving dataset to {db_path}")
        self.df.to_csv(db_path)

def main():
    features = [
        '1369',
        '1309',
        '1408',
        '1349',
        '1299',
        '130622',
        '130624',
        '130626',
        '130638',
        '130694',
        '130696',
        '130736',
        '130746',
        '131052',
        '131054',
        '131638',
        '131894',
        '132122',
        '132123',
        '132150',
        '132156',
        '132157',
        '132168',
        '132206',
        '132234',
        '132244',
        '132280',
        '20433',
        '20434',
        '20445',
        '20446',
        '20449',
        '20505',
        '20510',
        '20515',
        '20516',
        '20519',
        '20520',
        '2090',
        '2100',
        '21001',
        '21002',
        '21022',
        '21024',
        '21026',
        '21045',
        '21047',
        '21050',
        '21062',
        '21063',
        '21065',
        '22127',
        '23099',
        '2714',
        '2724',
        '2734',
        '2754',
        '2774',
        '2784',
        '30010',
        '30020',
        '30030',
        '30800',
        '31',
        '34',
        '3591',
        '3710',
        '3720',
        '3839',
        '3849',
        '40006',
        '41272',
        '6152',
        '120009',
        '120016',
        '120017',
        '120026',
        '120028',
        '120043',
        '120044',
        '120114',
        '6154',
        '132128',
        '132106',
        '132112',
        '132146',
        '130736',
        '131628',
        '131626',
        '3581',
        '3741',
        '131638',
        '131640',
        '131630',
        '21031',
        '21045',
        '40008',
        '2976',
        '2754',
        '2764',
        '2824',
        '2794',
        '2804',
        '21050',
        '132124',
        '132130',
        '132264',
        '131604',
        '132070',
        '131928',
        '132162',
    ]

    db_creator = UKBDatasetCreator(features)
    db_creator.create_dataset()
    db_creator.save_dataset()

if __name__ == "__main__":
    main()