import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# EXPOSURE.TSV

df_exposure = pd.read_csv("original_dataset/clinical/exposure.tsv", sep="\t",
                          usecols=['project.project_id','cases.case_id','exposures.pack_years_smoked',
                                   'exposures.tobacco_smoking_onset_year','exposures.tobacco_smoking_quit_year',
                                   'exposures.tobacco_smoking_status'])
df_exposure.dropna(inplace=True) # remove rows with wrong formatting

# consider only columns with >50% of data not null
cols = []
for i in df_exposure.columns:
    if len(df_exposure[df_exposure[i] == "\'--"]) <= df_exposure.shape[0]/2:
        cols.append(i)
print("Columns kept from exposure.tsv: " + str(cols))

features_df = pd.DataFrame(data=df_exposure, columns = cols)

# add cols
new_col_smoker = pd.Series(pd.NA, index=range(len(features_df.index))) # Null, True or False
new_col_years = pd.Series(pd.NA, index=range(len(features_df.index))) # Null, count_years (int)

# change obj type to int
columns_smoking_years = ["exposures.tobacco_smoking_onset_year", "exposures.tobacco_smoking_quit_year"]
for i in columns_smoking_years:
    features_df[i] = features_df[i].astype(str)
    features_df.loc[features_df[i] == '\'--', i] = '0'
    features_df[i] = features_df[i].astype(int)

# if both years not null --> True, data_stop - data_start
new_col_smoker.loc[(features_df['exposures.tobacco_smoking_quit_year'] != 0) & (features_df['exposures.tobacco_smoking_onset_year'] != 0)] = True
new_col_years.loc[(features_df['exposures.tobacco_smoking_quit_year'] != 0) & (features_df['exposures.tobacco_smoking_onset_year'] != 0)] = features_df['exposures.tobacco_smoking_quit_year'] - features_df['exposures.tobacco_smoking_onset_year']

# if only one year missing --> True, Null
new_col_smoker.loc[
    ((features_df['exposures.tobacco_smoking_quit_year'] != 0) & (features_df['exposures.tobacco_smoking_onset_year'] == 0))
    | ((features_df['exposures.tobacco_smoking_quit_year'] == 0) & (features_df['exposures.tobacco_smoking_onset_year'] != 0)
        )] = True

# if both years missing --> check status: if 'Lifelong Non-Smoker' --> False, else Null
new_col_smoker.loc[
    (features_df['exposures.tobacco_smoking_quit_year'] == 0) &
    (features_df['exposures.tobacco_smoking_onset_year'] == 0) &
    (features_df['exposures.tobacco_smoking_status'] == 'Lifelong Non-Smoker')] = False

# add new cols
features_df['exposures.tobacco_smoker'] = new_col_smoker # Null, True or False
features_df['exposures.tobacco_years'] = new_col_years # Null, count_years (int)
print("Columns added: ['exposures.tobacco_smoker', 'exposures.tobacco_years']")

# drop cols used and not more useful
features_df.drop(['exposures.tobacco_smoking_onset_year'], axis=1, inplace=True)
features_df.drop(['exposures.tobacco_smoking_quit_year'], axis=1, inplace=True)
features_df.drop(['exposures.tobacco_smoking_status'], axis=1, inplace=True)


# CLINICAL.TSV

df_clinical = pd.read_csv("original_dataset/clinical/clinical.tsv", sep="\t")
df_exposure.dropna(inplace=True) # remove rows with wrong formatting

# for each patient (case_id) keep only first row with classification_of_tumor == 'primary'
only_primary_df = df_clinical[df_clinical["diagnoses.classification_of_tumor"] == "primary"]
only_primary_df = only_primary_df.drop_duplicates(subset=["cases.case_id"])
only_primary_df.reset_index(inplace=True, drop=True)

cols = [
    'project.project_id',
    'cases.case_id',
    'demographic.age_at_index',
    'demographic.country_of_residence_at_enrollment',
    'demographic.ethnicity',
    'demographic.gender',
    'demographic.race',
    'diagnoses.ajcc_pathologic_m',
    'diagnoses.ajcc_pathologic_n',
    'diagnoses.ajcc_pathologic_stage',
    'diagnoses.ajcc_pathologic_t',
    'diagnoses.icd_10_code',
    'diagnoses.laterality',
    'diagnoses.sites_of_involvement',
    'diagnoses.tissue_or_organ_of_origin']


features_df = features_df.join(only_primary_df[cols].set_index(['project.project_id', 'cases.case_id']),
                               on=['project.project_id', 'cases.case_id'])
print("Columns joined from clinical.tsv: " + str(cols))

features_df.replace(['\'--', 'not reported', 'Unknown', 'MX', 'NX', 'TX'], pd.NA, inplace=True)

features_df.to_csv(r"files/clinical/features_considered.tsv", sep="\t", index=False)

for col in features_df.drop(columns=['project.project_id', 'cases.case_id']).columns:
    print("LUSC:\n" + str(features_df[features_df['project.project_id'] == 'TCGA-LUSC'][col].value_counts()))
    print("LUAD:\n" + str(features_df[features_df['project.project_id'] == 'TCGA-LUAD'][col].value_counts()))
    print()

print("\nEncoding features to numerical...")

#print(features_df.nunique())

# for project.project_id, remap tumor class to 0-1
mapping = {
    'TCGA-LUAD': 0,
    'TCGA-LUSC': 1
}
features_df['project.project_id'] = features_df['project.project_id'].map(mapping)

# for exposures.tobacco_smoker 0 will be missing value
mapping = {
    True: 1,
    False: -1
}
features_df['exposures.tobacco_smoker'] = features_df['exposures.tobacco_smoker'].map(mapping).fillna(0)

# features already numerical, set NA to 0:
cols = ['exposures.pack_years_smoked', 'exposures.tobacco_years', 'demographic.age_at_index']
for col in cols:
    median_val = pd.to_numeric(features_df[col], errors='coerce').median()  # if set to 0 it may be considered outlier
    features_df[col] = features_df[col].fillna(median_val)

# for diagnoses.ajcc_pathologic_stage, ordinal:
stage_map = {'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1, 'Stage II': 2, 'Stage IIA': 2,
             'Stage IIB': 2, 'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IV': 4}
features_df['diagnoses.ajcc_pathologic_stage'] = features_df['diagnoses.ajcc_pathologic_stage'].map(stage_map).fillna(0)

# for categorical features:
cols = [
    'demographic.country_of_residence_at_enrollment',
    'demographic.ethnicity',
    'demographic.gender',
    'demographic.race',
    'diagnoses.ajcc_pathologic_m',
    'diagnoses.ajcc_pathologic_n',
    'diagnoses.ajcc_pathologic_t',
    'diagnoses.icd_10_code',
    'diagnoses.laterality',
    'diagnoses.sites_of_involvement',
    'diagnoses.tissue_or_organ_of_origin'
]

categories = [features_df[col].dropna().unique() for col in cols]  # all possible not NA vals of each category

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(categories=categories, handle_unknown='ignore'), cols)],
    remainder='passthrough'  # Keep other columns as-is
)
encoded_array = ct.fit_transform(features_df)
encoded_df = pd.DataFrame(encoded_array, columns=ct.get_feature_names_out())

# to change encoded names from 'encoder__demographic.country_of_residence_at_enrollment_Australia' to 'country_of_residence_at_enrollment_Australia'
encoded_df = encoded_df.rename(columns=dict(zip(encoded_df.columns, [s.split('.')[-1] for s in encoded_df.columns])))

# reorder names istg
new_names = ['project_id', 'case_id', 'age_at_index', 'tobacco_years', 'pack_years_smoked']
new_names.extend(encoded_df.drop(columns=['project_id', 'case_id', 'age_at_index', 'tobacco_years', 'pack_years_smoked']).columns.values)

'''
Print encoded_df.columns to be sure names are cleaned. If prefixes remains:
encoded_df.columns = [c.replace('encoder__', '').replace('remainder__', '').split('.')[-1] for c in encoded_df.columns]
'''

encoded_df = encoded_df[new_names]

# CREATE FILE
encoded_df.to_csv(r"files/clinical/features_encoded.tsv", sep="\t", index=False)

print("DONE")

'''
DATA SUBDIVISION PER CLASS:

LUSC:
exposures.pack_years_smoked
40.0     41
50.0     36
30.0     25
60.0     24
25.0     19
         ..
114.0     1
3.0       1
27.0      1
41.0      1
25.6      1

LUAD:
exposures.pack_years_smoked
50.0    24
40.0    20
20.0    20
30.0    19
25.0    11
        ..
53.0     1
58.0     1
42.0     1
11.0     1
55.5     1



LUSC:
exposures.tobacco_smoker
True     331
False     13

LUAD:
exposures.tobacco_smoker
True     225
False     48



LUSC:
exposures.tobacco_years
40    23
50    21
35    11
30    10
20    10
45     9
25     7
60     6
41     6
56     5
43     5
54     4
53     4
48     4
51     4
57     4
37     3
52     3
33     3
44     3
28     3
55     2
42     2
26     2
36     2
39     2
38     2
46     2
15     2
24     2
23     2
10     2
47     2
59     1
17     1
61     1
14     1
11     1
49     1
9      1
29     1
31     1
8      1

LUAD:
exposures.tobacco_years
40    16
30    13
20    10
50     7
35     7
15     6
37     5
25     5
45     4
41     4
10     4
24     3
12     3
42     3
28     3
29     2
33     2
47     2
36     2
13     2
19     2
34     2
22     1
14     1
17     1
64     1
39     1
31     1
52     1
9      1
2      1
38     1
3      1
56     1
44     1
60     1
48     1



LUSC:
demographic.age_at_index
73    28
70    23
71    19
68    19
65    17
64    17
74    17
67    16
69    16
66    15
72    14
60    14
75    13
57    13
63    12
76    12
62    11
59    10
58    10
77    10
61     9
78     9
56     8
55     6
80     6
52     6
83     5
84     5
81     5
53     4
79     4
47     4
54     3
49     2
48     2
45     2
46     2
50     1
39     1
51     1
44     1
85     1
40     1

LUAD:
demographic.age_at_index
70    20
59    19
61    17
60    14
71    12
65    12
72    12
75    11
58    11
73    10
69    10
52    10
76    10
67     9
68     9
74     8
54     8
62     8
77     8
56     8
63     8
64     7
79     7
66     7
51     7
57     7
53     6
55     6
49     4
78     4
50     4
81     3
45     3
42     3
84     2
46     2
40     2
47     2
48     2
85     2
80     2
41     2
82     2
44     1
83     1
38     1
33     1
43     1



LUSC:
demographic.country_of_residence_at_enrollment
United States    218
Germany           47
Russia            43
Australia         38
Canada            37
Switzerland        6
Ukraine            6
Romania            4
Vietnam            4

LUAD:
demographic.country_of_residence_at_enrollment
United States    238
Germany           39
Australia         27
Russia            16
Canada            13
Romania            2
Vietnam            1
Ukraine            1



LUSC:
demographic.ethnicity
not hispanic or latino    249
hispanic or latino          6

LUAD:
demographic.ethnicity
not hispanic or latino    246
hispanic or latino          4
Name: count, dtype: int64



LUSC:
demographic.gender
male      299
female    104
Name: count, dtype: int64

LUAD:
demographic.gender
female    181
male      157
Name: count, dtype: int64



LUSC:
demographic.race
white                        271
black or african american     20
asian                          8

LUAD:
demographic.race
white                        248
black or african american     41
asian                          2
Name: count, dtype: int64



LUSC:
diagnoses.ajcc_pathologic_m
M0     333
M1       5
M1a      1

LUAD:
diagnoses.ajcc_pathologic_m
M0     232
M1      15
M1b      3
Name: count, dtype: int64



LUSC:
diagnoses.ajcc_pathologic_n
N0    253
N1    113
N2     28
N3      5

LUAD:
diagnoses.ajcc_pathologic_n
N0    211
N1     65
N2     52
N3      2



LUSC:
diagnoses.ajcc_pathologic_stage
Stage IB      122
Stage IIB      78
Stage IA       70
Stage IIIA     52
Stage IIA      51
Stage IIIB     13
Stage IV        6
Stage II        3
Stage III       3
Stage I         1

LUAD:
diagnoses.ajcc_pathologic_stage
Stage IB      91
Stage IA      80
Stage IIIA    49
Stage IIB     48
Stage IIA     29
Stage IV      19
Stage IIIB    10
Stage I        3
Stage II       1



LUSC:
diagnoses.ajcc_pathologic_t
T2     137
T2a     67
T3      60
T1      37
T1b     35
T2b     30
T1a     19
T4      18

LUAD:
diagnoses.ajcc_pathologic_t
T2     116
T1      48
T2a     47
T3      32
T1a     31
T1b     30
T2b     17
T4      15



LUSC:
diagnoses.icd_10_code
C34.1    203
C34.3    144
C34.9     34
C34.2     13
C34.8      8
C34.0      1

LUAD:
diagnoses.icd_10_code
C34.1    207
C34.3    110
C34.2     13
C34.9      4
C34.8      2
C34.0      2



LUSC:
diagnoses.laterality
Right    210
Left     175

LUAD:
diagnoses.laterality
Right    202
Left     127



LUSC:
diagnoses.sites_of_involvement
Central Lung       117
Peripheral Lung     79

LUAD:
diagnoses.sites_of_involvement
Peripheral Lung    88
Central Lung       41



LUSC:
diagnoses.tissue_or_organ_of_origin
Upper lobe, lung              203
Lower lobe, lung              144
Lung, NOS                      34
Middle lobe, lung              13
Overlapping lesion of lung      8
Main bronchus                   1

LUAD:
diagnoses.tissue_or_organ_of_origin
Upper lobe, lung              199
Lower lobe, lung              110
Middle lobe, lung              13
Lung, NOS                      12
Overlapping lesion of lung      2
Main bronchus                   2
'''