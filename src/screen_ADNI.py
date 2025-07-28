import os
import pandas as pd

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.abspath(os.path.join(script_dir, '..', 'data'))

    input_csv = os.path.join(data_dir, 'ADNI-2_patients.csv')
    out_full = os.path.join(data_dir, 'filtered_MPRAGE_records.csv')
    out_ids = os.path.join(data_dir, 'mprage_subjects.csv')
    out_visits_all = os.path.join(data_dir, 'mprage_visit_counts.csv')
    out_dist_sense2 = os.path.join(data_dir, 'dist_mprage_sense2.csv')
    out_dist_nosense2 = os.path.join(data_dir, 'dist_mprage_nosense2.csv')

    df = pd.read_csv(input_csv)

    #filter for mprage including SENSE2
    mask = df['Description'].str.contains('MPRAGE', case=False, na=False)
    mprage_df = df[mask].copy()
    mprage_df.to_csv(out_full, index=False)

    unique_ids = pd.DataFrame({'Subject ID': mprage_df['Subject ID'].unique()})
    unique_ids.to_csv(out_ids, index=False)

    # split into SENSE2 vs no‑SENSE2
    sense2_mask = mprage_df['Description'].str.contains('SENSE2', case=False, na=False)
    mprage_sense2 = mprage_df[sense2_mask]
    mprage_nosense2 = mprage_df[~sense2_mask]

    # distinct ages point toward unique visits
    visits_sense2 = (
        mprage_sense2
        .groupby('Subject ID')['Age']
        .nunique()
        .reset_index(name='visits'))
    visits_nosense2 = (
        mprage_nosense2
        .groupby('Subject ID')['Age']
        .nunique()
        .reset_index(name='visits'))

    # combine into table
    visits_sense2['Type'] = 'MPRAGE SENSE2'
    visits_nosense2['Type'] = 'MPRAGE only'
    all_visits = pd.concat([visits_sense2, visits_nosense2], ignore_index=True)
    all_visits.to_csv(out_visits_all, index=False)

    # compute distributions
    dist_sense2 = (
        visits_sense2['visits']
        .value_counts()
        .sort_index()
        .rename_axis('visits')
        .reset_index(name='num_subjects'))
    dist_nosense2 = (
        visits_nosense2['visits']
        .value_counts()
        .sort_index()
        .rename_axis('visits')
        .reset_index(name='num_subjects'))

    dist_sense2.to_csv(out_dist_sense2, index=False)
    dist_nosense2.to_csv(out_dist_nosense2, index=False)

if __name__ == '__main__':
    main()





"""
import os
import pandas as pd

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_csv        = os.path.join(script_dir, 'ADNI-3_patients.csv')
    out_filtered     = os.path.join(script_dir, 'filtered_mprage.csv')
    out_visits       = os.path.join(script_dir, 'mprage_visit_counts.csv')
    out_dist_visits  = os.path.join(script_dir, 'dist_mprage_visits.csv')
    out_dist_types   = os.path.join(script_dir, 'dist_mprage_types.csv')

    df = pd.read_csv(input_csv)
    mprage_df = df[df['Description']
                   .str.contains('mprage', case=False, na=False)].copy()
    mprage_df.to_csv(out_filtered, index=False)

    # 2) visits per subject is distinct Age values
    visits = (
        mprage_df
        .groupby('Subject ID')['Age']
        .nunique()
        .reset_index(name='visits'))
    visits.to_csv(out_visits, index=False)

    # distribution based on visits per subject
    dist_visits = (
        visits['visits']
        .value_counts()
        .sort_index()
        .rename_axis('visits')
        .reset_index(name='num_subjects'))
    dist_visits.to_csv(out_dist_visits, index=False)

    # distribution of each MPRAGE type ("Description")
    dist_types = (
        mprage_df
        .groupby('Description')['Subject ID']
        .nunique()
        .reset_index(name='num_subjects')
        .sort_values('num_subjects', ascending=False))
    dist_types.to_csv(out_dist_types, index=False)

if __name__ == '__main__':
    main()


"""