import pandas as pd
# from data import get_dataset
from experiment_utils import *
from utils import *
from processor import *
import defaults

def dataset_analysis(dataset_name, verbose=False):

    period_dict = {
        'ml-1m': "5months-9months", # was 4months-4months
        'amz-b': "3weeks-3weeks",
        'amz-g': "6weeks-6weeks",
        'steam': "1weeks-1weeks",
        'ml-10m': "1weeks-1weeks",
        'zvuk': "2weeks-2weeks"
        }
    
    period = period_dict[dataset_name]

    if verbose:
        print('dataset analysis (set verbose to False to skip)')
        # Read raw data for analysis
        data = pd.read_csv(f'./data/{dataset_name}.gz')
        userid, itemid, timeid = entity_names(dataset_name)
        
        # Apply same split as in read_dataset
        train_data, valid_data, test_data = split_offsets(data, period, timeid)
        
        # Convert timestamps for display
        timestamps = pd.to_datetime(data[timeid], unit='s')
        min_time = timestamps.min()
        max_time = timestamps.max()
        
        # Compute thresholds as in split_offsets
        time_offsets = read_offsets(period)
        test_time_threshold = max_time - time_offsets[-1]
        valid_time_threshold = test_time_threshold - time_offsets[-2]
        
        print(f"\n=== TEMPORAL RANGES ({dataset_name.upper()}) ===")
        print(f"Split period: {period}")
        print(f"TRAIN:  {min_time} -> {valid_time_threshold}")
        print(f"VALID:  {valid_time_threshold} -> {test_time_threshold}")  
        print(f"TEST:   {test_time_threshold} -> {max_time}")
                # Statistics on records
        train_count = len(train_data)
        valid_count = len(valid_data)
        test_count = len(test_data)
        total_count = len(data)
        
        print(f"\n=== NUMBER OF RECORDS ===")
        print(f"TRAIN:  {train_count:,} ({train_count/total_count*100:.1f}%)")
        print(f"VALID:  {valid_count:,} ({valid_count/total_count*100:.1f}%)")
        print(f"TEST:   {test_count:,} ({test_count/total_count*100:.1f}%)")
        print(f"TOTAL:  {total_count:,}")
        
        # User statistics
        train_users = set(train_data[userid].unique())
        valid_users = set(valid_data[userid].unique()) 
        test_users = set(test_data[userid].unique())
        all_users = set(data[userid].unique())
        
        print(f"\n=== NUMBER OF USERS ===")
        print(f"TRAIN:  {len(train_users):,}")
        print(f"VALID:  {len(valid_users):,}")
        print(f"TEST:   {len(test_users):,}")
        print(f"TOTAL:  {len(all_users):,}")
                # User intersections
        train_valid_overlap = len(train_users & valid_users)
        train_test_overlap = len(train_users & test_users)
        valid_test_overlap = len(valid_users & test_users)
        
        print(f"\n=== USER INTERSECTIONS ===")
        print(f"TRAIN ∩ VALID: {train_valid_overlap:,} ({train_valid_overlap/len(valid_users)*100:.1f}% of VALID)")
        print(f"TRAIN ∩ TEST:  {train_test_overlap:,} ({train_test_overlap/len(test_users)*100:.1f}% of TEST)")
        print(f"VALID ∩ TEST:  {valid_test_overlap:,} ({valid_test_overlap/len(test_users)*100:.1f}% of TEST)")
        
        # Users who were in TRAIN and TEST, but not in VALID (this is normal!)
        dormant_users = (train_users & test_users) - valid_users
        if len(dormant_users) > 0:
            print(f"\n📊 INACTIVE USERS: {len(dormant_users)} users were in TRAIN and TEST, but inactive in VALID")
            # print("(This is normal - users can be inactive in certain periods)")
        else:
            print("\n✅ All users have continuous activity")
        
        # Cold start analysis
        cold_start_valid = len(valid_users - train_users)
        cold_start_test = len(test_users - train_users)
        
        print(f"\n=== COLD START USERS ===")
        print(f"VALID (new): {cold_start_valid:,} ({cold_start_valid/len(valid_users)*100:.1f}% of VALID)")
        print(f"TEST (new):  {cold_start_test:,} ({cold_start_test/len(test_users)*100:.1f}% of TEST)")
        
        # Analysis of interaction sequence lengths (as they are REALLY used in SASRec)
        print(f"\n=== INTERACTION SEQUENCE LENGTHS (actual usage) ===")
        
        # Create sequences AS IN THE ACTUAL MODEL
        # 1. TRAINING sequences (train data -> grouping by users)
        train_user_sequences = (
            train_data
            .sort_values(timeid)
            .groupby(userid, sort=False)[itemid]
            .apply(list)
        )
        
        print("TRAINING SEQUENCES (used for model training):")
        if len(train_user_sequences) > 0:
            train_lengths = train_user_sequences.apply(len)
            
            mean_length = train_lengths.mean()
            median_length = train_lengths.median()
            min_length = train_lengths.min()
            max_length = train_lengths.max()
            std_length = train_lengths.std()
            q25 = train_lengths.quantile(0.25)
            q75 = train_lengths.quantile(0.75)
            
            print(f"  Mean:         {mean_length:.2f}")
            print(f"  Median:       {median_length:.2f}")
            print(f"  Std:          {std_length:.2f}") 
            print(f"  Min:          {min_length}")
            print(f"  Max:          {max_length}")
            print(f"  Q25:          {q25:.2f}")
            print(f"  Q75:          {q75:.2f}")
            print(f"  Users:        {len(train_lengths):,}")
            
            # Distribution by lengths
            users_with_1_interaction = (train_lengths == 1).sum()
            users_with_2_5_interactions = ((train_lengths >= 2) & (train_lengths <= 5)).sum()
            users_with_6_10_interactions = ((train_lengths >= 6) & (train_lengths <= 10)).sum()
            users_with_11_50_interactions = ((train_lengths >= 11) & (train_lengths <= 50)).sum()
            users_with_more_50_interactions = (train_lengths > 50).sum()
            
            total_users = len(train_lengths)
            print(f"  1 interaction:        {users_with_1_interaction:,} ({users_with_1_interaction/total_users*100:.1f}%)")
            print(f"  2-5 interactions:     {users_with_2_5_interactions:,} ({users_with_2_5_interactions/total_users*100:.1f}%)")
            print(f"  6-10 interactions:    {users_with_6_10_interactions:,} ({users_with_6_10_interactions/total_users*100:.1f}%)")
            print(f"  11-50 interactions:   {users_with_11_50_interactions:,} ({users_with_11_50_interactions/total_users*100:.1f}%)")
            print(f"  >50 interactions:     {users_with_more_50_interactions:,} ({users_with_more_50_interactions/total_users*100:.1f}%)")

            # --- NEW: TEST SEQUENCES ANALYSIS ----------------------------------------
            print("\nTEST sequences (train + valid + testset):")
            test_user_sequences = (
                pd.concat([training, testset_valid, testset_], ignore_index=True)
                .sort_values('timestamp')
                .groupby('userid', sort=False)['itemid']
                .apply(list)
            )
            if len(test_user_sequences) > 0:
                test_lengths = test_user_sequences.apply(len)
                mean_length_test = test_lengths.mean()
                median_length_test = test_lengths.median()
                min_length_test = test_lengths.min()
                max_length_test = test_lengths.max()
                std_length_test = test_lengths.std()
                q25_test = test_lengths.quantile(0.25)
                q75_test = test_lengths.quantile(0.75)

                print(f"  Users:            {len(test_lengths):,}")
                print(f"  Mean:             {mean_length_test:.2f}")
                print(f"  Median:           {median_length_test:.2f}")
                print(f"  Std:              {std_length_test:.2f}")
                print(f"  Min:              {min_length_test}")
                print(f"  Max:              {max_length_test}")
                print(f"  Q25:              {q25_test:.2f}")
                print(f"  Q75:              {q75_test:.2f}")

                # Distribution by lengths
                users_with_1_interaction_test = (test_lengths == 1).sum()
                users_with_2_5_interactions_test = ((test_lengths >= 2) & (test_lengths <= 5)).sum()
                users_with_6_10_interactions_test = ((test_lengths >= 6) & (test_lengths <= 10)).sum()
                users_with_11_50_interactions_test = ((test_lengths >= 11) & (test_lengths <= 50)).sum()
                users_with_more_50_interactions_test = (test_lengths > 50).sum()

                total_users_test = len(test_lengths)
                print(f"  1 interaction:        {users_with_1_interaction_test:,} ({users_with_1_interaction_test/total_users_test*100:.1f}%)")
                print(f"  2-5 interactions:     {users_with_2_5_interactions_test:,} ({users_with_2_5_interactions_test/total_users_test*100:.1f}%)")
                print(f"  6-10 interactions:    {users_with_6_10_interactions_test:,} ({users_with_6_10_interactions_test/total_users_test*100:.1f}%)")
                print(f"  11-50 interactions:   {users_with_11_50_interactions_test:,} ({users_with_11_50_interactions_test/total_users_test*100:.1f}%)")
                print(f"  >50 interactions:     {users_with_more_50_interactions_test:,} ({users_with_more_50_interactions_test/total_users_test*100:.1f}%)")
            else:
                print("  No sequences found.")
            # -------------------------------------------------------------------------
        print()
        
        # 2. FULL sequences (train+valid for final test)
        train_full_data = pd.concat([train_data, valid_data], ignore_index=True)
        train_full_user_sequences = (
            train_full_data
            .sort_values(timeid)
            .groupby(userid, sort=False)[itemid]
            .apply(list)
        )
        
        print("FULL SEQUENCES (train+valid, used in final test):")
        if len(train_full_user_sequences) > 0:
            full_lengths = train_full_user_sequences.apply(len)
            
            mean_length = full_lengths.mean()
            median_length = full_lengths.median()
            min_length = full_lengths.min()
            max_length = full_lengths.max()
            std_length = full_lengths.std()
            q25 = full_lengths.quantile(0.25)
            q75 = full_lengths.quantile(0.75)
            
            print(f"  Mean:         {mean_length:.2f}")
            print(f"  Median:       {median_length:.2f}")
            print(f"  Std:          {std_length:.2f}")
            print(f"  Min:          {min_length}")
            print(f"  Max:          {max_length}")
            print(f"  Q25:          {q25:.2f}")
            print(f"  Q75:          {q75:.2f}")
            print(f"  Users:        {len(full_lengths):,}")
            
            # Distribution by lengths
            users_with_1_interaction = (full_lengths == 1).sum()
            users_with_2_5_interactions = ((full_lengths >= 2) & (full_lengths <= 5)).sum()
            users_with_6_10_interactions = ((full_lengths >= 6) & (full_lengths <= 10)).sum()
            users_with_11_50_interactions = ((full_lengths >= 11) & (full_lengths <= 50)).sum()
            users_with_more_50_interactions = (full_lengths > 50).sum()
            
            total_users = len(full_lengths)
            print(f"  1 interaction:        {users_with_1_interaction:,} ({users_with_1_interaction/total_users*100:.1f}%)")
            print(f"  2-5 interactions:     {users_with_2_5_interactions:,} ({users_with_2_5_interactions/total_users*100:.1f}%)")
            print(f"  6-10 interactions:    {users_with_6_10_interactions:,} ({users_with_6_10_interactions/total_users*100:.1f}%)")
            print(f"  11-50 interactions:   {users_with_11_50_interactions:,} ({users_with_11_50_interactions/total_users*100:.1f}%)")
            print(f"  >50 interactions:     {users_with_more_50_interactions:,} ({users_with_more_50_interactions/total_users*100:.1f}%)")
        print()
        
        # 3. PREDICTION TASK STATISTICS
        print("PREDICTION TASKS:")
        print(f"VALID predictions:    {len(valid_data):,} (each interaction = separate task)")
        print(f"TEST predictions:     {len(test_data):,} (each interaction = separate task)")
        print(f"TOTAL tasks:          {len(valid_data) + len(test_data):,}")
        print()
        
        # 4. RAW DATA ANALYSIS (for reference)
        print("RAW DATA SUMMARY (for comparison):")
        all_user_sequences = data.groupby(userid).size()
        if len(all_user_sequences) > 0:
            mean_length = all_user_sequences.mean()
            median_length = all_user_sequences.median()
            min_length = all_user_sequences.min()
            max_length = all_user_sequences.max()
            
            print(f"  Mean interactions per user: {mean_length:.2f}")
            print(f"  Median:     {median_length:.2f}")
            print(f"  Min:        {min_length}")
            print(f"  Max:        {max_length}")
            print(f"  Users:      {len(all_user_sequences):,}")
            print(f"  Interactions: {len(data):,}")
        
        print("="*50)


def analyze_get_dataset(path, dataset_name, verbose=False, quantile=0.95):
    """
    Analyzes dataset initialized through get_dataset from data.py
    
    Args:
        path: path to CSV file of dataset
        dataset_name: dataset name for display
        verbose: if True, outputs detailed information
    
    Returns:
        dict: dictionary with analysis results
    """
    
    # Get dataset through get_dataset
    training, data_description, dict_popularities, testset_valid, testset_, holdout_valid, holdout_ = get_dataset(
        path=path, splitting='temporal', verbose=False, quantile=quantile
    )
    
    if verbose:
        print(f"=== DATASET ANALYSIS {dataset_name.upper()}===")
        print(f"Path: {path}")
        
        # Read raw data for temporal analysis
        original_data = pd.read_csv(path)
        
        print(f"\n=== GENERAL INFORMATION ===")
        print(f"Total number of users (n_users): {data_description['n_users']:,}")
        print(f"Total number of items (n_items): {data_description['n_items']:,}")
        print(f"Total records in dataframe: {len(original_data):,}")
        
        # Temporal ranges
        timestamps = pd.to_datetime(original_data['timestamp'], unit='s')
        min_time = timestamps.min()
        max_time = timestamps.max()
        
        # Compute thresholds for temporal split (95% quantile)
        test_timepoint = original_data['timestamp'].quantile(q=0.95, interpolation='nearest')
        test_time_threshold = pd.to_datetime(test_timepoint, unit='s')
        
        # Compute durations
        total_duration = max_time - min_time
        train_duration = test_time_threshold - min_time
        test_duration = max_time - test_time_threshold
        
        print(f"\n=== TEMPORAL RANGES ===")
        print(f"Full range: {min_time} -> {max_time}")
        print(f"  Duration: {total_duration.days} days ({total_duration.days/365.25:.1f} years)")
        print(f"TRAIN:  {min_time} -> {test_time_threshold}")
        print(f"  Duration: {train_duration.days} days ({train_duration.days/365.25:.1f} years)")
        print(f"  Share of total time: {train_duration/total_duration*100:.1f}%")
        print(f"TEST:   {test_time_threshold} -> {max_time}")
        print(f"  Duration: {test_duration.days} days ({test_duration.days/365.25:.1f} years)")
        print(f"  Share of total time: {test_duration/total_duration*100:.1f}%")
        
        # Detailed analysis of each split
        train_size = len(training)
        testset_valid_size = len(testset_valid)
        holdout_valid_size = len(holdout_valid)
        testset_final_size = len(testset_)
        holdout_final_size = len(holdout_)
        total_test_size = testset_valid_size + holdout_valid_size + testset_final_size + holdout_final_size
        total_size = len(original_data)
        
        print(f"\n=== DETAILED SPLIT ANALYSIS ===")
        print(f"TRAINING:           {train_size:,} records ({train_size/total_size*100:.1f}%)")
        print(f"TESTSET_VALID:      {testset_valid_size:,} records ({testset_valid_size/total_size*100:.1f}%)")
        print(f"HOLDOUT_VALID:      {holdout_valid_size:,} records ({holdout_valid_size/total_size*100:.1f}%)")
        print(f"TESTSET_ (final):   {testset_final_size:,} records ({testset_final_size/total_size*100:.1f}%)")
        print(f"HOLDOUT_ (final):   {holdout_final_size:,} records ({holdout_final_size/total_size*100:.1f}%)")
        print(f"TOTAL TEST data:    {total_test_size:,} records ({total_test_size/total_size*100:.1f}%)")
        print(f"TOTAL:              {total_size:,} records")
        
        # Detailed user analysis for each split
        train_users = set(training['userid'].unique())
        testset_valid_users = set(testset_valid['userid'].unique())
        holdout_valid_users = set(holdout_valid['userid'].unique())
        testset_final_users = set(testset_['userid'].unique())
        holdout_final_users = set(holdout_['userid'].unique())
        
        # Combine validation users
        all_valid_users = testset_valid_users | holdout_valid_users
        # Combine final test users  
        all_final_users = testset_final_users | holdout_final_users
        # Combine all test users
        all_test_users = all_valid_users | all_final_users
        
        print(f"\n=== DETAILED USER ANALYSIS ===")
        print(f"TRAINING users:         {len(train_users):,}")
        print(f"TESTSET_VALID users:   {len(testset_valid_users):,}")
        print(f"HOLDOUT_VALID users:   {len(holdout_valid_users):,}")
        print(f"TESTSET_ users:         {len(testset_final_users):,}")
        print(f"HOLDOUT_ users:         {len(holdout_final_users):,}")
        print(f"TOTAL VALID users:      {len(all_valid_users):,}")
        print(f"TOTAL FINAL users:      {len(all_final_users):,}")
        print(f"TOTAL TEST users:       {len(all_test_users):,}")
        
        # User intersection analysis
        train_valid_overlap = train_users & all_valid_users
        train_final_overlap = train_users & all_final_users
        train_test_overlap = train_users & all_test_users
        valid_final_overlap = all_valid_users & all_final_users
        
        print(f"\n=== USER INTERSECTIONS ===")
        print(f"TRAIN ∩ VALID:      {len(train_valid_overlap):,} ({len(train_valid_overlap)/len(all_valid_users)*100:.1f}% of VALID)")
        print(f"TRAIN ∩ FINAL:      {len(train_final_overlap):,} ({len(train_final_overlap)/len(all_final_users)*100:.1f}% of FINAL)")
        print(f"TRAIN ∩ ALL_TEST:   {len(train_test_overlap):,} ({len(train_test_overlap)/len(all_test_users)*100:.1f}% of ALL_TEST)")
        print(f"VALID ∩ FINAL:      {len(valid_final_overlap):,} ({len(valid_final_overlap)/len(all_final_users)*100:.1f}% of FINAL)")
        
        # Cold start analysis
        valid_cold_start = all_valid_users - train_users
        final_cold_start = all_final_users - train_users
        
        print(f"\n=== COLD START USERS ===")
        print(f"VALID (new):         {len(valid_cold_start):,} ({len(valid_cold_start)/len(all_valid_users)*100:.1f}% of VALID)")
        print(f"FINAL (new):         {len(final_cold_start):,} ({len(final_cold_start)/len(all_final_users)*100:.1f}% of FINAL)")
        
        # Inactive users (were in train and final, but not in valid)
        dormant_users = (train_users & all_final_users) - all_valid_users
        if len(dormant_users) > 0:
            print(f"\n📊 INACTIVE USERS: {len(dormant_users):,} users were in TRAIN and FINAL, but inactive in VALID")
        else:
            print(f"\n✅ All users have continuous activity")
            
        # Check consistency of holdout data
        testset_valid_users_check = set(testset_valid['userid'].unique())
        holdout_valid_users_check = set(holdout_valid['userid'].unique())
        testset_final_users_check = set(testset_['userid'].unique())
        holdout_final_users_check = set(holdout_['userid'].unique())
        
        print(f"\n=== HOLDOUT DATA CONSISTENCY ===")
        print(f"All TESTSET_VALID users are in HOLDOUT_VALID: {testset_valid_users_check.issubset(holdout_valid_users_check)}")
        print(f"All HOLDOUT_VALID users are in TESTSET_VALID: {holdout_valid_users_check.issubset(testset_valid_users_check)}")
        print(f"All TESTSET_ users are in HOLDOUT_: {testset_final_users_check.issubset(holdout_final_users_check)}")
        print(f"All HOLDOUT_ users are in TESTSET_: {holdout_final_users_check.issubset(testset_final_users_check)}")
        
        if testset_valid_users_check == holdout_valid_users_check:
            print("✅ TESTSET_VALID and HOLDOUT_VALID have the same users")
        else:
            print("⚠️  TESTSET_VALID and HOLDOUT_VALID have different users")
            
        if testset_final_users_check == holdout_final_users_check:
            print("✅ TESTSET_ and HOLDOUT_ have the same users")
        else:
            print("⚠️  TESTSET_ and HOLDOUT_ have different users")
        
        # Interaction sequence analysis
        train_user_sequences = (
            training
            .sort_values('timestamp')
            .groupby('userid', sort=False)['itemid']
            .apply(list)
        )
        
        print(f"\n=== INTERACTION SEQUENCES ===")
        print("TRAINING sequences:")
        
        if len(train_user_sequences) > 0:
            train_lengths = train_user_sequences.apply(len)
            
            mean_length = train_lengths.mean()
            median_length = train_lengths.median()
            min_length = train_lengths.min()
            max_length = train_lengths.max()
            std_length = train_lengths.std()
            q25 = train_lengths.quantile(0.25)
            q75 = train_lengths.quantile(0.75)
            
            print(f"  Users:            {len(train_lengths):,}")
            print(f"  Mean:             {mean_length:.2f}")
            print(f"  Median:            {median_length:.2f}")
            print(f"  Std:               {std_length:.2f}")
            print(f"  Min:               {min_length}")
            print(f"  Max:               {max_length}")
            print(f"  Q25:               {q25:.2f}")
            print(f"  Q75:               {q75:.2f}")
            
            # Distribution by lengths
            users_with_1_interaction = (train_lengths == 1).sum()
            users_with_2_5_interactions = ((train_lengths >= 2) & (train_lengths <= 5)).sum()
            users_with_6_10_interactions = ((train_lengths >= 6) & (train_lengths <= 10)).sum()
            users_with_11_50_interactions = ((train_lengths >= 11) & (train_lengths <= 50)).sum()
            users_with_more_50_interactions = (train_lengths > 50).sum()
            
            total_users = len(train_lengths)
            print(f"  1 interaction:        {users_with_1_interaction:,} ({users_with_1_interaction/total_users*100:.1f}%)")
            print(f"  2-5 interactions:     {users_with_2_5_interactions:,} ({users_with_2_5_interactions/total_users*100:.1f}%)")
            print(f"  6-10 interactions:    {users_with_6_10_interactions:,} ({users_with_6_10_interactions/total_users*100:.1f}%)")
            print(f"  11-50 interactions:   {users_with_11_50_interactions:,} ({users_with_11_50_interactions/total_users*100:.1f}%)")
            print(f"  >50 interactions:     {users_with_more_50_interactions:,} ({users_with_more_50_interactions/total_users*100:.1f}%)")
        
        # NEW: TESTSET_VALID sequences ------------------------------------
        testset_valid_user_sequences = (
            testset_valid
            .sort_values('timestamp')
            .groupby('userid', sort=False)['itemid']
            .apply(list)
        )
        print("\nTESTSET_VALID sequences:")
        if len(testset_valid_user_sequences) > 0:
            tv_lengths = testset_valid_user_sequences.apply(len)
            tv_mean_length = tv_lengths.mean()
            tv_median_length = tv_lengths.median()
            tv_min_length = tv_lengths.min()
            tv_max_length = tv_lengths.max()
            tv_std_length = tv_lengths.std()
            tv_q25 = tv_lengths.quantile(0.25)
            tv_q75 = tv_lengths.quantile(0.75)

            print(f"  Users:            {len(tv_lengths):,}")
            print(f"  Mean:             {tv_mean_length:.2f}")
            print(f"  Median:            {tv_median_length:.2f}")
            print(f"  Std:               {tv_std_length:.2f}")
            print(f"  Min:               {tv_min_length}")
            print(f"  Max:               {tv_max_length}")
            print(f"  Q25:               {tv_q25:.2f}")
            print(f"  Q75:               {tv_q75:.2f}")
        else:
            print("  No sequences found.")
        # ---------------------------------------------------------------------------

        # Test tasks - more detailed analysis
        print(f"\n=== TESTING STRUCTURE ===")
        print("VALIDATION STAGE:")
        print(f"  testset_valid:    {len(testset_valid):,} records (for training validation model)")
        print(f"  holdout_valid:    {len(holdout_valid):,} records (for evaluating validation model)")
        print("FINAL STAGE:")
        print(f"  testset_:         {len(testset_):,} records (for training final model)")
        print(f"  holdout_:         {len(holdout_):,} records (for final evaluation)")
        print(f"TOTAL test tasks: {total_test_size:,}")
        
        # Analysis of testset/holdout ratio
        if len(testset_valid) > 0 and len(holdout_valid) > 0:
            valid_ratio = len(holdout_valid) / len(testset_valid)
            print(f"\nVALIDATION holdout/testset ratio: {valid_ratio:.3f}")
        
        if len(testset_) > 0 and len(holdout_) > 0:
            final_ratio = len(holdout_) / len(testset_)
            print(f"FINAL holdout/testset ratio: {final_ratio:.3f}")
        
        # # Item popularity analysis
        # print(f"\n=== ITEM POPULARITY ===")
        # print(f"Most popular item: {dict_popularities.max():,} interactions")
        # print(f"Least popular item: {dict_popularities.min():,} interactions")
        # print(f"Mean interactions per item: {dict_popularities.mean():.2f}")
        # print(f"Median interactions per item: {np.median(dict_popularities):.2f}")
        
        print("=" * 60)
    
    # Return structured results
    train_user_sequences = (
        training
        .sort_values('timestamp')
        .groupby('userid', sort=False)['itemid']
        .apply(list)
    )
    train_lengths = train_user_sequences.apply(len) if len(train_user_sequences) > 0 else pd.Series([])
    
    sequence_stats = {}
    if len(train_lengths) > 0:
        sequence_stats = {
            'mean': train_lengths.mean(),
            'median': train_lengths.median(),
            'std': train_lengths.std(),
            'min': train_lengths.min(),
            'max': train_lengths.max(),
            'q25': train_lengths.quantile(0.25),
            'q75': train_lengths.quantile(0.75)
        }
    
    # NEW: test sequence stats
    test_sequence_stats = {}
    if 'test_user_sequences' in locals() and len(test_user_sequences) > 0:
        test_lengths_series = test_user_sequences.apply(len)
        test_sequence_stats = {
            'mean': test_lengths_series.mean(),
            'median': test_lengths_series.median(),
            'std': test_lengths_series.std(),
            'min': test_lengths_series.min(),
            'max': test_lengths_series.max(),
            'q25': test_lengths_series.quantile(0.25),
            'q75': test_lengths_series.quantile(0.75)
        }
    
    # Additional statistics for return
    train_users = set(training['userid'].unique())
    testset_valid_users = set(testset_valid['userid'].unique()) 
    holdout_valid_users = set(holdout_valid['userid'].unique())
    testset_final_users = set(testset_['userid'].unique())
    holdout_final_users = set(holdout_['userid'].unique())
    
    all_valid_users = testset_valid_users | holdout_valid_users
    all_final_users = testset_final_users | holdout_final_users
    all_test_users = all_valid_users | all_final_users
    
    user_stats = {
        'train_users_count': len(train_users),
        'testset_valid_users_count': len(testset_valid_users),
        'holdout_valid_users_count': len(holdout_valid_users),
        'testset_final_users_count': len(testset_final_users),
        'holdout_final_users_count': len(holdout_final_users),
        'all_valid_users_count': len(all_valid_users),
        'all_final_users_count': len(all_final_users),
        'all_test_users_count': len(all_test_users),
        'train_valid_overlap_count': len(train_users & all_valid_users),
        'train_final_overlap_count': len(train_users & all_final_users),
        'valid_final_overlap_count': len(all_valid_users & all_final_users),
        'valid_cold_start_count': len(all_valid_users - train_users),
        'final_cold_start_count': len(all_final_users - train_users),
        'dormant_users_count': len((train_users & all_final_users) - all_valid_users)
    }
    
    split_stats = {
        'training_size': len(training),
        'testset_valid_size': len(testset_valid),
        'holdout_valid_size': len(holdout_valid),
        'testset_final_size': len(testset_),
        'holdout_final_size': len(holdout_),
        'total_test_size': len(testset_valid) + len(holdout_valid) + len(testset_) + len(holdout_),
        'total_size': len(pd.read_csv(path))
    }
    
    # NEW: Average sequence length calculations (train & test) ------------------
    train_user_sequences_tmp = (
        training
        .sort_values('timestamp')
        .groupby('userid', sort=False)['itemid']
        .apply(list)
    )
    train_avg_seq_len = train_user_sequences_tmp.apply(len).mean() if len(train_user_sequences_tmp) > 0 else 0.0

    test_df_tmp = pd.concat([testset_valid, testset_, holdout_valid, holdout_], ignore_index=True)
    test_user_sequences_tmp = (
        test_df_tmp
        .sort_values('timestamp')
        .groupby('userid', sort=False)['itemid']
        .apply(list)
    )
    test_avg_seq_len = test_user_sequences_tmp.apply(len).mean() if len(test_user_sequences_tmp) > 0 else 0.0

    # NEW: Average sequence length for testset_valid ------------------------------------
    testset_valid_user_sequences_tmp = (
        testset_valid
        .sort_values('timestamp')
        .groupby('userid', sort=False)['itemid']
        .apply(list)
    )
    testset_valid_avg_seq_len = testset_valid_user_sequences_tmp.apply(len).mean() if len(testset_valid_user_sequences_tmp) > 0 else 0.0
    # ------------------------------------------------------------------------------------

    results = {
        'data_description': data_description,
        'split_stats': split_stats,
        'user_stats': user_stats,
        'popularities': dict_popularities,
        'training': training,
        'testset_valid': testset_valid,
        'testset_': testset_,
        'holdout_valid': holdout_valid,
        'holdout_': holdout_,
        'train_sequence_stats': sequence_stats,
        'test_sequence_stats': test_sequence_stats,
        'train_avg_seq_len': train_avg_seq_len,
        'test_avg_seq_len': test_avg_seq_len,
        'testset_valid_avg_seq_len': testset_valid_avg_seq_len
    }
    
    return results


def create_datasets_summary_table(dataset_names, quantile=0.95, verbose=False):
    import pandas as pd
    
    summary_data = []
    
    for dataset_name in dataset_names:
        print(f"Analyzing dataset: {dataset_name}...")
        
        try:
            # Path to data
            data_path = f"{defaults.data_dir}/{dataset_name}.csv"
            
            # Get dataset through get_dataset
            training, data_description, dict_popularities, testset_valid, testset_, holdout_valid, holdout_ = get_dataset(
                path=data_path, splitting='temporal', verbose=False, quantile=quantile
            )
            
            # Read raw data for temporal analysis
            original_data = pd.read_csv(data_path)
            
            # Main characteristics
            n_users = data_description['n_users']
            n_items = data_description['n_items']
            total_interactions = len(original_data)
            
            # Temporal analysis
            timestamps = pd.to_datetime(original_data['timestamp'], unit='s')
            min_time = timestamps.min()
            max_time = timestamps.max()
            total_time_days = (max_time - min_time).days
            
            # Compute temporal threshold for split
            test_timepoint = original_data['timestamp'].quantile(q=quantile, interpolation='nearest')
            test_time_threshold = pd.to_datetime(test_timepoint, unit='s')
            
            train_days = (test_time_threshold - min_time).days
            test_days = (max_time - test_time_threshold).days
            
            # Test user analysis
            testset_valid_users = set(testset_valid['userid'].unique())
            holdout_valid_users = set(holdout_valid['userid'].unique())
            testset_final_users = set(testset_['userid'].unique())
            holdout_final_users = set(holdout_['userid'].unique())
            
            all_valid_users = testset_valid_users | holdout_valid_users
            all_final_users = testset_final_users | holdout_final_users
            all_test_users = all_valid_users | all_final_users
            test_users = len(all_test_users)
            
            # Sequence length analysis (training)
            train_user_sequences = (
                training
                .sort_values('timestamp')
                .groupby('userid', sort=False)['itemid']
                .apply(list)
            )
            
            if len(train_user_sequences) > 0:
                train_lengths = train_user_sequences.apply(len)
                avg_sequence_length = train_lengths.mean()
            else:
                avg_sequence_length = 0
            
            # Train interactions (95% of total, which corresponds to training data)
            train_interactions_095 = len(training)
            
            # Holdout data (final test without validation)
            holdout_final_size = len(testset_valid)
            
            # Add data to summary
            summary_data.append({
                'Dataset': dataset_name,
                'n_users': f"{n_users:,}",
                'n_items': f"{n_items:,}",
                'total numbers of interactions': f"{total_interactions:,}",
                'total time days': total_time_days,
                'train days': train_days,
                'test days': test_days,
                'test users': f"{test_users:,}",
                'average sequence length': f"{avg_sequence_length:.2f}",
                'train interactions 0.95': f"{train_interactions_095:,}",
                'holdout (without test and valid)': f"{holdout_final_size:,}"
            })
            
            if verbose:
                print(f"  ✅ {dataset_name} processed successfully")
                print(f"     Users: {n_users:,}, Items: {n_items:,}")
                print(f"     Interactions: {total_interactions:,}")
                print(f"     Period: {total_time_days} days ({train_days} train + {test_days} test)")
                print(f"     Average sequence length: {avg_sequence_length:.2f}")
                print()
                
        except Exception as e:
            print(f"  ❌ Error processing {dataset_name}: {str(e)}")
            # Add empty row with error
            summary_data.append({
                'Dataset': dataset_name,
                'n_users': 'ERROR',
                'n_items': 'ERROR', 
                'total numbers of interactions': 'ERROR',
                'total time days': 'ERROR',
                'train days': 'ERROR',
                'test days': 'ERROR',
                'test users': 'ERROR',
                'average sequence length': 'ERROR',
                'train interactions 0.95': 'ERROR',
                'holdout (without test and valid)': 'ERROR'
            })
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Rename columns for nice display
    df.columns = [
        'Dataset',
        'n_users',
        'n_items', 
        'total numbers of interactions',
        'total time days',
        'train days',
        'test days',
        'test users',
        'average sequence length',
        'train interactions 0.95',
        'holdout (without test and valid)'
    ]
    
    return df


def print_datasets_summary(dataset_names, quantile=0.95, verbose=False):
    """
    Prints a nicely formatted summary table for datasets.
    
    Args:
        dataset_names: list of dataset names. If None, uses standard list
        quantile: quantile for temporal split
        verbose: detailed output
    """

    
    print("=" * 120)
    print("DATASET STATISTICS SUMMARY TABLE")
    print("=" * 120)
    
    # Create table
    df = create_datasets_summary_table(dataset_names, quantile=quantile, verbose=verbose)
    
    # Print table
    print()
    print(df.to_string(index=False))
    print()
    print("=" * 120)
    
    # Additional information
    print(f"Datasets analyzed: {len(dataset_names)}")
    print(f"Quantile for temporal splitting: {quantile}")
    
    return df


def analyze_prepare_split(dataset_name, time_offset_q=(0.95, 0.97), verbose=False, split_type='exclude'):
    """
    Analyzes split obtained by `prepare_data` function from `processor.py`.

    Supports split of the form:
        tune_datapack, test_datapack = prepare_data(dataset_name, time_offset_q=[0.95, 0.97])

    Where
        tune_datapack = (train_df, valid_df, data_index)
        test_datapack = (train_full_df, test_df, data_index)

    Parameters
    ----------
    dataset_name : str
        Dataset name (without extension), matching the csv filename
        in the `ScalableSASRec/preprocessed_data/` directory.
    time_offset_q : tuple(float, float), optional
        Quantile fractions used to split the temporal axis into
        (train+valid)/test and train/valid.  Default is ``(0.95, 0.97)``.
    verbose : bool, optional
        If ``True``, prints detailed statistics.  If ``False``,
        returns a dictionary with results without console output.

    Returns
    -----------
    dict
        Dictionary with key split statistics.
    """

    import pandas as pd
    import numpy as np
    from datetime import timedelta

    # 1. Get splits ----------------------------------------------------
    try:
        if split_type == 'exclude':
            tune_datapack, test_datapack = prepare_data_exclude(dataset_name, time_offset_q=time_offset_q)
        elif split_type == 'cut':
            tune_datapack, test_datapack = prepare_data_cut(dataset_name, time_offset_q=time_offset_q)
        else:
            raise ValueError(f"Unknown split type: {split_type}")

    except ValueError as e:
        raise RuntimeError(f"Failed to prepare data: {e}") from e

    # Unpack datapacks
    train_df, valid_df, _ = tune_datapack
    train_full_df, test_df, _ = test_datapack

    # 2. Global basic information --------------------------------------
    data_path = f"{defaults.data_dir}/{dataset_name}.csv"
    original_data = pd.read_csv(data_path)

    userid = 'userid'
    itemid = 'itemid'
    timeid = 'timestamp'

    n_users = original_data[userid].nunique()
    n_items = original_data[itemid].nunique()
    total_interactions = len(original_data)

    # 3. Time ---------------------------------------------------------------
    timestamps = pd.to_datetime(original_data[timeid], unit='s')
    min_time, max_time = timestamps.min(), timestamps.max()

    train_end_ts = train_df[timeid].max()
    valid_start_ts = valid_df[timeid].min()
    valid_end_ts = valid_df[timeid].max()
    test_start_ts = test_df[timeid].min()

    train_end_time = pd.to_datetime(train_end_ts, unit='s')
    valid_start_time = pd.to_datetime(valid_start_ts, unit='s')
    valid_end_time = pd.to_datetime(valid_end_ts, unit='s')
    test_start_time = pd.to_datetime(test_start_ts, unit='s')

    # Durations
    total_duration = max_time - min_time
    train_duration = train_end_time - min_time
    valid_duration = valid_end_time - valid_start_time
    test_duration = max_time - test_start_time

    # 4. Split sizes -----------------------------------------------------
    train_size = len(train_df)
    valid_size = len(valid_df)
    test_size = len(test_df)

    # 5. Users --------------------------------------------------------
    train_users = set(train_df[userid].unique())
    valid_users = set(valid_df[userid].unique())
    test_users = set(test_df[userid].unique())

    all_eval_users = valid_users | test_users

    train_valid_overlap = train_users & valid_users
    train_test_overlap = train_users & test_users
    valid_test_overlap = valid_users & test_users

    # Cold-start
    valid_cold_start = valid_users - train_users
    test_cold_start = test_users - train_users

    # 6. Sequence length statistics --------------------------------
    def _seq_stats(df):
        if df.empty:
            return {}
        seq = (
            df.sort_values(timeid)
              .groupby(userid, sort=False)[itemid]
              .apply(list)
        )
        # print(f"seq: {seq}")
        
        lengths = seq.apply(len)
        return {
            'mean': lengths.mean(),
            'median': lengths.median(),
            'std': lengths.std(),
            'min': lengths.min(),
            'max': lengths.max(),
            'q25': lengths.quantile(0.25),
            'q75': lengths.quantile(0.75),
        }

    train_seq_stats = _seq_stats(train_df)
    valid_seq_stats = _seq_stats(valid_df)
    test_seq_stats = _seq_stats(test_df)

    # 6b. Full sequences for TEST (train+valid+test) -------------
    combined_df = pd.concat([train_full_df, test_df], ignore_index=True)
    full_test_df = combined_df[combined_df[userid].isin(test_users)]
    full_test_seq_stats = _seq_stats(full_test_df)

    # ------------------------- verbose output ------------------------------
    if verbose:
        print(f"=== SPLIT ANALYSIS (prepare_data) – {dataset_name.upper()} ===")
        print(f"File: {data_path}\n")
        print("=== GENERAL CHARACTERISTICS ===")
        print(f"Users: {n_users:,}")
        print(f"Items: {n_items:,}")
        print(f"Total records: {total_interactions:,}\n")

        print("=== TEMPORAL RANGES ===")
        print(f"Full range: {min_time} -> {max_time}  (∑ {total_duration.days} days)")
        print(f"TRAIN:  {min_time} -> {train_end_time}  ({train_duration.days} days, {train_duration/total_duration*100:.1f}% of time)")
        print(f"VALID:  {valid_start_time} -> {valid_end_time}  ({valid_duration.days} days)")
        print(f"TEST:   {test_start_time} -> {max_time}  ({test_duration.days} days, {test_duration/total_duration*100:.1f}% of time)\n")

        print("=== SPLIT VOLUMES ===")
        print(f"TRAIN:  {train_size:,} ({train_size/total_interactions*100:.1f}%)")
        print(f"VALID:  {valid_size:,} ({valid_size/total_interactions*100:.1f}%)")
        print(f"TEST:   {test_size:,} ({test_size/total_interactions*100:.1f}%)\n")

        print("=== USERS ===")
        print(f"TRAIN users: {len(train_users):,}")
        print(f"VALID users: {len(valid_users):,}")
        print(f"TEST  users: {len(test_users):,}")
        print(f"TRAIN ∩ VALID: {len(train_valid_overlap):,} ({len(train_valid_overlap)/len(valid_users)*100 if valid_users else 0:.1f}% of VALID)")
        print(f"TRAIN ∩ TEST:  {len(train_test_overlap):,} ({len(train_test_overlap)/len(test_users)*100 if test_users else 0:.1f}% of TEST)")
        if valid_users:
            print(f"VALID ∩ TEST:  {len(valid_test_overlap):,} ({len(valid_test_overlap)/len(test_users)*100 if test_users else 0:.1f}% of TEST)")
        print(f"VALID cold-start users: {len(valid_cold_start):,}")
        print(f"TEST  cold-start users: {len(test_cold_start):,}\n")

        def _print_seq_stats(name, stats):
            if not stats:
                print(f"{name}: no sequences.")
                return
            print(f"{name}:  mean={stats['mean']:.2f}, median={stats['median']:.2f}, min={stats['min']}, max={stats['max']}, std={stats['std']:.2f}")

        print("=== SEQUENCE LENGTHS ===")
        _print_seq_stats('TRAIN', train_seq_stats)
        _print_seq_stats('VALID', valid_seq_stats)
        _print_seq_stats('TEST ', test_seq_stats)
        print("\n=== FULL SEQUENCE LENGTHS FOR TEST ===")
        _print_seq_stats('TEST (full history)', full_test_seq_stats)
        print("="*60)

    # ------------------------- return dict -------------------------------
    results = {
        'n_users': n_users,
        'n_items': n_items,
        'total_interactions': total_interactions,
        'time': {
            'min': min_time,
            'max': max_time,
            'train_end': train_end_time,
            'valid_start': valid_start_time,
            'valid_end': valid_end_time,
            'test_start': test_start_time,
        },
        'split_sizes': {
            'train': train_size,
            'valid': valid_size,
            'test': test_size,
        },
        'user_stats': {
            'train_users': len(train_users),
            'valid_users': len(valid_users),
            'test_users': len(test_users),
            'train_valid_overlap': len(train_valid_overlap),
            'train_test_overlap': len(train_test_overlap),
            'valid_test_overlap': len(valid_test_overlap),
            'valid_cold_start': len(valid_cold_start),
            'test_cold_start': len(test_cold_start),
        },
        'sequence_stats': {
            'train': train_seq_stats,
            'valid': valid_seq_stats,
            'test': test_seq_stats,
            'test_full': full_test_seq_stats,
        },
        'dataframes': {
            'train': train_df,
            'valid': valid_df,
            'test': test_df,
        },
    }

    return results


def create_datasets_summary_table_prepare_split(dataset_names, time_offset_q=(0.95, 0.97), split_type='exclude', verbose=False):
    """
    Creates a summary table for datasets based on information from analyze_prepare_split().
    
    Args:
        dataset_names: list of dataset names
        time_offset_q: quantiles for temporal split (default (0.95, 0.97))
        split_type: split type ('exclude' or 'cut')
        verbose: detailed output
    
    Returns:
        pandas.DataFrame: table with statistics for datasets
    """
    import pandas as pd
    
    summary_data = []
    
    for dataset_name in dataset_names:
        if verbose:
            print(f"Analyzing dataset: {dataset_name}...")
        
        try:
            # Get statistics through analyze_prepare_split
            stats = analyze_prepare_split(dataset_name, time_offset_q=time_offset_q, verbose=False, split_type=split_type)
            
            # Extract data from statistics
            n_users = stats['n_users']
            n_items = stats['n_items']
            total_interactions = stats['total_interactions']
            
            # Temporal characteristics
            min_time = stats['time']['min']
            max_time = stats['time']['max']
            train_end_time = stats['time']['train_end']
            valid_start_time = stats['time']['valid_start']
            valid_end_time = stats['time']['valid_end']
            test_start_time = stats['time']['test_start']
            
            total_duration_days = (max_time - min_time).days
            train_duration_days = (train_end_time - min_time).days
            valid_duration_days = (valid_end_time - valid_start_time).days
            test_duration_days = (max_time - test_start_time).days
            
            # Split sizes
            train_size = stats['split_sizes']['train']
            valid_size = stats['split_sizes']['valid']
            test_size = stats['split_sizes']['test']
            
            # User statistics
            train_users = stats['user_stats']['train_users']
            valid_users = stats['user_stats']['valid_users']
            test_users = stats['user_stats']['test_users']
            train_valid_overlap = stats['user_stats']['train_valid_overlap']
            train_test_overlap = stats['user_stats']['train_test_overlap']
            valid_test_overlap = stats['user_stats']['valid_test_overlap']
            valid_cold_start = stats['user_stats']['valid_cold_start']
            test_cold_start = stats['user_stats']['test_cold_start']
            
            # Sequence statistics
            train_seq_stats = stats['sequence_stats']['train']
            valid_seq_stats = stats['sequence_stats']['valid']
            test_seq_stats = stats['sequence_stats']['test']
            test_full_seq_stats = stats['sequence_stats']['test_full']
            
            # Compute dataset density
            density = total_interactions / (n_users * n_items) if n_users > 0 and n_items > 0 else 0
            # -----------------------------------------------------------------
            # Share of rows with repeating timestamp within one user
            # -----------------------------------------------------------------

            import defaults
            # Load only necessary columns for speed
            data_path = f"{defaults.data_dir}/{dataset_name}.csv"
            _df_ts = pd.read_csv(data_path, usecols=['userid', 'timestamp'])
            # Count records where (userid, timestamp) appears >1 time
            dup_counts = _df_ts.groupby(['userid', 'timestamp']).size()
            # print(f"{dup_counts}")
            duplicated_rows = dup_counts[dup_counts > 1].sum()
            timestamp_dup_ratio = duplicated_rows / len(_df_ts)

            
            # -----------------------------------------------------------------
            # Are there repeating itemid in user sequence?
            # -----------------------------------------------------------------
            _df_item = pd.read_csv(data_path, usecols=['userid', 'itemid'])
            item_dup_exists = _df_item.duplicated(subset=['userid', 'itemid']).any()

            
            # Add data to summary
            summary_data.append({
                'Dataset': dataset_name,
                'n_users': f"{n_users:,}",
                'n_items': f"{n_items:,}",
                'total_interactions': f"{total_interactions:,}",
                'density': f"{density:.6f}",
                'timestamp_dup_ratio': f"{timestamp_dup_ratio:.4f}",
                'item_dup_exists': 'yes' if item_dup_exists else 'no',
                'total_duration_days': total_duration_days,
                'train_duration_days': train_duration_days,
                'valid_duration_days': valid_duration_days,
                'test_duration_days': test_duration_days,
                'train_size': f"{train_size:,}",
                'valid_size': f"{valid_size:,}",
                'test_size': f"{test_size:,}",
                'train_users': f"{train_users:,}",
                'valid_users': f"{valid_users:,}",
                'test_users': f"{test_users:,}",
                'train_valid_overlap': f"{train_valid_overlap:,}",
                'train_test_overlap': f"{train_test_overlap:,}",
                'valid_test_overlap': f"{valid_test_overlap:,}",
                'valid_cold_start': f"{valid_cold_start:,}",
                'test_cold_start': f"{test_cold_start:,}",
                'train_seq_mean': f"{train_seq_stats.get('mean', 0):.2f}",
                'train_seq_median': f"{train_seq_stats.get('median', 0):.2f}",
                'valid_seq_mean': f"{valid_seq_stats.get('mean', 0):.2f}",
                'valid_seq_median': f"{valid_seq_stats.get('median', 0):.2f}",
                'test_seq_mean': f"{test_seq_stats.get('mean', 0):.2f}",
                'test_seq_median': f"{test_seq_stats.get('median', 0):.2f}",
                'test_full_seq_mean': f"{test_full_seq_stats.get('mean', 0):.2f}",
                'test_full_seq_median': f"{test_full_seq_stats.get('median', 0):.2f}",
                'split_type': split_type,
                'time_offset_q': f"{time_offset_q[0]}-{time_offset_q[1]}"
            })
            
            if verbose:
                print(f"  ✅ {dataset_name} processed successfully")
                print(f"     Users: {n_users:,}, Items: {n_items:,}")
                print(f"     Interactions: {total_interactions:,}")
                print(f"     Period: {total_duration_days} days")
                print(f"     Split: {split_type}, quantiles: {time_offset_q}")
                print()
                
        except Exception as e:
            print(f"  ❌ Error processing {dataset_name}: {str(e)}")
            # Add empty row with error
            summary_data.append({
                'Dataset': dataset_name,
                'n_users': 'ERROR',
                'n_items': 'ERROR',
                'total_interactions': 'ERROR',
                'density': 'ERROR',
                'timestamp_dup_ratio': 'ERROR',
                'item_dup_exists': 'ERROR',
                'total_duration_days': 'ERROR',
                'train_duration_days': 'ERROR',
                'valid_duration_days': 'ERROR',
                'test_duration_days': 'ERROR',
                'train_size': 'ERROR',
                'valid_size': 'ERROR',
                'test_size': 'ERROR',
                'train_users': 'ERROR',
                'valid_users': 'ERROR',
                'test_users': 'ERROR',
                'train_valid_overlap': 'ERROR',
                'train_test_overlap': 'ERROR',
                'valid_test_overlap': 'ERROR',
                'valid_cold_start': 'ERROR',
                'test_cold_start': 'ERROR',
                'train_seq_mean': 'ERROR',
                'train_seq_median': 'ERROR',
                'valid_seq_mean': 'ERROR',
                'valid_seq_median': 'ERROR',
                'test_seq_mean': 'ERROR',
                'test_seq_median': 'ERROR',
                'test_full_seq_mean': 'ERROR',
                'test_full_seq_median': 'ERROR',
                'split_type': split_type,
                'time_offset_q': f"{time_offset_q[0]}-{time_offset_q[1]}"
            })
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    return df


def create_datasets_summary_table_short(dataset_names, time_offset_q=(0.95, 0.97), split_type='exclude', verbose=False):
    """
    Creates a brief summary table for datasets with main metrics.
    
    Args:
        dataset_names: list of dataset names
        time_offset_q: quantiles for temporal split (default (0.95, 0.97))
        split_type: split type ('exclude' or 'cut')
        verbose: detailed output
    
    Returns:
        pandas.DataFrame: table with columns: #Users, #Items, #Interactions, Avg. Seq. Len, Density, Duplicates
    """
    import pandas as pd
    import defaults
    
    summary_data = []
    
    for dataset_name in dataset_names:
        if verbose:
            print(f"Analyzing dataset: {dataset_name}...")
        
        try:
            # Get statistics through analyze_prepare_split
            stats = analyze_prepare_split(dataset_name, time_offset_q=time_offset_q, verbose=False, split_type=split_type)
            
            # Extract data from statistics
            n_users = stats['n_users']
            n_items = stats['n_items']
            total_interactions = stats['total_interactions']
            
            # Average sequence length (across entire test_full dataset)
            test_full_seq_stats = stats['sequence_stats']['test_full']
            avg_seq_len = test_full_seq_stats.get('mean', 0)
            
            # Compute dataset density
            density = total_interactions / (n_users * n_items) if n_users > 0 and n_items > 0 else 0
            
            # Share of rows with repeating timestamp within one user
            data_path = f"{defaults.data_dir}/{dataset_name}.csv"
            _df_ts = pd.read_csv(data_path, usecols=['userid', 'timestamp'])
            dup_counts = _df_ts.groupby(['userid', 'timestamp']).size()
            duplicated_rows = dup_counts[dup_counts > 1].sum()
            timestamp_dup_ratio = duplicated_rows / len(_df_ts)
            
            # Add data to summary
            summary_data.append({
                'Dataset': dataset_name,
                '#Users': f"{n_users:,}",
                '#Items': f"{n_items:,}",
                '#Interactions': f"{total_interactions:,}",
                'Avg. Seq. Len': f"{avg_seq_len:.2f}",
                'Density': f"{density:.6f}",
                'Duplicates': f"{timestamp_dup_ratio:.4f}",
            })
            
            if verbose:
                print(f"  ✅ {dataset_name} processed successfully")
                print()
                
        except Exception as e:
            print(f"  ❌ Error processing {dataset_name}: {str(e)}")
            summary_data.append({
                'Dataset': dataset_name,
                '#Users': 'ERROR',
                '#Items': 'ERROR',
                '#Interactions': 'ERROR',
                'Avg. Seq. Len': 'ERROR',
                'Density': 'ERROR',
                'Duplicates': 'ERROR',
            })
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    return df


def print_datasets_summary_prepare_split(dataset_names, time_offset_q=(0.95, 0.97), split_type='exclude', verbose=False):
    """
    Prints a nicely formatted summary table for datasets based on analyze_prepare_split().
    
    Args:
        dataset_names: list of dataset names
        time_offset_q: quantiles for temporal split
        split_type: split type ('exclude' or 'cut')
        verbose: detailed output
    """
    
    print("=" * 150)
    print("DATASET STATISTICS SUMMARY TABLE (analyze_prepare_split)")
    print("=" * 150)
    
    # Create table
    df = create_datasets_summary_table_prepare_split(dataset_names, time_offset_q=time_offset_q, split_type=split_type, verbose=verbose)
    
    # Print table
    print()
    print(df.to_string(index=False))
    print()
    print("=" * 150)
    
    # Additional information
    print(f"Datasets analyzed: {len(dataset_names)}")
    print(f"Split type: {split_type}")
    print(f"Quantiles for temporal splitting: {time_offset_q}")
    
    return df


def create_datasets_basic_stats(dataset_names, time_offset_q=(0.95, 0.97), split_type='exclude', use_train_sequences=True):
    """
    Creates DataFrame with summary statistics for datasets.

    Metrics:
      - n_users: number of unique users
      - n_items: number of unique items
      - n_interactions: number of interactions (rows)
      - avg_seq_len: average sequence length
          * if use_train_sequences=True — by train split (as in model)
          * else — by entire original dataset (rough estimate)
      - density: interaction density = n_interactions / (n_users * n_items)

    Args:
        dataset_names: list of dataset names (without extension; searched in defaults.data_dir)
        time_offset_q: quantiles for double temporal split, e.g. (0.95, 0.97)
        split_type: type of temporal axis split ('exclude' | 'cut')
        use_train_sequences: whether to use sequences from train split

    Returns:
        pandas.DataFrame
    """

    summary_rows = []

    for dataset_name in dataset_names:
        try:
            data_path = f"{defaults.data_dir}/{dataset_name}.csv"

            # Basic counts for original dataset
            original_data = pd.read_csv(data_path, usecols=['userid', 'itemid', 'timestamp'])
            n_users = original_data['userid'].nunique()
            n_items = original_data['itemid'].nunique()
            n_interactions = len(original_data)
            density = (n_interactions / (n_users * n_items)) if (n_users > 0 and n_items > 0) else 0.0

            # Average sequence length
            if use_train_sequences:
                # Use double temporal split through prepare_data_* (as in analyze_prepare_split)
                if split_type == 'exclude':
                    tune_datapack, _ = prepare_data_exclude(dataset_name, time_offset_q=time_offset_q)
                elif split_type == 'cut':
                    tune_datapack, _ = prepare_data_cut(dataset_name, time_offset_q=time_offset_q)
                else:
                    raise ValueError(f"Unknown split_type: {split_type}")

                train_df, _, _ = tune_datapack

                if not train_df.empty:
                    train_user_sequences = (
                        train_df
                        .sort_values('timestamp')
                        .groupby('userid', sort=False)['itemid']
                        .apply(list)
                    )
                    avg_seq_len = float(train_user_sequences.apply(len).mean()) if len(train_user_sequences) > 0 else 0.0
                else:
                    avg_seq_len = 0.0
            else:
                # Rough estimate across entire dataset
                user_counts = original_data.groupby('userid').size()
                avg_seq_len = float(user_counts.mean()) if len(user_counts) > 0 else 0.0

            summary_rows.append({
                'Dataset': dataset_name,
                'n_users': n_users,
                'n_items': n_items,
                'n_interactions': n_interactions,
                'avg_seq_len': round(avg_seq_len, 2),
                'density': density,
            })

        except Exception as e:
            summary_rows.append({
                'Dataset': dataset_name,
                'n_users': 'ERROR',
                'n_items': 'ERROR',
                'n_interactions': 'ERROR',
                'avg_seq_len': 'ERROR',
                'density': 'ERROR',
            })

    return pd.DataFrame(summary_rows)
