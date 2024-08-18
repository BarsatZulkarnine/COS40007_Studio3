import pandas as pd
from sklearn.utils import shuffle

#loAD
csv_files = ['./ampc/ampc/w1.csv', './ampc/ampc/w2.csv', './ampc/ampc/w3.csv', './ampc/ampc/w4.csv']  # Replace with actual file names

#combine
combined_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

combined_df.to_csv('combined_data.csv', index=False)

# Shuffle 
shuffled_df = shuffle(combined_df, random_state=1)
shuffled_df.to_csv('all_data.csv', index=False)
