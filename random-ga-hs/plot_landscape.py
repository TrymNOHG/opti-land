import pandas as pd
import random
import matplotlib.pylab as plt

df = pd.read_csv("lookup_tables/svm_feature.csv")
df['features'] = df['features'].astype(str).apply(lambda x: x.zfill(11))
feature_loss_dict = dict(zip(df['features'], df['loss']))

def flip_bit(s, idx):
    flipped = list(s)
    flipped[idx] = '1' if s[idx] == '0' else '0'
    return ''.join(flipped)

df['int_rep'] = df['features'].apply(lambda x: int(x, 2))

# plt.figure(figsize=(10, 6))
# plt.scatter(df['int_rep'], df['loss'], color='b')
# plt.xlabel('Integer Representation (int_rep)')
# plt.ylabel('Loss')
# plt.title('Integer Representation vs Loss (Scatter Plot)')
# plt.grid(True)
# plt.show()

df_sorted = df.sort_values(by='int_rep')


# def find_common_and_absent_features_in_odd_sets(start=501, end=1001, group_size=8, bit_length=11):
#     odd_numbers = [i for i in range(start, end) if i % 2 == 1]
#     grouped = [odd_numbers[i:i+group_size] for i in range(0, len(odd_numbers), group_size)]

#     for group_idx, group in enumerate(grouped):
#         if len(group) < group_size:
#             break  # skip incomplete group

#         binaries = [format(num, f'0{bit_length}b') for num in group]

#         common_features = []
#         absent_features = []

#         for i in range(bit_length):
#             bits_at_position = [b[i] for b in binaries]
#             if all(bit == '1' for bit in bits_at_position):
#                 # Convert index to feature number (rightmost is 1)
#                 common_features.append(bit_length - i)
#             elif all(bit == '0' for bit in bits_at_position):
#                 absent_features.append(bit_length - i)

#         print(f"Group {group_idx + 1} (numbers {group}):")
#         print(f"  Common Features (always 1): {sorted(common_features)}")
#         print(f"  Absent Features (always 0): {sorted(absent_features)}\n")

# find_common_and_absent_features_in_odd_sets()

# plt.figure(figsize=(10, 6))
# plt.plot(df_sorted['int_rep'][500:701], df_sorted['loss'][500:701], marker='o', linestyle='-', color='b')
# plt.xlabel('Integer Representation (int_rep)')
# plt.ylabel('Loss')
# plt.title('Integer Representation vs Loss')
# plt.grid(True)
# plt.show()

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Integer Representation vs Loss (Segmented View)', fontsize=16)

# Plot 1: 0 to 500
axs[0, 0].plot(df_sorted['int_rep'][:500], df_sorted['loss'][:500], marker='o', linestyle='-', color='b')
axs[0, 0].set_title('Index 0–499')
axs[0, 0].set_xlabel('int_rep')
axs[0, 0].set_ylabel('loss')
axs[0, 0].grid(True)

# Plot 2: 500 to 1000
axs[0, 1].plot(df_sorted['int_rep'][500:1000], df_sorted['loss'][500:1000], marker='o', linestyle='-', color='g')
axs[0, 1].set_title('Index 500–999')
axs[0, 1].set_xlabel('int_rep')
axs[0, 1].set_ylabel('loss')
axs[0, 1].grid(True)

# Plot 3: 1000 to 1500
axs[1, 0].plot(df_sorted['int_rep'][1000:1500], df_sorted['loss'][1000:1500], marker='o', linestyle='-', color='r')
axs[1, 0].set_title('Index 1000–1499')
axs[1, 0].set_xlabel('int_rep')
axs[1, 0].set_ylabel('loss')
axs[1, 0].grid(True)

# Plot 4: 1500 to end
axs[1, 1].plot(df_sorted['int_rep'][1500:], df_sorted['loss'][1500:], marker='o', linestyle='-', color='m')
axs[1, 1].set_title('Index 1500–end')
axs[1, 1].set_xlabel('int_rep')
axs[1, 1].set_ylabel('loss')
axs[1, 1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

