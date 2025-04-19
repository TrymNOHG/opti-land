# import pandas as pd
# import random
# import matplotlib.pylab as plt

# df = pd.read_csv("lookup_tables/svm_feature.csv")
# df['features'] = df['features'].astype(str).apply(lambda x: x.zfill(11))
# feature_loss_dict = dict(zip(df['features'], df['loss']))

# def flip_bit(s, idx):
#     flipped = list(s)
#     flipped[idx] = '1' if s[idx] == '0' else '0'
#     return ''.join(flipped)

# all_neighbors_losses = []  
# curr_loss = []
# original = ''.join(random.choice('01') for _ in range(11))
# curr_loss.append(feature_loss_dict.get(original,None))
# for _ in range(50):
#     neighbor_losses = []
#     mutated_list = []
#     for i in range(11):
#         mutated = flip_bit(original, i)
#         mutated_list.append(mutated)
#         loss = feature_loss_dict.get(mutated, None)  
#         neighbor_losses.append(loss)
    
#     all_neighbors_losses.append(neighbor_losses)
#     original = random.choice(mutated_list)
#     curr_loss.append(feature_loss_dict.get(original,None))


# plt.figure(figsize=(14, 6))

# x = []
# y = []
# colors = []

# red_x = []
# red_y = []

# for i in range(len(all_neighbors_losses)):
#     # Current loss (red dot)
#     curr_x = i * 2
#     curr_y = curr_loss[i]
#     x.append(curr_x)
#     y.append(curr_y)
#     colors.append('red')

#     red_x.append(curr_x)
#     red_y.append(curr_y)

#     # Neighbor losses (blue dots)
#     for loss in all_neighbors_losses[i]:
#         x.append(i * 2 + 1)
#         y.append(loss)
#         colors.append('blue')

# # Append last red dot
# last_x = len(all_neighbors_losses) * 2
# last_y = curr_loss[-1]
# x.append(last_x)
# y.append(last_y)
# colors.append('red')

# red_x.append(last_x)
# red_y.append(last_y)

# # Plot everything
# plt.scatter(x, y, c=colors, alpha=0.7)
# plt.plot(red_x, red_y, color='red', linestyle='-', linewidth=2, label='Current Loss Path')
# plt.xlabel("Mutation Step (alternating current and neighbors)")
# plt.ylabel("Loss")
# plt.title("Traversal of Mutation Operator Over Energy Landscape")
# plt.grid(True)
# plt.legend()
# plt.show()


import pandas as pd
import random
import matplotlib.pylab as plt

# Load and preprocess data
df = pd.read_csv("lookup_tables/svm_feature.csv")
df['features'] = df['features'].astype(str).apply(lambda x: x.zfill(11))
feature_loss_dict = dict(zip(df['features'], df['loss']))

def flip_bit(s, idx):
    flipped = list(s)
    flipped[idx] = '1' if s[idx] == '0' else '0'
    return ''.join(flipped)

def traverse_landscape(initial, steps=50):
    """Returns curr_loss list and all_neighbors_losses list for one run."""
    curr_loss = [feature_loss_dict.get(initial, None)]
    all_neighbors_losses = []
    original = initial
    
    for _ in range(steps):
        neighbor_losses = []
        mutated_list = []
        for i in range(11):
            mutated = flip_bit(original, i)
            mutated_list.append(mutated)
            neighbor_losses.append(feature_loss_dict.get(mutated, None))
        all_neighbors_losses.append(neighbor_losses)
        
        original = random.choice(mutated_list)
        curr_loss.append(feature_loss_dict.get(original, None))
    
    return curr_loss, all_neighbors_losses

# Number of independent runs
n_runs = 10
steps = 50

# Prepare subplots: 2 rows × 5 cols
fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
axes = axes.flatten()

for run_idx in range(n_runs):
    # generate a random 11‐bit start
    init_str = ''.join(random.choice('01') for _ in range(11))
    curr_loss, all_neighbors_losses = traverse_landscape(init_str, steps=steps)
    
    # build x, y and separate out red‐point coords
    x_all, y_all, colors = [], [], []
    red_x, red_y = [], []
    
    for i, neigh in enumerate(all_neighbors_losses):
        # current point
        cx = 2 * i
        cy = curr_loss[i]
        x_all.append(cx); y_all.append(cy); colors.append('red')
        red_x.append(cx); red_y.append(cy)
        
        # neighbour points
        for lv in neigh:
            x_all.append(cx + 1)
            y_all.append(lv)
            colors.append('blue')
    
    # last current point
    last_x = 2 * len(all_neighbors_losses)
    last_y = curr_loss[-1]
    x_all.append(last_x); y_all.append(last_y); colors.append('red')
    red_x.append(last_x); red_y.append(last_y)
    
    # plot on its own axis
    ax = axes[run_idx]
    ax.scatter(x_all, y_all, c=colors, alpha=0.6, s=20)
    ax.plot(red_x, red_y, '-r', lw=1.5)
    ax.set_title(f"Run {run_idx+1}")
    ax.grid(True)
    if run_idx % 5 == 0:
        ax.set_ylabel("Loss")
    if run_idx >= 5:
        ax.set_xlabel("Step (curr vs neighbors)")

plt.tight_layout()
plt.suptitle("Mutation‐Landscape Traversals for 10 Random Starts", y=1.02, fontsize=16)
plt.show()
