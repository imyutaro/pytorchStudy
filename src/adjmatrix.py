import os
import gc
import time
import tqdm
import numpy as np

def make_adj(num_items, itemset, transactions):

    adjMatrix = np.zeros((num_items, num_items), dtype=np.float32)

    for t in tqdm.tqdm(transactions):
        for i, item1 in enumerate(t):
            # sub_t = t[i:] # こっちは対角成分が自分の出現回数になる
            sub_t = t[i+1:] # こっちはただの隣接行列
            idx1 = itemset.index(item1)
            for item2 in sub_t:
                idx2 = itemset.index(item2)
                adjMatrix[idx1][idx2] += 1

    return adjMatrix.astype(np.int32)

if __name__=="__main__":
    from dataloader import assoc 

    s_time = time.time()
    # filename = "T10I4D100K"
    filename = "retail"
    print(f"Data is {filename}")

    data = assoc(filename)
    transactions = data.transactions

    itemset = sorted(data.item, key=int)
    num_items = data.item_len()

    del data
    gc.collect()

    adjMatrix = make_adj(num_items, itemset, transactions)
    print(adjMatrix)

    save_dir = "../adj/"
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir+filename}", adjMatrix)
    f_time = time.time()

    print(f"passed time {f_time-s_time}")
