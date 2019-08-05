from dataloader import assoc
import time
import gc

def test1(t):
    t_len = len(t)
    a = []
    start = time.time()
    for i, item1 in enumerate(t):
        for j in range(i+1, t_len):
            a.append(item1+t[j])
    finish = time.time()
    print(f"total time {finish-start} seconds")
    return a

# test2 is much faster than test1!!!
def test2(t):
    from tqdm import trange, tqdm

    b = []
    start = time.time()
    for i, item1 in enumerate(t):
        t_sub = t[i+1:]
        for item2 in t_sub:
            b.append(item1+item2)
    finish = time.time()
    print(f"total time {finish-start} seconds")
    return b


if __name__=="__main__":

    # load dataset
    data = assoc("retail")

    count = 0
    # counting loop
    start = time.time()
    for _ in range(3):
        trans = data.get_trans()
        for t in trans:
            t_len = len(t)
            for i in range(t_len):
                for j in range(i+1, t_len):
                    count += 1 
    finish = time.time()

    print("Finish counting")
    print(f"    1 loop {count//3}")
    print(f"total loop {count}")
    print(f"total time {finish-start} seconds\n")
    del data 
    del trans
    del t_len
    del count
    gc.collect()

    # trans data is so big that I cannot exec test1 and 2.
    t = [i for i in range(7000)]

    a = test1(t)
    b = test2(t)

    if a==b:
        print("true")

