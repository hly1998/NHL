import numpy as np
from scipy.special import comb
import random
from scipy.linalg import hadamard, eig
import copy
import time
import copy
from numba import jit
import time


def get_margin(bit, n_class):

    L = bit
    right = (2 ** L) / n_class

    d_min = 0
    d_max = 0
    for j in range(2 * L + 4):
        dim = j
        sum_1 = 0
        sum_2 = 0
        for i in range((dim - 1) // 2 + 1):
            sum_1 += comb(L, i)
        for i in range((dim) // 2 + 1):
            sum_2 += comb(L, i)
        if sum_1 <= right and sum_2 > right:
            d_min = dim
    for i in range(2 * L + 4):
        dim = i
        sum_1 = 0
        sum_2 = 0
        for j in range(dim):
            sum_1 += comb(L, j)
        for j in range(dim - 1):
            sum_2 += comb(L, j)
        if sum_1 >= right and sum_2 < right:
            d_max = dim
            break

    alpha_neg = L - 2 * d_max

    alpha_pos = L

    return d_max, d_min

def CSQ_init(n_class, bit):
    h_k = hadamard(bit)
    h_2k = np.concatenate((h_k, -h_k), 0)
    hash_center = h_2k[:n_class]

    if h_2k.shape[0] < n_class:
        hash_center = np.resize(hash_center, (n_class, bit))
        for k in range(10):
            for index in range(h_2k.shape[0], n_class):
                ones = np.ones(bit)
                ones[random.sample(list(range(bit)), bit // 2)] = -1
                hash_center[index] = ones
            c = []
            for i in range(n_class):
                for j in range(i, n_class):
                        c.append(sum(hash_center[i] != hash_center[j]))
            c = np.array(c)
            if c.min() > bit / 4 and c.mean() >= bit / 2:
                break
    return hash_center

def init_hash(n_class, bit):
    hash_centers = -1 + 2 * np.random.random((n_class, bit))
    hash_centers = np.sign(hash_centers)
    return hash_centers

@jit(nopython=True)
def cal_Cx(x, H):
    return np.dot(H, x)

@jit(nopython=True)
def cal_M(H):
    return np.dot(H.T, H)/H.shape[0]

@jit(nopython=True)
def cal_b(H):
    return np.dot(np.ones(H.shape[0], dtype=np.float64), H) / H.shape[0]

@jit(nopython=True)
def cal_one_hamm(b, H):
    temp = 0.5 * (b.shape[0] - np.dot(H, b))
    return temp.mean() + temp.min(), temp.min()

@jit(nopython=True)
def cal_hamm(H):
    dist = []
    for i in range(H.shape[0]):
        for j in range(i+1, H.shape[0]):
                TF = np.sum(H[i] != H[j])
                dist.append(TF)
    dist = np.array(dist)
    st = dist.mean() + dist.min()
    return st, dist.mean(), dist.min(), dist.var(), dist.max()

@jit(nopython=True)
def in_range(z1, z2, z3, bit):
    flag = True
    for item in z1:
        if item < -1 and item > 1:
            flag = False
            return flag
    for item in z3:
        if item < 0:
            flag = False
            return flag
    res = 0
    for item in z2:
        res += item**2
    if abs(res - bit) > 0.001:
        flag = False
        return flag
    return flag

@jit(nopython=True)
def get_min(b, H):
    temp = []
    for i in range(H.shape[0]):
        TF = np.sum(b != H[i])
        temp.append(TF)
    temp = np.array(temp)
    # print(temp.min())
    return temp.min()

@jit(nopython=True)
def Lp_box_one(b, H, d_max, n_class, bit, rho, gamma, error, W_ex, Wi):
    b = b.astype(np.float64)
    H = H.astype(np.float64)

    d = bit - 2 * d_max
    Wei_ = np.dot(W_ex, Wi)
    Wei_mean = np.mean(Wei_)
    Wei_ -= Wei_mean
    Wei = -bit + (Wei_-min(Wei_)) / (max(Wei_)-min(Wei_)) * (bit+bit)
    Wei = Wei.astype(np.float64)

    M = cal_M(H) #  n x n
    C = cal_b(H) #  n x 1
    out_iter = 10000
    in_iter = 10
    upper_rho = 1e9
    learning_fact = 1.07
    count = 0
    best_eval, best_min = cal_one_hamm(np.sign(b), H)
    best_B = b

    z1 = b.copy()
    z2 = b.copy()
    z3 = d - cal_Cx(np.sign(b), H) 
    y1 = np.random.rand(bit)
    y2 = np.random.rand(bit)
    y3 = np.random.rand(n_class-1)

    z1 = z1.astype(np.float64)
    z2 = z2.astype(np.float64)
    z3 = z3.astype(np.float64)
    y1 = y1.astype(np.float64)
    y2 = y2.astype(np.float64)
    y3 = y3.astype(np.float64)
    alpha = 1.0


    for e in range(out_iter):
        # print(e)
        for ei in range(in_iter):
            left = ((rho+rho) * np.eye(bit, dtype=np.float64) + (rho+2*alpha) * np.dot(H.T, H))
            left = left.astype(np.float64)
            right = (rho * z1 + rho * z2 + rho * np.dot(H.T, (d - z3)) - y1 - y2 - np.dot(H.T, y3) - C + 2*alpha*np.dot(H.T, Wei))
            right = right.astype(np.float64)
            b = np.dot(np.linalg.inv(left), right)


            z1 = b + 1/rho * y1

            z2 = b + 1/rho * y2

            z3 = d - np.dot(H, b) - 1/rho * y3

            if in_range(z1, z2, z3, bit):
                y1 = y1 + gamma * rho * (b - z1)
                y2 = y2 + gamma * rho * (b - z2)
                y3 = y3 + gamma * rho * (np.dot(H, b) + z3 - d)
                break
            else:
                z1[z1 > 1] = 1
                z1[z1 < -1] = -1

                norm_x = np.linalg.norm(z2)
                z2 = np.sqrt(bit) * z2 / norm_x

                z3[z3 < 0] = 0

                y1 = y1 + gamma * rho * (b - z1)
                y2 = y2 + gamma * rho * (b - z2)
                y3 = y3 + gamma * rho * (np.dot(H, b) + z3 - d)

        rho = min(learning_fact * rho, upper_rho)
        if rho == upper_rho:
            count += 1
            eval, mini = cal_one_hamm(np.sign(b), H)
            if eval > best_eval:
                best_eval = eval
                best_min = mini
                best_B = np.sign(b)
        if count == 100:
            # best_B = np.sign(b)
            break
    # best_B = np.sign(b)
    return best_B, H

@jit(nopython=True)
def Lp_box(B, best_B, n_class, d_max, bit, rho, gamma, error, best_st, W):
    count = 0
    for oo in range(20):
        for i in range(n_class):
            H = np.vstack((B[:i], B[i+1:])) # m-1 x n
            W_ex = np.vstack((W[:i], W[i+1:]))
            Wi = W[i]
            B[i], _ = Lp_box_one(B[i], H, d_max, n_class, bit, rho, gamma, error, W_ex, Wi)
        eval_st, eval_mean, eval_min, eval_var, eval_max = cal_hamm(B)
        print(eval_st, eval_min, eval_mean, eval_var, eval_max)
        if eval_st > best_st:
            best_st = eval_st
            best_B = B.copy()
            count = 0
        else:
            count += 1
        if count >= 2 and eval_min >= d_max:
            break
    return best_B

if __name__ == '__main__':
    config = {
        "cls_model": "ResNet",
        "dataset": "imagenet"
    }
    n_class = 100
    dataset = config["dataset"]
    for bit in [8, 16, 32, 64, 128]:
        print("bit:", bit)
        # load the semantic categories saved in weight folder
        # W = np.load("./weight/ResNet_imagenet_class_head_0.005.npy")
        W = np.load(f"./tmp_file/SHCIR/{config['cls_model']}_{config['dataset']}_class_head.npy")
        # n_class = 10
        initWithCSQ = True
        if bit == 48:
            initWithCSQ = False
        d_max, d_min = get_margin(bit, n_class)
        d_max = 0
        print(f"d_max is {d_max}, d_min is {d_min}")
        # parameter initialization
        rho = 5e-5
        gamma = (1+5**0.5)/2
        error = 1e-5
        # hash centers initialization
        random.seed(80)
        np.random.seed(80)
        d = bit - 2 * d_max
        if initWithCSQ:
            B = CSQ_init(n_class, bit) # initialize with CSQ
        else:
            B = init_hash(n_class, bit) # random initialization
        # metric initialization
        best_st, best_mean, best_min, best_var, best_max = cal_hamm(B)
        best_B = copy.deepcopy(B)
        if bit > 64:
            np.save(f'./tmp_file/SHCIR/SHCIR_{dataset}_{bit}.npy', best_B)
            exit()
        count = 0
        error_index = {}
        print(f"best_st is {best_st}, best_min is {str(best_min)}, best_mean is {best_mean}, best_var is {best_var}, best_max is {str(best_max)}")
        best_st = 0
        print(f"eval st, eval min, eval mean, eval var, eval max")
        begin = time.time()
        time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(begin))
        print(time_string)
        # print(B)
        # exit()
        best_B = Lp_box(B, best_B, n_class, d_max, bit, rho, gamma, error, best_st, W)
        end = time.time()
        time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
        print(time_string)
        ev_st, ev_mean, ev_min, ev_var, ev_max = cal_hamm(best_B)
        print(f"ev_st is {ev_st}, ev_min is {str(ev_min)}, ev_mean is {ev_mean}, ev_var is {ev_var}, ev_max is {str(ev_max)}")
        if(ev_min >= d_max):
            np.save(f'./tmp_file/SHCIR/SHCIR_{dataset}_{bit}.npy', best_B)
