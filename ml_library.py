#importing modules 
import numpy as np
from collections import Counter

#Preprocess data, to scale it and deal will emply datapoints
def scalar(X, stats=None):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1,1)
    if stats is None:
        mean = np.nanmean(X, axis=0)
        std  = np.nanstd(X, axis=0)
        mean = np.where(np.isnan(mean), 0.0, mean)
        std  = np.where(std == 0, 1.0, std)
        stats = (mean, std)
    mean, std = stats
    X = (X - mean) / std
    return X, stats

def handle_nan(X, fill=None):
    X = np.asarray(X, float)
    if fill is None:
        fill = np.nanmean(X, axis=0)
        fill = np.where(np.isnan(fill), 0.0, fill)
    idx = np.isnan(X)
    X[idx] = np.take(fill, np.where(idx)[1])
    return X, fill

def preprocess(X, stats=None, fill=None):
    X, fill = handle_nan(X, fill)
    X, stats = scalar(X, stats)
    return X, stats, fill

#Train Test Split for testing purpose
def train_test_split(X, y, test_size=0.2):
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    test_n = int(n * test_size)
    split = n - test_n
    X_train = X[:split]
    X_test  = X[split:]
    y_train = y[:split]
    y_test  = y[split:]
    return X_train, X_test, y_train, y_test

#Metrics fuctions for evaluations
def accuracy_score(y_pred, y_act):
    y_pred=np.asarray(y_pred,float).ravel()
    y_act=np.asarray(y_act,float).ravel()
    m=np.isfinite(y_pred)&np.isfinite(y_act)
    diff=y_pred[m]-y_act[m]
    return np.sum(diff==0)/len(diff)

def r2_score(y_true,y_pred):
    y_true=np.asarray(y_true,float)
    y_pred=np.asarray(y_pred,float)
    m=np.isfinite(y_true)&np.isfinite(y_pred)
    y_true=y_true[m]
    y_pred=y_pred[m]
    ss_res=np.sum((y_true-y_pred)**2)
    ss_tot=np.sum((y_true-np.mean(y_true))**2)
    return 1-ss_res/ss_tot

#LINEAR REGRESSION
def lr_fit(X, y, lr=0.01, ep=2000):
    global lr_model
    lr_model={}
    X, stats, fill = preprocess(X)
    y = np.asarray(y, float)
    w = np.zeros(X.shape[1])
    b = 0
    for i in range(ep):
        yp = X @ w + b
        err = yp - y
        w -= lr * (X.T @ err) / len(y)
        b -= lr * err.mean()
        if i % 500 == 0:
            print("ep", i, "mse", np.mean(err**2))
    lr_model["w"] = w
    lr_model["b"] = b
    lr_model["stats"] = stats
    lr_model["fill"] = fill
    return lr_model

def lr_predict(X):
    X,_,_ = preprocess(X, lr_model["stats"], lr_model["fill"])
    y = X @ lr_model["w"] + lr_model["b"]
    return y

#POLYNOMIAL REGRESSION
def poly_feat(x,d):
    return np.hstack([x**i for i in range(d+1)])

def poly_fit(x,y,d):
    global poly_model
    x=np.asarray(x,float)
    y=np.asarray(y,float)
    mask=np.isfinite(y)
    x=x[mask]
    y=y[mask]
    x,stats,fill=preprocess(x.reshape(-1,1))
    X=poly_feat(x,d)
    w=np.linalg.pinv(X.T@X)@X.T@y     #we using pseudoinverse as the matrix is nonsquared, this process help minimise the least square, keeping the code vectorised
    poly_model={"w":w,"d":d,"stats":stats,"fill":fill}
    return poly_model

def poly_predict(x):
    x=np.asarray(x,float)
    x,_,_=preprocess(x.reshape(-1,1),poly_model["stats"],poly_model["fill"])
    X=poly_feat(x,poly_model["d"])
    return X@poly_model["w"]

#LOGISTIC REGRESSION
def log_fit(X, y, lr=0.01, epochs=1000):
    global log_model
    log_model={}
    X, stats, fill = preprocess(X)
    y = np.asarray(y).reshape(-1,1)
    m, n = X.shape
    Xb = np.c_[np.ones((m,1)), X]
    theta = np.random.randn(n+1,1)*0.01
    for _ in range(epochs):
        z = Xb @ theta
        h = sigmoid(z)
        grad = (Xb.T @ (h - y)) / m
        theta -= lr * grad
    log_model["theta"] = theta
    log_model["stats"] = stats
    log_model["fill"] = fill
    return log_model

def log_predict(X):
    X, _, _ = preprocess(X, log_model["stats"], log_model["fill"])
    Xb = np.c_[np.ones((X.shape[0],1)), X]
    return (sigmoid(Xb @ log_model["theta"]) >= 0.5).astype(int)

#K-NEAREST NEIGHBOUR
def knn_predict(X, y, x, k=5):
    X, stats, fill = preprocess(X)
    x, _, _ = preprocess(x, stats, fill)
    y = np.asarray(y).reshape(-1)
    global_majority = Counter(y).most_common(1)[0][0]
    preds = []
    for i in range(x.shape[0]):
        dists = np.sum((X - x[i])**2, axis=1)
        if np.all(np.isnan(dists)):
            preds.append(global_majority)
            continue
        idx = np.argpartition(dists, k)[:k]
        labels = y[idx]
        distances = dists[idx]
        weights = 1.0 / (distances + 1e-8)
        vote = {}
        for lbl, w in zip(labels, weights):
            vote[lbl] = vote.get(lbl, 0) + w
        preds.append(max(vote, key=vote.get))
        if i % 1000 == 0:
            print("datapoints predicted :", i)
    return np.array(preds, dtype=int)

#KMEANS
def k_init(X, k):
    C = [X[np.random.randint(len(X))]]
    for _ in range(1, k):
        d = np.min([np.sum((X-c)**2,axis=1) for c in C], axis=0)
        p = d / np.sum(d)
        C.append(X[np.random.choice(len(X), p=p)])
    return np.array(C)

def kmeans_fit(X, k=3, ep=200):
    global kmeans_model
    kmeans_model = {}
    X, stats, fill = preprocess(X)
    C = k_init(X, k)
    for _ in range(ep):
        D = np.array([np.sum((X-c)**2,axis=1) for c in C])
        y = np.argmin(D, axis=0)
        Cnew = []
        for i in range(k):
            if np.any(y==i):
                Cnew.append(X[y==i].mean(axis=0))
            else:
                Cnew.append(C[i])
        Cnew = np.array(Cnew)
        if np.allclose(C, Cnew):
          break
        C = Cnew
    kmeans_model["C"] = C
    kmeans_model["stats"] = stats
    kmeans_model["fill"] = fill
    return y

def kmeans_predict(X):
    X,_,_ = preprocess(X, kmeans_model["stats"], kmeans_model["fill"])
    D = np.array([np.sum((X-c)**2,axis=1) for c in kmeans_model["C"]])
    return np.argmin(D, axis=0)

#DECISION TREE
def entropy(y):
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-8))

def best_split(X, y, n_thresholds=20):
    best_gain = -1
    best_f, best_t = None, None
    n_samples, n_features = X.shape

    parent_entropy = entropy(y)

    for f in range(n_features):
        col = X[:, f]
        thresholds = np.linspace(np.min(col), np.max(col), n_thresholds)
        for t in thresholds:
            left_mask = col <= t
            right_mask = col > t
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            n = len(y)
            e = (left_mask.sum()/n)*entropy(y[left_mask]) + (right_mask.sum()/n)*entropy(y[right_mask])
            g = parent_entropy - e
            if g > best_gain:
                best_gain = g
                best_f, best_t = f, t
    return best_f, best_t

def tree_generation(X, y, depth=0, max_depth=10):
    if len(np.unique(y)) == 1:
        return {"value": y[0]}
    if X.shape[0] == 0 or depth == max_depth:
        return {"value": np.bincount(y).argmax()}

    f, t = best_split(X, y)
    if f is None:
        return {"value": np.bincount(y).argmax()}

    left_mask = X[:, f] <= t
    right_mask = X[:, f] > t

    return {"f": f, "t": t, "left": tree_generation(X[left_mask], y[left_mask], depth+1, max_depth), "right": tree_generation(X[right_mask], y[right_mask], depth+1, max_depth)}

def tree_fit(X, y, max_depth=10):
    global tree_model
    tree_model = {}
    X, stats, fill = preprocess(X)
    y = np.asarray(y, int)

    tree = tree_generation(X, y, 0, max_depth)
    tree_model["tree"] = tree
    tree_model["stats"] = stats
    tree_model["fill"] = fill
    return tree_model

def tree_predict_one(x, tree):
    if "value" in tree:
        return tree["value"]
    if x[tree["f"]] <= tree["t"]:
        return tree_predict_one(x, tree["left"])
    else:
        return tree_predict_one(x, tree["right"])

def tree_predict(X):
    X, _, _ = preprocess(X, tree_model["stats"], tree_model["fill"])
    return np.array([tree_predict_one(x, tree_model["tree"]) for x in X])

#NEURAL NETWORK
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

def init_params(ls):
    w, b, mw, vw, mb, vb = [], [], [], [], [], []
    for i in range(len(ls)-1):
        lim = np.sqrt(1 / ls[i])  # scale weights by sqrt(1/n) so neuron activations keep constant variance across layers
        wi = np.random.randn(ls[i], ls[i+1]) * lim    #initialising weight with normal distribution from 0 to 1/n
        bi = np.zeros((1, ls[i+1]))
        w.append(wi)
        b.append(bi)
        mw.append(np.zeros_like(wi))
        vw.append(np.zeros_like(wi))
        mb.append(np.zeros_like(bi))
        vb.append(np.zeros_like(bi))
    return w, b, mw, vw, mb, vb


def fwd(x, w, b, task):
    acts = [x]
    a = x
    for i in range(len(w)):
        z = a @ w[i] + b[i]
        if i == len(w)-1:
            if task == "reg":
                a = z
            else:  # binary or multiclass
                a = sigmoid(z)
        else:
            a = sigmoid(z)
        acts.append(a)
    return acts

def loss(y, yp, task):
    eps = 1e-8
    yp = np.clip(yp, eps, 1-eps)
    if task == "clf":  # binary
        return -np.mean(y*np.log(yp) + (1-y)*np.log(1-yp))
    elif task == "clf_multi":  # multiclass: binary sigmoid per output
        return -np.mean(np.sum(y*np.log(yp) + (1-y)*np.log(1-yp), axis=1))
    else:  # regression
        return np.mean((y - yp)**2)
def back(y, w, b, acts, lr, mw, vw, mb, vb, t, task):
    m = y.shape[0]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    if task in ["clf", "clf_multi", "reg"]:
        da = (acts[-1] - y) / m
    for i in reversed(range(len(w))):
        ap, ac = acts[i], acts[i+1]
        dz = da if (i == len(w)-1 and task == "reg") else da * sigmoid_derivative(ac)
        dw = ap.T @ dz
        db = np.sum(dz, axis=0, keepdims=True)
        da = dz @ w[i].T
        #Adam optimisation
        # θ ← θ−α*((∂L/∂θ·β1^t-corrected)/(sqrt((∂L/∂θ)^2·β2^t-corrected)+ε))
        mw[i] = beta1 * mw[i] + (1 - beta1) * dw
        vw[i] = beta2 * vw[i] + (1 - beta2) * (dw**2)
        mb[i] = beta1 * mb[i] + (1 - beta1) * db
        vb[i] = beta2 * vb[i] + (1 - beta2) * (db**2)
        mwc = mw[i] / (1 - beta1**t)
        vwc = vw[i] / (1 - beta2**t)
        mbc = mb[i] / (1 - beta1**t)
        vbc = vb[i] / (1 - beta2**t)
        w[i] -= lr * mwc / (np.sqrt(vwc) + eps)
        b[i] -= lr * mbc / (np.sqrt(vbc) + eps)

def nn_fit(x, y, ls, lr=0.01, ep=3000, task="clf"):
    global nn_model
    nn_model = {}
    x, stats, fill = preprocess(x)
    y = np.asarray(y)
    if task == "clf_multi":
        y = y.reshape(-1)
        classes = np.unique(y)               # automatically find all classes
        nn_model["classes"] = classes        # store mapping
        class_to_index = {c:i for i,c in enumerate(classes)}
        y = np.array([class_to_index[c] for c in y])
        num_classes = len(classes)
        ls[-1] = num_classes                 # adjust last layer size
        y = np.eye(num_classes)[y]           # one-hot
    w, b, mw, vw, mb, vb = init_params(ls)
    t = 0
    for i in range(ep):
        t += 1
        acts = fwd(x, w, b, task)
        l = loss(y, acts[-1], task)
        back(y, w, b, acts, lr, mw, vw, mb, vb, t, task)
        if i % 500 == 0:
            print("ep", i, "loss", l)
    nn_model["w"] = w
    nn_model["b"] = b
    nn_model["stats"] = stats
    nn_model["fill"] = fill
    nn_model["task"] = task
    return nn_model

def nn_predict(x):
    x, _, _ = preprocess(x, nn_model["stats"], nn_model["fill"])
    out = fwd(x, nn_model["w"], nn_model["b"], nn_model["task"])[-1]
    if nn_model["task"] == "clf":
        return (out > 0.5).astype(int)
    elif nn_model["task"] == "clf_multi":
        # return original class labels
        idx = np.argmax(out, axis=1)
        return np.array([nn_model["classes"][i] for i in idx])
    return out

