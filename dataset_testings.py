import pandas as pd

#LINEAR REGRESSION
df = pd.read_csv("/content/drive/MyDrive/linear_regression_train.csv")
df2 = pd.read_csv("/content/drive/MyDrive/Linear Regression Test.csv")
x = df.drop(columns=['target']).values
y = df['target'].values
x_test = df2.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
lr_fit(x_train, y_train, lr=0.001, ep=4500)
my_pred = lr_predict(x_test)
print("R2 score:", r2_score(y_test, my_pred))

lr_fit(x, y, lr=0.001, ep=4500)
pred = lr_predict(x_test)
predictions_df = pd.DataFrame(pred, columns=['Prediction'])
predictions_df.to_csv('linear_regression_pred.csv', index=False)

print('Predictions saved')

#POLYNOMIAL REGRESSION
df = pd.read_csv("/content/drive/MyDrive/poly_train.csv")
df2 = pd.read_csv("/content/drive/MyDrive/poly_test.csv")
x = df.iloc[:,0].values.astype(float)
y = df.iloc[:,1].values.astype(float)
x_test = df2.iloc[:,0].values.astype(float)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
degree = 3
poly_fit(x_train, y_train, degree)
my_pred = poly_predict(x_test)
print("R2 Score:", r2_score(y_test, my_pred))

poly_fit(x, y, degree)
pred = poly_predict(x_test)
predictions_df = pd.DataFrame(pred, columns=['Prediction'])
predictions_df.to_csv('poly_pred.csv', index=False)

print('Predictions saved')

#LOGISIC REGRESSION
df = pd.read_csv('/content/drive/MyDrive/train_binary.csv')
df2 = pd.read_csv('/content/drive/MyDrive/test_binary.csv')
X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)
log_fit(X_train, y_train)
y_pred = log_predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

log_fit(X, y)
pred = log_predict(df2)
predictions_df = pd.DataFrame(pred, columns=['Prediction'])
predictions_df.to_csv('log_pred.csv', index=False)

print('Predictions saved')

#KNN (Multi Class Classification)
df = pd.read_csv("/content/drive/MyDrive/train_multi_class.csv")
df2 = pd.read_csv("/content/drive/MyDrive/test_multi_class.csv")
df = df.dropna(subset=["target"])

X = df.drop(columns=["target"]).to_numpy()
y = df["target"].to_numpy()
x_test = df2.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
k = 5
my_pred = knn_predict(x_train, y_train, x_test, k)
print("Accuracy:", accuracy_score(y_test, my_pred))

pred= knn_predict(X, y, x_test, k)
predictions_df = pd.DataFrame(pred, columns=['Prediction'])
predictions_df.to_csv('knn_multi_class_pred.csv', index=False)

print('Predictions saved')

#BEST K FINDER USING ELBOW METHOD FOR KMEANS
def kmeans_wcss(X, k, ep=200):
    y = kmeans_fit(X, k, ep)
    C = kmeans_model["C"]

    wcss = 0
    for i in range(k):
        Xi = X[y == i]
        if len(Xi) > 0:
            wcss += np.sum((Xi - C[i]) ** 2)
    return wcss

def find_best_k_elbow(X, k_min=1, k_max=10, ep=200):
    Ks = list(range(k_min, k_max + 1))
    wcss_vals = []
    for k in Ks:
        print(f"Running KMeans for k={k}")
        wcss_vals.append(kmeans_wcss(X, k, ep))
    return Ks, wcss_vals

import matplotlib.pyplot as plt

def plot_elbow(Ks, wcss_vals):
    plt.figure(figsize=(8,5))
    plt.plot(Ks, wcss_vals, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.show()

df = pd.read_csv("/content/drive/MyDrive/unsupervised_data.csv")
X = df.values.astype(float)
Ks, wcss_vals = find_best_k_elbow(X, 1, 10)
plot_elbow(Ks, wcss_vals)

#KMEANS
k_best = 9 
labels = kmeans_fit(X, k_best)
df3= pd.DataFrame()
df3['ID'] = df['ID']
df3["cluster"] = labels
df3.to_csv("unsupervised_clustered_kmean.csv", index=False)

print("Saved clustered file: unsupervised_clustered.csv")

#DECISION TREES (MULTI CLASS)
df = pd.read_csv("/content/drive/MyDrive/train_multi_class.csv")
df2 = pd.read_csv("/content/drive/MyDrive/test_multi_class.csv")
df = df.dropna(subset=["target"])
X = df.drop(columns=["target"]).to_numpy()
y = df["target"].to_numpy()
x_test = df2.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
tree_fit(x_train, y_train, max_depth=10)
y_pred=tree_predict(x_test)
acc=accuracy_score(y_test, y_pred)
print("Model accuracy:", acc)
tree_fit(X, y, max_depth=10)
y_pred=tree_predict(x_test)
predictions_df = pd.DataFrame(y_pred, columns=['Prediction'])
predictions_df.to_csv('tree_multi_class_pred.csv', index=False)

print('Predictions saved')

#DECISION TREES (BINARY CLASSIFICATION)
df = pd.read_csv('/content/drive/MyDrive/train_binary.csv')
df2 = pd.read_csv('/content/drive/MyDrive/test_binary.csv')
X = df.drop(columns=['label'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
tree_fit(X_train, y_train, max_depth=10)
y_pred=tree_predict(X_test)
acc=accuracy_score(y_test, y_pred)
print(acc)
tree_fit(X, y, max_depth=10)
y_pred=tree_predict(df2)
predictions_df = pd.DataFrame(pred, columns=['Prediction'])
predictions_df.to_csv('tree_binary_pred.csv', index=False)

print('Predictions saved')

#NEURAL NETWORK
#---CLASSIFIACTION--
y_cls = np.array(data['target_cls']).reshape(-1)
num_classes = len(np.unique(y_cls))
ls_cls = [X.shape[1], 16, 8, num_classes]

xtr, xts, ytr, yts = train_test_split(X, y_cls, test_size=0.3)
nn_fit(xtr, ytr, ls_cls, lr=0.01, ep=2000, task="clf_multi")
yp_cls = nn_predict(xts)
print("Classification Accuracy:", accuracy_score(yts, yp_cls))
nn_fit(X, y_cls, ls_cls, lr=0.01, ep=2000, task="clf_multi")
yp_cls = nn_predict(data2)  # must match preprocessing
print('Predictions made for classification')
#---REGRESSION----
y_reg = np.array(data['target_reg']).reshape(-1,1)
ls_reg = [X.shape[1], 16, 8, 1]
xtr, xts, ytr, yts = train_test_split(X, y_reg, test_size=0.2)
nn_fit(xtr, ytr, ls_reg, lr=0.01, ep=3000, task="reg")
yp_reg = nn_predict(xts)
print("Regression R2 score:", r2_score(yts, yp_reg))
nn_fit(X, y_reg, ls_reg, lr=0.01, ep=3000, task="reg")
yp_reg = nn_predict(data2)  # must match preprocessing
print('Predictions made for regression')
yp_cls = yp_cls.ravel()
yp_reg = yp_reg.ravel()

predictions_df = pd.DataFrame({'cls_pred': yp_cls, 'reg_pred': yp_reg})
predictions_df.to_csv('nn_pred.csv', index=False)
print('Predictions saved')
