# Import des librairies
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Fonction pour afficher les graphiques
def plot_data(title, x_label, y_label, data_1, data_2, labels, subplot_pos):
    plt.subplot(1, 3, subplot_pos)
    plt.plot(data_1, color='b', marker='+')
    plt.plot(data_2, color='r', marker='+')
    plt.title(title)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(labels, loc=0)

# Graphiques
plot_data('Marge du haut', 'nombre de billets', 'marge des billets (en cm)', vrai['margin_up'], faux['margin_up'], ["Vrai", "Faux"], 2)
plot_data('Longueur', 'nombre de billets', 'longueur des billets (en cm)', vrai['length'], faux['length'], ["Vrai", "Faux"], 3)
plt.show()

sns.pairplot(df[['is_genuine','diagonal','height_left','height_right','margin_up','length']], hue='is_genuine', corner=True)
plt.show()

# Régression logistique
y_billet = df["is_genuine"]
X_billet = sm.tools.add_constant(df.drop("is_genuine", axis=1))
reg_log = sm.Logit(y_billet, X_billet)
res_log = reg_log.fit()
X_billet = sm.add_constant(df[["height_right", "margin_up", "length"]])
model_reg_log = sm.Logit(y_billet, X_billet).fit()

df["proba"] = model_reg_log.predict(X_billet)
df["y_pred"] = (df["proba"] >= 0.5).astype(int)

# K-Means
n_clust = 2
features = ["diagonal","height_left","height_right","margin_low","margin_up","length"]
km = KMeans(n_clusters=n_clust, random_state=1994).fit(df[features])
clusters_km = km.labels_

pca_km = PCA(n_components=3).fit_transform(df[features])
plt.scatter(pca_km[:,0], pca_km[:,1], c=clusters_km)
plt.title("Projection des individus sur le premier plan factoriel")
plt.show()

# Comparatif des algorithmes
print('Régression logistique :')
metrics = [accuracy_score, precision_score, recall_score]
for metric in metrics:
    print(metric.__name__, metric(df["is_genuine"], df["y_pred"]))
    
print('\nK-means :')
for metric in metrics:
    print(metric.__name__, metric(df["is_genuine"], clusters_km))
