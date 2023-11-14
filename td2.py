import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paramètres de la matrice
n = 100  # Nombre d'individus
p = 5    # Nombre de variables

# Création d'une matrice aléatoire
np.random.seed(0)  # Pour la reproductibilité
X = np.random.randn(n, p)

# Conversion en DataFrame pour une meilleure manipulation
df = pd.DataFrame(X, columns=[f'Variable {i+1}' for i in range(p)])

# Visualisation de la matrice
print("Visualisation des premières lignes de la matrice X :")
print(df.head())

# Calcul et affichage des indicateurs statistiques
print("\nIndicateurs statistiques pour chaque variable :")
print(df.describe())

# Calcul de la variance pour chaque variable
variance = df.var()
print("\nVariance pour chaque variable :")
print(variance)

# Calcul de l'écart-type pour chaque variable
ecart_type = df.std()
print("\nÉcart-type pour chaque variable :")
print(ecart_type)

# Calcul de la matrice de covariance
covariance = df.cov()
print("\nMatrice de covariance :")
print(covariance)

# Affichage de la matrice de covariance sous forme graphique
plt.figure(figsize=(10, 8))
sns.heatmap(covariance, annot=True, fmt=".2f")
plt.title("Matrice de covariance")
plt.show()
