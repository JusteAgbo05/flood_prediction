"""
utils.py — Fonctions réutilisables pour le projet Flood Prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time


def charger_donnees(chemin: str, cible: str = "FloodProbability") -> tuple:
    """
    Charge le dataset et sépare les features de la cible.

    Args:
        chemin : chemin vers le fichier CSV
        cible  : nom de la colonne cible

    Returns:
        X (DataFrame), y (Series)
    """
    df = pd.read_csv(chemin)
    features = [c for c in df.columns if c not in ["id", cible]]
    X = df[features]
    y = df[cible]
    print(f"✅ Dataset chargé : {df.shape[0]:,} lignes × {len(features)} features")
    return X, y


def evaluer_modele(nom: str, modele, X_train, X_test, y_train, y_test) -> dict:
    """
    Entraîne un modèle et retourne ses métriques de performance.

    Args:
        nom    : nom du modèle (pour l'affichage)
        modele : instance sklearn du modèle
        X_train, X_test, y_train, y_test : données splitées

    Returns:
        dict avec modèle, prédictions et métriques
    """
    t0 = time.time()
    modele.fit(X_train, y_train)
    duree = round(time.time() - t0, 2)

    pred_train = modele.predict(X_train)
    pred_test  = modele.predict(X_test)

    metriques = {
        "modele"      : modele,
        "predictions" : pred_test,
        "R2_train"    : r2_score(y_train, pred_train),
        "R2_test"     : r2_score(y_test, pred_test),
        "MAE"         : mean_absolute_error(y_test, pred_test),
        "RMSE"        : np.sqrt(mean_squared_error(y_test, pred_test)),
        "temps"       : duree,
    }

    print(f"✅ {nom}")
    print(f"   R² train : {metriques['R2_train']:.4f} | R² test : {metriques['R2_test']:.4f}")
    print(f"   MAE      : {metriques['MAE']:.4f}    | RMSE    : {metriques['RMSE']:.4f}")
    print(f"   Temps    : {duree}s\n")

    return metriques


def comparer_modeles(resultats: dict) -> pd.DataFrame:
    """
    Génère un tableau comparatif de tous les modèles testés.

    Args:
        resultats : dict {nom_modele: dict_metriques}

    Returns:
        DataFrame trié par R² décroissant
    """
    tableau = pd.DataFrame({
        nom: {
            "R² Train" : round(r["R2_train"], 4),
            "R² Test"  : round(r["R2_test"], 4),
            "MAE"      : round(r["MAE"], 4),
            "RMSE"     : round(r["RMSE"], 4),
            "Temps (s)": r["temps"],
        }
        for nom, r in resultats.items()
    }).T.sort_values("R² Test", ascending=False)

    print("=== COMPARAISON DES MODÈLES ===")
    print(tableau.to_string())
    print(f"\n🏆 Meilleur modèle : {tableau.index[0]} (R²={tableau['R² Test'].iloc[0]:.4f})")
    return tableau


def plot_residus(y_test, predictions, nom_modele: str = "Modèle"):
    """
    Affiche 3 graphiques d'analyse des résidus.

    Args:
        y_test       : valeurs réelles
        predictions  : valeurs prédites
        nom_modele   : nom du modèle pour le titre
    """
    residus = np.array(y_test) - np.array(predictions)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Résidus vs prédictions
    axes[0].scatter(predictions[:5000], residus[:5000], alpha=0.3, s=5, color="#378ADD")
    axes[0].axhline(0, color="#E24B4A", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Valeurs prédites")
    axes[0].set_ylabel("Résidus")
    axes[0].set_title("Résidus vs Prédictions")

    # Distribution des résidus
    axes[1].hist(residus, bins=60, color="#378ADD", edgecolor="white", linewidth=0.3)
    axes[1].axvline(0, color="#E24B4A", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Résidus")
    axes[1].set_ylabel("Fréquence")
    axes[1].set_title("Distribution des résidus")

    # Réel vs prédit
    axes[2].scatter(np.array(y_test)[:5000], predictions[:5000], alpha=0.3, s=5, color="#1D9E75")
    lim = [np.array(y_test).min(), np.array(y_test).max()]
    axes[2].plot(lim, lim, color="#E24B4A", linewidth=1.5, linestyle="--", label="Parfait")
    axes[2].set_xlabel("Valeurs réelles")
    axes[2].set_ylabel("Valeurs prédites")
    axes[2].set_title("Réel vs Prédit")
    axes[2].legend()

    plt.suptitle(f"Analyse des résidus — {nom_modele}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"plots/residus_{nom_modele.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   % erreur < 0.05 : {(np.abs(residus) < 0.05).mean()*100:.1f}%")


def plot_feature_importance(modele, features: list, nom_modele: str = "Modèle"):
    """
    Affiche l'importance des variables (pour les modèles basés sur des arbres).

    Args:
        modele     : modèle entraîné (doit avoir feature_importances_)
        features   : liste des noms de features
        nom_modele : nom du modèle pour le titre
    """
    if not hasattr(modele, "feature_importances_"):
        print("⚠️ Ce modèle n'a pas de feature_importances_. Utilise les coefficients.")
        return

    fi = pd.Series(modele.feature_importances_, index=features).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    fi.plot(kind="barh", ax=ax, color="#1D9E75", edgecolor="white")
    ax.set_title(f"Importance des variables — {nom_modele}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"plots/importance_{nom_modele.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    plt.show()
