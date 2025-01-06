# Projet : Maintenance Prédictive - Pyspark

## Description
La maintenance prédictive est un sujet crucial dans l'industrie pour prévoir l'état des actifs afin d'éviter les arrêts de production et les défaillances. pour prévoir l'état des actifs afin d'éviter les arrêts de production et les défaillances. Cet ensemble de données est une version Kaggle du célèbre jeu de données public de modélisation de la dégradation des actifs de la NASA. Il contient des données simulées de Run-to-Failure (fonctionnement jusqu'à la défaillance) provenant de moteurs à réaction turbofan.
Cet ensemble de données est disponible publiquement et peut être téléchargé via ce  [lien](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

La simulation de dégradation du moteur a été réalisée à l'aide du logiciel C-MAPSS (Commercial Modular Aero-Propulsion System Simulation). Quatre ensembles distincts ont été simulés sous différentes combinaisons de conditions opérationnelles et de modes de panne. Plusieurs canaux de capteurs ont été enregistrés pour caractériser l'évolution des défaillances. L'ensemble de données a été fourni par le Prognostics Center of Excellence (CoE) de la NASA Ames Research Center.


### Objectifs Généraux
1. Prédire le Remaining Useful Life (RUL) des moteurs à l'aide de modèles de régression.
2. Classifier les moteurs en zones d'alerte (« Warning » ou « Alarm ») via des modèles de classification binaire.


## Approches Utilisées

### Préparation des Données
- Nettoyage des données :
  - Supprimer les caractéristiques à faible variance
  - Supprimer le bruit 
  - Normalisation des variables continues.
  

### Régression
1. **Régression Linéaire Multiple** :
2. **Arbre de Décision de Régression** :

### Classification Binaire
1. **Régression Logistique** :
2. **Arbre de Décision** :
3. **Random Forest** :

### Évaluation
- **Régression** :
  - RMSE (Root Mean Squared Error).
  - R² (Coefficient de détermination).
- **Classification** :
  - Précision.
  - F1-score.
  - Courbe ROC-AUC.



## Fonctionnalités Implémentées
1. Prédiction du Remaining Useful Life (RUL) via des modèles de régression.
2. Classification des moteurs en zones Warning et Alarm.

## Prérequis
- Python 3.8 ou supérieur.
- Pyspark 3.4.2
- Bibliothèques Python :
  - `scikit-learn`
  - `matplotlib`
  - `numpy`

## Instructions pour Exécution
1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/ton-projet/predictive-maintenance.git
   ```


## Auteurs
- Zineb MANAD
- Thiziri KASHI
