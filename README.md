# Bank Churn Prediction – MLOps Project (Azure + FastAPI)




---

##  Objectif du projet
L’objectif de ce projet est de **prédire le churn bancaire** (résiliation d’un client) à partir de données clients en utilisant :
- un **modèle de Machine Learning**
- une **API FastAPI**
- une **conteneurisation Docker**
- un **déploiement automatique sur Microsoft Azure**
- une **pipeline CI/CD GitHub Actions**

👉 Le projet est 100 % fonctionnel et déployé dans le cloud.

---

##  Problématique métier
Le churn bancaire correspond à la perte de clients.  
Pouvoir prédire ce comportement permet à une banque de :
- identifier les clients à risque
- mettre en place des actions de rétention
- réduire les pertes financières

---
## Entraînement du modèle
 - Algorithme : Scikit-learn (classification)
 - Données : data/churn.csv
 - Script : train.py
 - Sortie : model/model.pkl

Le modèle est entraîné localement puis sauvegardé afin d’être chargé par l’API.

---
## Swagger
 - La documentation Swagger est accessible à l’adresse suivante :
🔗 Swagger UI

https://churn-api-ines-060126.azurewebsites.net/docs

---
##  Déploiement Azure
 - URL publique de l’application
   
🔗 Application Web

https://churn-api-ines-060126.azurewebsites.net

🔗 Health Check

https://churn-api-ines-060126.azurewebsites.net/health

---
## CI/CD – GitHub Actions
 - Le pipeline CI/CD est défini dans :
   .github/workflows/deploy.yml
   
Fonctionnalités :
 - Build automatique de l’image Docker
 - Push vers Azure Container Registry
 - Déploiement automatique vers Azure App Service

 ---
## Interface Web (Frontend)
Une interface web simple permet de tester les prédictions :
 - Formulaire de saisie des données client
 - Bouton Predict
 - Affichage du risque de churn et de la probabilité

🔗 Interface Web

https://churn-api-ines-060126.azurewebsites.net


 ---
## Tests réalisés
 - Test /health
 - Test /predict via Swagger
 - Test /predict via interface web
 - Test Docker local
 - Test déploiement Azure
---
### MLflow (Local)

Dans ce projet, MLflow est utilisé **en local** pour tracker les expérimentations du modèle :
- paramètres (type de modèle, test_size, etc.)
- métriques (accuracy, f1, precision, recall, roc_auc)
- artifacts (confusion_matrix.txt, classification_report.txt)
- modèle ML (loggé dans MLflow)

Commande utilisée :

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001


accés :

http://127.0.0.1:5001



## Drift Detection (Evidently)

Cette partie vérifie si la distribution des données a changé (data drift).

Commande utilisée :

start reports\drift_report.html






   ---
   Étudiante : Ines Elouaer
   Établissement : Polytech Sousse
   Année : 2025 / 2026
