# Interface utilisateur Flask pour la prédiction de maladie

Cette application Flask permet de sélectionner des symptômes et de prédire la maladie la plus probable.

## Installation

1. Créez un environnement virtuel Python :
   ```bash
   python -m venv venv
   ```
2. Activez l'environnement :
   - Windows : `venv\Scripts\activate`
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Exécution

Lancez l'application avec :

```bash
python app.py
```

Puis ouvrez `http://127.0.0.1:5000` dans votre navigateur.

## Structure

- `app.py` : application Flask.
- `templates/index.html` : interface utilisateur.
- `dataset_medical_final_sans_sexe.csv` : jeu de données utilisé pour entraîner le modèle.
