# Assistant Réglementation & Offres

Application Streamlit simple pour interroger un petit référentiel de règlements et d'offres bancaires à l'aide du modèle Gemini.

## Prérequis

- Python 3.9+
- Une clé API valide pour [Google Gemini](https://ai.google.dev/)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration de la clé API

Vous pouvez :

1. Coller votre clé directement dans la barre latérale de l'application Streamlit (champ **Clé API**). Elle reste stockée uniquement dans votre session.
2. Définir la variable d'environnement `GEMINI_API_KEY` **ou** créer un fichier `.streamlit/secrets.toml` contenant :

   ```toml
   GEMINI_API_KEY = "votre_cle_api"
   ```

## Lancer l'application

```bash
streamlit run app.py
```

Ouvrez ensuite l'URL fournie par Streamlit (par défaut `http://localhost:8501`).

## Utilisation

1. Filtrez le référentiel et fournissez votre clé API dans la barre latérale.
2. Posez vos questions via la zone de discussion en bas de page (style ChatGPT).
3. Chaque réponse du modèle Gemini s'appuie sur le contexte filtré et est ajoutée à l'historique de conversation.
