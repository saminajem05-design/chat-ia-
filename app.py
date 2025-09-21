"""Streamlit app for browsing banking regulations and offers using Gemini."""
from __future__ import annotations

import os
from textwrap import dedent
from typing import Optional

import pandas as pd
import streamlit as st

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover - streamlit runtime dependency
    raise ImportError(
        "The google-generativeai package is required. Install dependencies with `pip install -r requirements.txt`."
    ) from exc


DATA = pd.DataFrame(
    {
        "id": range(1, 6),
        "type": ["Règlementation", "Règlementation", "Offre", "Offre", "Contrat"],
        "titre": [
            "Durée maximale d'un crédit immobilier",
            "Conditions d’octroi d’un crédit consommation",
            "Livret A",
            "Assurance Vie",
            "Conditions générales Compte Courant",
        ],
        "description": [
            "La durée maximale d’un crédit immobilier est de 25 ans.",
            "Un crédit à la consommation ne peut pas excéder 75 000€.",
            "Taux réglementé, plafond fixé par l'État.",
            "Produit d’épargne à long terme, fiscalité avantageuse.",
            "Règles générales applicables à la gestion des comptes bancaires.",
        ],
    }
)


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    """Return the static dataset as a DataFrame."""
    return DATA.copy()


def build_context(df: pd.DataFrame) -> str:
    """Return a formatted context string for Gemini."""
    lines = [
        f"{row.type} : {row.titre}. {row.description}"
        for row in df.itertuples()
    ]
    return "\n".join(lines)


def get_api_key_from_config() -> Optional[str]:
    """Fetch the Gemini API key from Streamlit secrets or environment variables."""
    return st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))


def resolve_api_key() -> Optional[str]:
    """Return the stored manual key or fall back to configuration."""
    stored_key: Optional[str] = st.session_state.get("stored_api_key")
    return stored_key or get_api_key_from_config()


@st.cache_resource(show_spinner=False)
def load_model(api_key: str):
    """Configure and return the Gemini model instance."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def ask_gemini(model, question: str, context: str) -> str:
    """Generate a response from Gemini using the curated context."""
    prompt = dedent(
        f"""
        Tu es un assistant spécialisé en réglementation bancaire et offres produits.
        Voici les données disponibles :
        {context}

        Question : {question}

        Réponds clairement en citant uniquement les éléments pertinents (titre + résumé).
        Ajoute aussi un nombre total de résultats si applicable.
        """
    ).strip()

    response = model.generate_content(prompt)
    return response.text


def main() -> None:
    st.set_page_config(
        page_title="Assistant Réglementation & Offres",
        page_icon="📑",
        layout="wide",
    )

    st.title("📑 Assistant Réglementation & Offres")
    st.write(
        "Posez vos questions sur les règles, contrats ou offres bancaires comme "
        "dans un salon de discussion. Les réponses s'appuient sur les fiches internes disponibles."
    )

    df = get_dataset()
    context = build_context(df)

    with st.sidebar:
        st.header("🔍 Filtrer le référentiel")
        type_selection = st.multiselect(
            "Types à afficher",
            options=sorted(df["type"].unique()),
            default=list(sorted(df["type"].unique())),
        )

        st.markdown("---")
        st.header("🔐 Clé API Gemini")
        st.caption(
            "Collez votre clé API personnelle pour interroger Gemini."
        )

        stored_api_key = st.session_state.get("stored_api_key", "")
        manual_key = st.text_input(
            "Clé API",
            value=stored_api_key,
            type="password",
            help="La clé reste stockée uniquement dans votre session Streamlit.",
        )

        normalized_key = manual_key.strip() if manual_key else ""
        if normalized_key and normalized_key != stored_api_key:
            st.session_state["stored_api_key"] = normalized_key
            st.session_state.pop("model", None)
            st.session_state.pop("messages", None)
            st.success("Clé API prête à l'emploi.")
        elif not normalized_key and stored_api_key:
            st.session_state.pop("stored_api_key", None)
            st.session_state.pop("model", None)
            st.session_state.pop("messages", None)

        st.caption(
            "Vous pouvez aussi définir la clé via `.streamlit/secrets.toml` ou la variable "
            "d'environnement `GEMINI_API_KEY`."
        )

    filtered_df = df[df["type"].isin(type_selection)] if type_selection else df
    context = build_context(filtered_df)

    st.subheader("📚 Référentiel disponible")
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
    )

    if filtered_df.empty:
        st.warning(
            "Aucun document ne correspond aux filtres sélectionnés. "
            "Ajustez-les pour pouvoir interroger l'assistant."
        )
        return

    st.markdown("---")

    api_key = resolve_api_key()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.markdown("### 💬 Conversation")
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not api_key:
        st.info(
            "⚠️ Ajoutez votre clé API Gemini pour discuter avec l'assistant."
        )
        return

    if "model" not in st.session_state:
        try:
            st.session_state["model"] = load_model(api_key)
        except Exception as error:  # pragma: no cover - handled at runtime
            st.error(f"Impossible d'initialiser le modèle Gemini : {error}")
            return

    prompt = st.chat_input("Posez votre question sur les produits bancaires…")

    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consultation du modèle Gemini..."):
                try:
                    answer = ask_gemini(st.session_state["model"], prompt.strip(), context)
                except Exception as error:  # pragma: no cover - handled at runtime
                    st.error(f"Une erreur est survenue lors de l'appel à Gemini : {error}")
                else:
                    st.markdown(answer)
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": answer}
                    )


if __name__ == "__main__":
    main()
