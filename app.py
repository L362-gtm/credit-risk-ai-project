
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "german.data"
MODEL_PATH = BASE_DIR / "models" / "credit_model.pkl"

FEATURE_COLUMNS = [
    "checking_account", "duration", "credit_history",
    "purpose", "credit_amount", "savings",
    "employment", "installment_rate", "personal_status",
    "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing",
    "existing_credits", "job", "dependents",
    "telephone", "foreign_worker"
]

COLUMNS = FEATURE_COLUMNS + ["target"]

st.set_page_config(
    page_title="Credit Risk AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. DESIGN PROFESSIONNEL
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #F4F6FA;
    color: #17233C;
}

[data-testid="stSidebar"] {
    background-color: #101C35;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    color: #142544;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    color: #78849A;
    font-size: 14px;
    margin-bottom: 25px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #E8ECF3;
    box-shadow: 0 3px 12px rgba(20,35,60,0.04);
    min-height: 125px;
}

.metric-label {
    color: #7B879B;
    font-size: 12px;
    font-weight: 600;
}

.metric-value {
    color: #142544;
    font-size: 27px;
    font-weight: 800;
    margin-top: 10px;
}

.section-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #E8ECF3;
    margin-bottom: 15px;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background-color: #D5A94B;
    color: #142544;
    border: none;
    border-radius: 9px;
    font-weight: 700;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background-color: #E5BC65;
    color: #142544;
}

h1, h2, h3 {
    color: #142544;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #E8ECF3;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. DICTIONNAIRE DES CATEGORIES
# =====================================================

LABELS = {

    "checking_account": {
        "A11": "Solde négatif",
        "A12": "Solde inférieur à 200 DM",
        "A13": "Solde supérieur ou égal à 200 DM",
        "A14": "Aucun compte courant"
    },

    "credit_history": {
        "A30": "Aucun crédit antérieur ou crédits remboursés",
        "A31": "Crédits antérieurs remboursés auprès de cette banque",
        "A32": "Crédits existants remboursés normalement",
        "A33": "Retards de remboursement antérieurs",
        "A34": "Historique de crédit critique"
    },

    "purpose": {
        "A40": "Achat d'une voiture neuve",
        "A41": "Achat d'une voiture d'occasion",
        "A42": "Mobilier et équipements",
        "A43": "Radio et télévision",
        "A44": "Appareils électroménagers",
        "A45": "Réparations",
        "A46": "Formation et études",
        "A48": "Reconversion professionnelle",
        "A49": "Activité professionnelle",
        "A410": "Autres projets"
    },

    "savings": {
        "A61": "Épargne inférieure à 100 DM",
        "A62": "Épargne entre 100 et 500 DM",
        "A63": "Épargne entre 500 et 1 000 DM",
        "A64": "Épargne supérieure ou égale à 1 000 DM",
        "A65": "Épargne inconnue ou inexistante"
    },

    "employment": {
        "A71": "Sans emploi",
        "A72": "Moins d'un an",
        "A73": "Entre 1 et 4 ans",
        "A74": "Entre 4 et 7 ans",
        "A75": "7 ans ou plus"
    },

    "personal_status": {
        "A91": "Homme divorcé ou séparé",
        "A92": "Femme divorcée, séparée ou mariée",
        "A93": "Homme célibataire",
        "A94": "Homme marié ou veuf",
        "A95": "Femme célibataire"
    },

    "other_debtors": {
        "A101": "Aucun autre débiteur",
        "A102": "Co-emprunteur",
        "A103": "Garant"
    },

    "property": {
        "A121": "Bien immobilier",
        "A122": "Épargne immobilière ou assurance-vie",
        "A123": "Véhicule ou autre bien",
        "A124": "Aucun bien connu"
    },

    "other_installment_plans": {
        "A141": "Plan de paiement auprès d'une banque",
        "A142": "Plan de paiement auprès d'un magasin",
        "A143": "Aucun autre plan de paiement"
    },

    "housing": {
        "A151": "Locataire",
        "A152": "Propriétaire",
        "A153": "Logement gratuit"
    },

    "job": {
        "A171": "Sans emploi ou non qualifié non résident",
        "A172": "Travailleur non qualifié résident",
        "A173": "Employé qualifié",
        "A174": "Cadre, indépendant ou hautement qualifié"
    },

    "telephone": {
        "A191": "Pas de téléphone enregistré",
        "A192": "Téléphone enregistré"
    },

    "foreign_worker": {
        "A201": "Oui",
        "A202": "Non"
    }
}

FIELD_LABELS = {
    "checking_account": "Situation du compte courant",
    "credit_history": "Historique de crédit",
    "purpose": "Objet du crédit",
    "savings": "Situation de l'épargne",
    "employment": "Ancienneté professionnelle",
    "personal_status": "Situation personnelle",
    "other_debtors": "Autres débiteurs",
    "property": "Patrimoine",
    "other_installment_plans": "Autres plans de paiement",
    "housing": "Situation du logement",
    "job": "Profession",
    "telephone": "Téléphone",
    "foreign_worker": "Travailleur étranger"
}

def display_category(field, code):
    return LABELS[field][code]


# =====================================================
# 4. CHARGEMENT DES DONNEES ET DU MODELE
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS
    )

    if len(df.columns) != 21:
        raise ValueError("Le dataset doit contenir 21 colonnes.")

    if not set(df["target"].unique()).issubset({1, 2}):
        raise ValueError("Les valeurs de la variable target sont incorrectes.")

    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:

    data = load_data()
    model = load_model()

    X = data[FEATURE_COLUMNS].copy()
    y = (data["target"] == 1).astype(int)

except Exception as error:

    st.error("Impossible de charger les données ou le modèle.")

    st.code(str(error))

    st.info(
        "Vérifie les fichiers data/german.data "
        "et models/credit_model.pkl."
    )

    st.stop()


# =====================================================
# 5. EVALUATION DU MODELE
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)

good_index = list(model.classes_).index(1)

good_probabilities = probabilities[:, good_index]

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

auc = roc_auc_score(
    y_test, good_probabilities
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)


# =====================================================
# 6. COMPOSANT POUR LES CARTES
# =====================================================

def metric_card(label, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def page_header(title, subtitle):

    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


# =====================================================
# 7. NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown("## 💳 CREDIT RISK AI")

    st.caption("INTELLIGENT CREDIT ANALYSIS")

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "📋 Évaluation du crédit",
            "📊 Analyse du modèle",
            "ℹ️ À propos"
        ]
    )

    st.markdown("---")

    st.caption("MACHINE LEARNING")

    st.markdown("**Random Forest Classifier**")

    st.caption("Version 1.0")


# =====================================================
# 8. DASHBOARD
# =====================================================

if page == "🏠 Dashboard":

    page_header(
        "Credit Risk Dashboard",
        "Vue générale des données et des performances du modèle."
    )

    total = len(data)

    good = int((data["target"] == 1).sum())

    bad = int((data["target"] == 2).sum())

    good_rate = good / total * 100

    bad_rate = bad / total * 100

    # Cartes statistiques

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("DOSSIERS ANALYSÉS", f"{total:,}".replace(",", " "))

    with col2:
        metric_card("BONS CRÉDITS", f"{good:,}".replace(",", " "))

    with col3:
        metric_card("MAUVAIS CRÉDITS", f"{bad:,}".replace(",", " "))

    with col4:
        metric_card("TAUX DE BONS CRÉDITS", f"{good_rate:.1f}%")

    st.write("")

    st.subheader("Vue d'ensemble du portefeuille")

    left, right = st.columns(2)

    with left:

        st.markdown("**Répartition des dossiers de crédit**")

        distribution = pd.DataFrame({
            "Nombre de dossiers": [good, bad]
        }, index=["Bon crédit", "Mauvais crédit"])

        st.bar_chart(distribution)

        st.caption(
            f"Bons crédits : {good_rate:.1f}% | "
            f"Mauvais crédits : {bad_rate:.1f}%"
        )

    with right:

        st.markdown("**Distribution des montants de crédit**")

        amount_distribution = pd.cut(
            data["credit_amount"],
            bins=8
        ).value_counts(sort=False)

        amount_distribution.index = (
            amount_distribution.index.astype(str)
        )

        amount_distribution = amount_distribution.rename(
            "Nombre de dossiers"
        )

        st.bar_chart(amount_distribution)

    st.divider()

    st.subheader("Profil général des demandeurs")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Répartition par tranche d'âge**")

        age_groups = pd.cut(
            data["age"],
            bins=[0, 25, 35, 45, 55, 100],
            labels=[
                "18-25 ans",
                "26-35 ans",
                "36-45 ans",
                "46-55 ans",
                "Plus de 55 ans"
            ]
        )

        age_distribution = age_groups.value_counts(
            sort=False
        )

        st.bar_chart(age_distribution)

    with col2:

        st.markdown("**Durée des crédits**")

        duration_distribution = pd.cut(
            data["duration"],
            bins=6
        ).value_counts(sort=False)

        duration_distribution.index = (
            duration_distribution.index.astype(str)
        )

        st.bar_chart(duration_distribution)

    st.divider()

    st.subheader("Performances du modèle")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{accuracy * 100:.2f}%")

    c2.metric("Precision", f"{precision * 100:.2f}%")

    c3.metric("Recall", f"{recall * 100:.2f}%")

    c4.metric("ROC-AUC", f"{auc:.3f}")

    st.caption(
        "Les statistiques du portefeuille sont calculées sur "
        "le dataset complet. Les performances sont calculées "
        "sur les 200 dossiers du jeu de test."
    )


# =====================================================
# 9. EVALUATION DU CREDIT
# =====================================================

elif page == "📋 Évaluation du crédit":

    page_header(
        "Évaluation du crédit",
        "Renseignez les informations du demandeur pour "
        "obtenir une estimation du modèle."
    )

    with st.form("credit_form"):

        st.subheader("01 — Informations financières")

        col1, col2 = st.columns(2)

        with col1:

            credit_amount = st.number_input(
                "Montant du crédit (DM)",
                min_value=1,
                max_value=100000,
                value=2500
            )

            duration = st.number_input(
                "Durée du crédit (mois)",
                min_value=1,
                max_value=120,
                value=24
            )

            checking = st.selectbox(
                "Situation du compte courant",
                list(LABELS["checking_account"]),
                format_func=lambda x: display_category(
                    "checking_account", x
                )
            )

            savings = st.selectbox(
                "Situation de l'épargne",
                list(LABELS["savings"]),
                format_func=lambda x: display_category(
                    "savings", x
                )
            )

        with col2:

            installment_rate = st.selectbox(
                "Part des mensualités dans le revenu disponible",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "35 % ou plus",
                    2: "25 % à moins de 35 %",
                    3: "20 % à moins de 25 %",
                    4: "Moins de 20 %"
                }[x]
            )

            history = st.selectbox(
                "Historique de crédit",
                list(LABELS["credit_history"]),
                format_func=lambda x: display_category(
                    "credit_history", x
                )
            )

            purpose = st.selectbox(
                "Objet du crédit",
                list(LABELS["purpose"]),
                format_func=lambda x: display_category(
                    "purpose", x
                )
            )

            other_plans = st.selectbox(
                "Autres plans de paiement",
                list(LABELS["other_installment_plans"]),
                format_func=lambda x: display_category(
                    "other_installment_plans", x
                )
            )

        st.divider()

        st.subheader("02 — Informations personnelles")

        col3, col4 = st.columns(2)

        with col3:

            age = st.number_input(
                "Âge du demandeur",
                min_value=18,
                max_value=100,
                value=30
            )

            employment = st.selectbox(
                "Ancienneté professionnelle",
                list(LABELS["employment"]),
                format_func=lambda x: display_category(
                    "employment", x
                )
            )

            personal_status = st.selectbox(
                "Situation personnelle",
                list(LABELS["personal_status"]),
                format_func=lambda x: display_category(
                    "personal_status", x
                )
            )

            job = st.selectbox(
                "Profession",
                list(LABELS["job"]),
                format_func=lambda x: display_category(
                    "job", x
                )
            )

        with col4:

            residence_since = st.selectbox(
                "Ancienneté dans le logement",
                [1, 2, 3, 4],
                format_func=lambda x: f"Catégorie {x}"
            )

            housing = st.selectbox(
                "Situation du logement",
                list(LABELS["housing"]),
                format_func=lambda x: display_category(
                    "housing", x
                )
            )

            dependents = st.selectbox(
                "Nombre de personnes à charge",
                [1, 2]
            )

            existing_credits = st.selectbox(
                "Nombre de crédits existants",
                [1, 2, 3, 4]
            )

        st.divider()

        st.subheader("03 — Informations complémentaires")

        col5, col6 = st.columns(2)

        with col5:

            property_value = st.selectbox(
                "Patrimoine",
                list(LABELS["property"]),
                format_func=lambda x: display_category(
                    "property", x
                )
            )

            other_debtors = st.selectbox(
                "Autres débiteurs",
                list(LABELS["other_debtors"]),
                format_func=lambda x: display_category(
                    "other_debtors", x
                )
            )

        with col6:

            telephone = st.selectbox(
                "Téléphone",
                list(LABELS["telephone"]),
                format_func=lambda x: display_category(
                    "telephone", x
                )
            )

            foreign_worker = st.selectbox(
                "Travailleur étranger",
                list(LABELS["foreign_worker"]),
                format_func=lambda x: display_category(
                    "foreign_worker", x
                )
            )

        submitted = st.form_submit_button(
            "LANCER L'ÉVALUATION",
            use_container_width=True
        )

    if submitted:

        applicant = pd.DataFrame([{

            "checking_account": checking,
            "duration": duration,
            "credit_history": history,
            "purpose": purpose,
            "credit_amount": credit_amount,
            "savings": savings,
            "employment": employment,
            "installment_rate": installment_rate,
            "personal_status": personal_status,
            "other_debtors": other_debtors,
            "residence_since": residence_since,
            "property": property_value,
            "age": age,
            "other_installment_plans": other_plans,
            "housing": housing,
            "existing_credits": existing_credits,
            "job": job,
            "dependents": dependents,
            "telephone": telephone,
            "foreign_worker": foreign_worker

        }], columns=FEATURE_COLUMNS)

        try:

            prediction = model.predict(applicant)[0]

            probability = model.predict_proba(applicant)[0]

            good_probability = probability[good_index] * 100

            bad_probability = 100 - good_probability

            st.divider()

            st.subheader("Résultat de l'évaluation")

            if prediction == 1:

                st.success(
                    "Le modèle classe ce dossier dans la "
                    "catégorie : BON CRÉDIT"
                )

            else:

                st.error(
                    "Le modèle classe ce dossier dans la "
                    "catégorie : CRÉDIT À RISQUE"
                )

            col1, col2 = st.columns(2)

            col1.metric(
                "Probabilité estimée de bon crédit",
                f"{good_probability:.2f}%"
            )

            col2.metric(
                "Probabilité estimée de mauvais crédit",
                f"{bad_probability:.2f}%"
            )

            st.progress(
                int(round(good_probability))
            )

            st.info(
                "Cette estimation correspond à la sortie "
                "probabiliste du modèle. Elle ne constitue "
                "pas une décision bancaire réelle ni une "
                "garantie de remboursement."
            )

        except Exception as error:

            st.error("Une erreur est survenue pendant la prédiction.")

            st.code(str(error))


# =====================================================
# 10. ANALYSE DU MODELE
# =====================================================

elif page == "📊 Analyse du modèle":

    page_header(
        "Analyse du modèle",
        "Évaluation technique du classificateur Random Forest."
    )

    st.subheader("Performances globales")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{accuracy * 100:.2f}%")

    col2.metric("Precision", f"{precision * 100:.2f}%")

    col3.metric("Recall", f"{recall * 100:.2f}%")

    col4.metric("F1-score", f"{f1 * 100:.2f}%")

    col5.metric("ROC-AUC", f"{auc:.3f}")

    st.divider()

    st.subheader("Matrice de confusion")

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "Réel : mauvais crédit",
            "Réel : bon crédit"
        ],
        columns=[
            "Prédit : mauvais crédit",
            "Prédit : bon crédit"
        ]
    )

    st.dataframe(
        matrix_df,
        use_container_width=True
    )

    st.caption(
        "La matrice compare les catégories réellement "
        "observées aux catégories prédites."
    )

    st.divider()

    st.subheader("Comparaison des indicateurs")

    performance = pd.DataFrame({

        "Score": [
            accuracy,
            precision,
            recall,
            f1,
            auc
        ]

    }, index=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ])

    st.bar_chart(performance)

    st.divider()

    st.subheader("Architecture du modèle")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        **Algorithme :** Random Forest

        **Nombre d'arbres :** 200

        **Prétraitement :**
        - StandardScaler
        - OneHotEncoder
        - SimpleImputer

        **Gestion des classes :** Balanced
        """)

    with col2:

        st.markdown("""
        **Dataset :** German Credit

        **Nombre de dossiers :** 1 000

        **Entraînement :** 80 %

        **Test :** 20 %

        **Séparation :** Stratifiée

        **Random state :** 42
        """)

    st.caption(
        "Les performances sont recalculées sur le même "
        "découpage que celui utilisé lors de l'entraînement "
        "dans model.py."
    )


# =====================================================
# 11. A PROPOS
# =====================================================

elif page == "ℹ️ À propos":

    page_header(
        "À propos de Credit Risk AI",
        "Une application intelligente d'analyse du risque de crédit."
    )

    st.markdown("""
    <div class="section-card">

    <h3>Présentation du projet</h3>

    Credit Risk AI est une application développée pour
    analyser le risque de crédit à l'aide de techniques
    d'intelligence artificielle et de Machine Learning.

    Elle utilise un modèle Random Forest pour estimer
    la catégorie de risque associée à un dossier de crédit.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">

    <h3>Technologies utilisées</h3>

    - Python
    - Pandas
    - Scikit-learn
    - Random Forest
    - Streamlit
    - Joblib

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">

    <h3>Objectifs du projet</h3>

    - Faciliter l'analyse des demandes de crédit.
    - Identifier les profils présentant différents niveaux de risque.
    - Présenter les résultats de manière claire.
    - Utiliser le Machine Learning comme outil d'aide à l'analyse.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.caption(
        "CREDIT RISK AI | Machine Learning Project | Version 1.0"
    )
