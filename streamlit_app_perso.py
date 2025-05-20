import streamlit as st
from collections import OrderedDict

st.set_page_config(page_title="DS - Orange - Supply Chain", page_icon="🚀",layout="wide")

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

st.title("Analyse des commentaires clients")

commentaire_defaut='très bonnes expériences avec showroomprivé : sérieux , choix , qualité , prix et rapidité de livraison.Très satisfaite aussi du service client : retours et remboursements .'

# ETOILE
def afficher_etoiles(note: float, max_etoiles: int = 5):
    """
    Affiche une note sous forme d'étoiles remplies et vides.
    
    :param note: Note sur max_etoiles (ex: 3.5 sur 5)
    :param max_etoiles: Nombre maximum d'étoiles (par défaut 5)
    """
    pleine = "⭐"
    vide = "☆"
    
    # Nombre d'étoiles pleines
    nb_pleines = int(note)  

    # Nombre d'étoiles vides
    nb_vides = max_etoiles - nb_pleines

    # Construction de l'affichage
    etoiles = "★" * nb_pleines  # Étoiles pleines
    etoiles += "☆" * nb_vides  # Étoiles vides

    html_code = f"""
    <div style="font-size: 32px; color: gold;">
        {etoiles}
    </div>
    """
    return html_code

from tabs import intro,LGBM_shap,Camembert_captum,LLM,Feature

PAGES = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (Feature.sidebar_name, Feature),
        (LGBM_shap.sidebar_name, LGBM_shap),
        (Camembert_captum.sidebar_name, Camembert_captum),
        (LLM.sidebar_name, LLM)
    ]
)

# Structure des pages
# st.sidebar.title("Sommaire")
# pages=["Présentation du projet","Exploration", "Feature Engineering", "DataVisualisation", "Performance des modèles", "Simulation LGBM + shap", "Simulation Camembert + Captum","Simulation LLM"]
# page=st.sidebar.radio("Aller vers:", pages)
# st.sidebar.divider()

# if page=="Présentation du projet":
#     st.write("Présentation du projet")

# elif page=="Exploration":
#     st.write("Exploration des données")

# elif page=="DataVisualisation":    
#     st.write("Possibilité d'utiliser streamlit-folium")

# elif page=="Simulation LGBM + shap":
    

# elif page=="Simulation Camembert + Captum":
    

# elif page=="Performance des modèles":
#     st.write("Comparaison sur un jeu de xx commentaire de l'acc / aobo de LGBM, Camembert réentrainé, un LLM éventuellement")
#     st.write("avec option pour échantillon stratifié ?")
#     st.write("champ mot de passe pour mettre une clé LLM")

# elif page=="Simulation LLM":

# elif page=="Feature Engineering":
 

def run():
    # st.sidebar.image(
    #     "https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png",
    #     width=200,
    # )
    tab_name = st.sidebar.radio("", list(PAGES.keys()), 0)
    st.sidebar.divider()

    st.sidebar.write("Sébastien S")
    tab = PAGES[tab_name]

    tab.run()

if __name__ == "__main__":
    run()