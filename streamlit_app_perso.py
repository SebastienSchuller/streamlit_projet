import streamlit as st
from collections import OrderedDict

st.set_page_config(page_title="DS - Orange - Supply Chain", page_icon="🚀",layout="wide")

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

st.title("Analyse des commentaires clients")

commentaire_defaut='très bonnes expériences avec showroomprivé : sérieux , choix , qualité , prix et rapidité de livraison.Très satisfaite aussi du service client : retours et remboursements .'

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

# pages à ajouter: =["Présentation du projet","Exploration", "Feature Engineering", "DataVisualisation", "Performance des modèles", "Simulation LGBM + shap", "Simulation Camembert + Captum","Simulation LLM"]

def run():
    # logo orange et DS ?
    # st.sidebar.image(
    #     "https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png",
    #     width=200,
    #)
    tab_name = st.sidebar.radio("Menu", list(PAGES.keys()), 0)
    st.sidebar.divider()

    st.sidebar.write("Sébastien S")
    tab = PAGES[tab_name]

    tab.run()

if __name__ == "__main__":
    run()