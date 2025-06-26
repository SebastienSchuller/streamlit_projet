import streamlit as st
import config
from collections import OrderedDict


def show_header(tab):
    st.markdown(f"<h2 style='color:#1f77b4'>Analyse des Avis Clients - {tab}</h2>", unsafe_allow_html=True)
    st.markdown("---")

def show_footer():
    st.markdown("<hr><p style='text-align:center; font-size:12px; color:gray;'>Projet jun24cds_supply_chain</p>", unsafe_allow_html=True)



st.set_page_config(page_title="jun24cds_supply_chain", page_icon="assets/icone-star.png",layout="wide")

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)



commentaire_defaut='très bonnes expériences avec showroomprivé : sérieux , choix , qualité , prix et rapidité de livraison.Très satisfaite aussi du service client : retours et remboursements .'

from tabs import intro,LGBM_shap,Camembert_captum,LLM,Feature
# import de camembert prend du temps

PAGES = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (Feature.sidebar_name, Feature),
        (LGBM_shap.sidebar_name, LGBM_shap),
        (Camembert_captum.sidebar_name, Camembert_captum),
        (LLM.sidebar_name, LLM)
    ]
)

def run():

    import base64

    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    image_path = "assets/DataScientest-TrustPilot.jpg"
    img_base64 = get_base64_of_bin_file(image_path)

    st.sidebar.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/jpeg;base64,{img_base64}" width="200"/>
        <p style="font-size: 10px; color: grey;">DataScientest - Source : TrustPilot (13 juin 2025)</p>
    </div>
    """,
    unsafe_allow_html=True
    )

    st.sidebar.title("Sommaire")
    tab_name = st.sidebar.radio("Navigation", list(PAGES.keys()), 0)
    st.sidebar.divider()

    st.sidebar.markdown("<h3 style='color:#1f77b4;'>Membres de l'équipe</h3>",
    unsafe_allow_html=True)
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = PAGES[tab_name]
    show_header(tab_name)
    tab.run()
    show_footer()

if __name__ == "__main__":
    run()