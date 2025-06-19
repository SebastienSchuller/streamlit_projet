import streamlit as st
from PIL import Image

sidebar_name = "Présentation du projet"



def run():
  # clear LightGBM analysis cache
  st.session_state["analyse_done"] = False
  # init default comment used by tabs
  commentaire_defaut='très bonnes expériences avec showroomprivé : sérieux , choix , qualité , prix et rapidité de livraison.Très satisfaite aussi du service client : retours et remboursements .'   
  if "c1" not in st.session_state:
    st.session_state["c1"] = commentaire_defaut
  
  st.markdown("<p style='font-size:24px; color:#1f77b4'>La satisfaction client au coeur des préoccupations</p>", unsafe_allow_html=True)
  
  image = Image.open("assets/supply_chain_img.png")
  col1, col2 = st.columns(2)
  with col1:
    st.image(image, caption="", use_container_width =False)
  with col2:
    st.markdown(
        """
        Le projet « Supply Chain – Satisfaction des clients » s’inscrit dans une démarche de comprendre et d’expliquer la satisfaction client exprimée à l’issue d’un geste d’achat.
Outre l’attribution d’un nombre d’étoiles, qui revient in fine à attribuer une note entière graduée souvent sur 5 niveaux, l’analyse des verbatims clients est une mine précieuse d’informations. Elle est essentielle à la compréhension de la satisfaction client. 
Dans le cadre du projet qui nous a été confié, nous nous sommes concentrés sur la prédiction du nombre d’étoiles attribué par le client sur la seule analyse du commentaire déposé par ce dernier ainsi que sur l'extraction des informations pertinentes depuis ce commentaire expliquant la note.
        """
    )

  st.markdown("<p style='font-size:16px; color:#1f77b4'>Infos utiles</p>", unsafe_allow_html=True)
  st.markdown(
    """
    - Données : remises au début du projet par DataScientest au format .csv (reviews_trust.csv), webscraped sur Trust Pilot et Trusted Shop (données d’accès public).
    - Repo GitHub du projet : https://github.com/DataScientest-Studio/jun24cds_supply_chain
    - Rapport : disponible au format pdf ici (https://github.com/DataScientest-Studio/jun24cds_supply_chain/blob/main/reports/Rapport_jun24cds_supply_chain-jun24_orange_ds_vf.pdf)

    """
  )
  st.markdown("---")
  st.markdown("<p style='font-size:24px; color:#1f77b4'>Les données, en un coup d'oeil</p>", unsafe_allow_html=True)

  options = ["Sélectionner...", "Nuage de mots du champ 'Commentaire'", "Nuage de mots du champ 'reponse'", "Nuage des émojis", "Heatmap des features créées"]
  choix = st.selectbox("Quelle visualisation souhaitez-vous afficher ?", options)

  if choix == options[1]:
    #st.write("Voici la partie sur l'analyse de sentiments...")
    image = Image.open("assets/commentaire_wordcloud.jpg")
    st.image(image, caption="Word Cloud du champ 'Commentaire'", use_container_width =False)

  if choix == options[2]:
    st.write("Le champ 'reponse' a un taux de valeurs manquantes de 57,32%. Le nuage a été généré à partir des champs non vides.")
    image = Image.open("assets/reponse_wordcloud.jpg")
    st.image(image, caption="Word Cloud du champ 'reponse'", use_container_width =False)

  if choix == options[3]:
    st.write("L'analyse des commentaires en français fait apparaître une fréquence très relative avec 0.98% des commentaires porteurs d'émojis. Les commentaires notés ⭐☆☆☆☆ et ⭐⭐⭐⭐⭐ sont les plus porteurs d'émojis dans cet ordre.")
    image = Image.open("assets/emojis_cloud.jpg")
    st.image(image, caption="Cloud des émojis rencontrés, toutes notes confondues", use_container_width =False) 
    col1, col2 = st.columns(2)
    with col1:
      image = Image.open("assets/emojis_cloud_star_1.png")
      st.image(image, caption="Cloud des émojis pour les commentaires notés ⭐☆☆☆☆", use_container_width =False)
    with col2: 
      image = Image.open("assets/emojis_cloud_star_5.png")
      st.image(image, caption="Cloud des émojis pour les commentaires notés ⭐⭐⭐⭐⭐", use_container_width =False)
  
  if choix == options[4]:
    st.markdown(
    """
    Des features caractéristiques directes du commentaire brut non nettoyé présentent des coefficients de corrélation élevés avec la variable cible à prédire (la note) :
    - la longueur du commentaire
    - le nombre de majuscules dans le commentaire
    - le nombre de ponctuations dans le commentaire
    - le nombre d'émojis
    """
    )
    image = Image.open("assets/heatmap_features_numeriques.png")

    st.image(image, caption="Corrélations observées entre la note et les features numériques du commentaire brut", use_container_width =True)
    


    
