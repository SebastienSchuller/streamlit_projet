import streamlit as st
from func import afficher_etoiles

sidebar_name = "Simulation LGBM + shap"


def run():
    st.write('## Saisssez un commentaire à analyser avec le modèle LGBM')

    commentaire_defaut='très bonnes expériences avec showroomprivé : sérieux , choix , qualité , prix et rapidité de livraison.Très satisfaite aussi du service client : retours et remboursements .'

    if "c1" not in st.session_state:
        st.session_state["c1"] = commentaire_defaut

    # zone de saisie du commentaire à tester
    inputcommentaire=st.text_input("Commentaire à analyser:",key="c1")#commentaire_defaut)

    # bouton de validation
    if st.button("Analyser"):
        st.divider()
        comm_length=len(inputcommentaire)
        st.write("Longueur du commentaire:",comm_length, "caractères")

        # mise en minuscule, on garde le commentaire initial dans inputcommentaire
        commentaire=inputcommentaire.lower()

        # suppression des chiffres
        import re
        numbers=re.compile('[0-9]+')
        commentaire=numbers.sub('',commentaire)

        # suppression des smileys
        import emoji
        commentaire= emoji.demojize(commentaire, language="fr")

        with st.spinner("Chargement des librairies..."):
            import spacy
            nlp=spacy.load('fr_core_news_sm')

        def lemmatisation_spacy(texte) :
            doc = nlp(texte)
            return ' '.join([token.lemma_ for token in doc])

        with st.spinner("Lemmatisation.."):        
            commentaire=lemmatisation_spacy(commentaire)

        st.write("Commentaire après lemmatisation Spacy:",commentaire)


        import joblib
        import numpy as np
        import pandas as pd

        # Vectorisation tf-idf: chargement du vocabulaire
        from sklearn.feature_extraction.text import TfidfVectorizer         
        vectorizer=joblib.load("./models/lgbm/tfidf.pkl")
        vector_commentaire=vectorizer.transform([commentaire])

        #st.write("Vecteur après tfidf:",vector_commentaire.shape)

        # min max sur la longueur
        from sklearn.preprocessing import MinMaxScaler
        scaler_length=joblib.load("./models/lgbm/scaler.pkl")
        comm_length=scaler_length.transform(np.array(comm_length).reshape(1,-1))
        #st.write("Vecteur longueur:",comm_length.shape)

        X_pred_vector=pd.DataFrame(np.hstack((vector_commentaire.todense(),comm_length)))
        #st.write("Vecteur d'entrée du modèle avec ajout de la longueur:",X_pred_vector.shape)

        # chargement du modèle et de ses paramètres
        from lightgbm import LGBMClassifier

        @st.cache_resource(ttl=86400)
        def load_model_lgbm():
            return joblib.load("./models/lgbm/lgbm.pkl")

        model=load_model_lgbm() #joblib.load("./models/lgbm/lgbm.pkl")

        y_test=model.predict(X_pred_vector)
        st.write("Le modèle LGBM prédit une note de:",y_test[0],"pour ce commentaire.")
       

        
        # Affichage de la note sous forme d'étoiles
        st.markdown(afficher_etoiles(y_test[0]), unsafe_allow_html=True)

        # Interprétabilité avec shap ?
        with st.spinner("Calcul de l'interprétabilité shap..."):
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values_pipe = explainer.shap_values(X_pred_vector) 

        individu=0

       

        feature_names = vectorizer.get_feature_names_out().tolist() + ['Commentaire_len']

        # pour l'individu 1 et toutes les i classes
        # Boucle pour afficher plusieurs graphiques SHAP 
        import matplotlib.pyplot as plt

        @st.cache_resource(ttl=86400)
        def shap_plot(expected_value,shap_value_ind,X_pred,feature_names,matplotlib):
            st.write(f"### Star: {i+1}")
            fig = plt.figure()
            shap.force_plot(expected_value,shap_value_ind,X_pred,feature_names,matplotlib=True)
            st.pyplot(plt.gcf())

        for i in range(5):
            shap_plot(explainer.expected_value[i],shap_values_pipe[individu,...,i],X_pred_vector,feature_names=feature_names,matplotlib=True)
            # ancienne version sans cache
            # st.write(f"### Star: {i+1}")
            # fig = plt.figure()
            # shap.force_plot(explainer.expected_value[i],shap_values_pipe[individu,...,i],X_pred_vector,feature_names=feature_names,matplotlib=True)
            # st.pyplot(plt.gcf())