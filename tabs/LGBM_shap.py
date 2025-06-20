import streamlit as st
sidebar_name = "Inférence LightGBM + SHAP"


def run():
    st.write("### Features en entrée du modèle :")
    st.write("- Commentaire (lemmatisation Spacy fr_core_news_sm, vectorisation TF-IDF avec stopwords)")
    st.write("- Longueur du commentaire (normalisée avec MinMaxScaler)")

    # zone de saisie du commentaire à tester
    valeur_defaut = st.session_state.get("c1", "")
    inputcommentaire=st.text_input("Commentaire à analyser",key="free_input_LGBM",value=valeur_defaut)
    # update c1
    st.session_state["c1"] = inputcommentaire

    # Charger les ressources en cache
    @st.cache_resource(ttl=86400, show_spinner=False)
    def load_resources():
        import joblib

        model = joblib.load("./models/lgbm/lgbm.pkl")
        vectorizer = joblib.load("./models/lgbm/tfidf.pkl")
        scaler = joblib.load("./models/lgbm/scaler.pkl")
        return model, vectorizer, scaler

    model, vectorizer, scaler = load_resources()

    # bouton de validation
    if st.button("Analyser"):
        st.session_state["analyse_done"] = True
        st.session_state["commentaire"] = inputcommentaire
        

    do_analysis =  st.session_state.get("lgbm_done") and st.session_state.get("lgbm_analysed_comment") == inputcommentaire  
    
    if (st.session_state.get("analyse_done") or do_analysis):
        from func import afficher_etoiles
        import spacy, re, emoji, numpy as np, pandas as pd
        import shap
        import matplotlib.pyplot as plt

        st.divider()
        commentaire = st.session_state["commentaire"]
        comm_length=len(commentaire)
        st.write("Longueur du commentaire :",comm_length, "caractères")

        # mise en minuscule, on garde le commentaire initial dans inputcommentaire
        commentaire=commentaire.lower()

        # suppression des chiffres
        numbers=re.compile('[0-9]+')
        commentaire=numbers.sub('',commentaire)

        # suppression des smileys
        import emoji
        commentaire= emoji.demojize(commentaire, language="fr")

        @st.cache_resource(ttl=86400, show_spinner=False)
        def load_spacy():
            return spacy.load("fr_core_news_sm")
        
        #    import spacy
        nlp=load_spacy()

        def lemmatisation_spacy(texte) :
            doc = nlp(texte)
            return ' '.join([token.lemma_ for token in doc])
      
        # commentaire=lemmatisation_spacy(commentaire)
        commentaire=lemmatisation_spacy(commentaire)
        st.write("Commentaire après lemmatisation Spacy :",commentaire)

        # Vectorisation TF-IDF: chargement du vocabulaire       
        vector_commentaire=vectorizer.transform([commentaire])

        comm_length=scaler.transform(pd.DataFrame(np.array(comm_length).reshape(1,-1), columns=["Commentaire_len"])) #avec nom de feature pour éviter le warning

        X_pred_vector=pd.DataFrame(np.hstack((vector_commentaire.todense(),comm_length)))

        y_test=model.predict(X_pred_vector)
        st.write("Le modèle LGBM prédit une note de :",y_test[0],"pour ce commentaire.")
       
        # Affichage de la note sous forme d'étoiles
        st.markdown(afficher_etoiles(y_test[0]), unsafe_allow_html=True)

        shap_methods = ["Waterfall", "Force_plot"]
        shap_method = st.selectbox("Interprétabilité de la prédiction via SHAP", shap_methods)
        feature_names = vectorizer.get_feature_names_out().tolist() + ['Commentaire_len']

        # Interprétabilité avec shap
        @st.cache_resource(ttl=86400, show_spinner=False)
        def get_shap_explainer():
            return shap.Explainer(model)
        
        explainer = get_shap_explainer()
        shap_values = explainer(X_pred_vector)
        shap_values.feature_names = feature_names

        i=y_test[0]-1

        from streamlit_shap import st_shap
        if shap_method == shap_methods[0]:
            st_shap(shap.plots.waterfall(shap_values[0][:, i], max_display=10), height=400, width=1000)
        else:
            st_shap(shap.force_plot(
                explainer.expected_value[i],
                shap_values.values[0][:, i],
                X_pred_vector,
                feature_names=feature_names
            ), height=200, width=1000)
    
        # save session state
        st.session_state["lgbm_analysed_comment"] = inputcommentaire
        st.session_state["lgbm_done"] = True
        st.session_state["analyse_done"] = False

