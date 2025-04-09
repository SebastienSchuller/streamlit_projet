import streamlit as st

st.set_page_config(page_title="DS - Orange - Supply Chain", page_icon="🚀")

st.title("Analyse des commentaires clients")

# Structure des pages
st.sidebar.title("Sommaire")
pages=["Présentation du projet","Exploration", "Feature Engineering", "DataVisualisation", "Performance des modèles", "Simulation LGBM + shap", "Simulation Camembert + Captum","Simulation LLM"]
page=st.sidebar.radio("Aller vers:", pages)
st.sidebar.divider()
st.sidebar.write("Sébastien S")

if page=="Présentation du projet":
    st.write("Présentation du projet")

elif page=="Exploration":
    st.write("Exploration des données")

elif page=="DataVisualisation":    
    st.write("Possibilité d'utiliser streamlit-folium")

elif page=="Simulation LGBM + shap":
    st.write('## Saisssez un commentaire à analyser avec le modèle LGBM')

    # zone de saisie du commentaire à tester
    inputcommentaire=st.text_input("Commentaire à analyser:","Super produit !")

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
        model=joblib.load("./models/lgbm/lgbm.pkl")

        y_test=model.predict(X_pred_vector)
        st.write("Le modèle LGBM prédit une note de:",y_test[0],"pour ce commentaire.")
       
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

        for i in range(5):
            st.write(f"### Star: {i+1}")
            fig = plt.figure()
            shap.force_plot(explainer.expected_value[i],shap_values_pipe[individu,...,i],X_pred_vector,feature_names=feature_names,matplotlib=True)
            st.pyplot(plt.gcf())

elif page=="Simulation Camembert + Captum":
    st.write('## Saisssez un commentaire à analyser avec le modèle Camembert')
    # zone de saisie du commentaire à tester
    inputcommentaire=st.text_input("Commentaire à analyser:","Super produit !")
    fenetre_occ_max=st.slider("Taille max de la fenêtre d'occlusion (! au temps de calcul):",1,20,10,1,None,None,"De 1 à ...")
    # bouton de validation
    if st.button("Analyser"):
        st.divider()

        # chargement du tokenizer et du modèle
        with st.spinner("Chargement du modèle..."):

            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            #model_path='./models/camembert/'
            model_path="Microbug/camembert-base-reviewfr"
            
            # chargement du tokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False,local_files_only=True)

            # chargement du modèle
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            

        new_comments = inputcommentaire#[inputcommentaire]
        encodings = tokenizer(new_comments, truncation=True, padding=True, max_length=128, return_tensors="pt")

        # Faire des prédictions
        model.eval()
        with torch.no_grad():
            outputs = model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=1)
            # st.write(predictions)
            st.write("Notation du modèle Camembert réentrainé:",predictions.to("cpu").numpy()[0] + 1)  # Revenir à la notation initiale (1-5)

        # interprétabilité par Occlusion avec captum
        from captum.attr import Occlusion

        def forward_func(inputs):
            inputs=inputs.to(torch.int64) #indispensable pour IntegratedGradient qui envoie un tableau de tensor modifiés, les remettre au bon type
            outputs = model(inputs)
            # print(torch.softmax(outputs.logits, dim=-1))
            return outputs[0] # on ne renvoie que les logits l'objet complet SequenceClassifierOutput
            #return torch.softmax(outputs.logits, dim=-1)[:, 1]  # Probabilité pour la classe positive       
        
        occlusion = Occlusion(forward_func)

        def interpretabilite_occlusion(model,x,y,sliding_window_shapes=(1,),show_progress=True):
            inputs = tokenizer(x, return_tensors="pt", truncation=True, padding=True)
            input_ids = inputs["input_ids"]
            
            #print("Inputs_ids:",input_ids)

            # Calculer les attributions par occlusion
            
            attributions = occlusion.attribute(
            inputs=input_ids,
            sliding_window_shapes=sliding_window_shapes,# (1,),  # Masquer un token à la fois
            baselines=torch.zeros_like(input_ids),  # Utiliser des zéros comme baseline
            target=int(y-1), # Note mises de 0 à 4 pour correspondre aux classes
            show_progress=show_progress #True
            )

            #print("Attributions:",attributions)
            
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
            return tokens,attributions[0].detach().numpy() # list(zip(tokens, attributions[0].detach().numpy()))
        
        # fonctions pour l'affichage coloré
        # Convertir les couleurs RGBA en format hexadécimal
        def rgba_to_hex(rgba):
            r, g, b, _ = rgba  # On ignore l'alpha (transparence)
            return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

        from matplotlib.colors import Normalize
        import matplotlib.pyplot as plt

        def show_texte_color(tokens,attributions):
            
            # Normaliser les attributions pour une échelle de couleurs
            norm = Normalize(vmin=min(attributions), vmax=max(attributions)) # car cmap fonctionne avec des valeurs entre 0 et 1

            # Choisir une colormap (rouge pour négatif, vert pour positif)
            cmap = plt.cm.RdYlGn  # Rouge -> Jaune -> Vert
            
            # Générer les couleurs pour chaque attribution
            colors = [cmap(norm(score)) for score in attributions]
            
            hex_colors = [rgba_to_hex(color) for color in colors]

            # Construire une représentation HTML avec les couleurs
            html_content = ""
            for token, color in zip(tokens, hex_colors):
                html_content += f'<span style="background-color:{color}; padding:2px; margin:1px; border-radius:4px;">{token}</span> '

            return html_content


        # appel de l'occlusion avec fenetre glissante
        st.divider()

        with st.spinner("Calcul de l'Occlusion..."):
            # calcul du nombre de token
            inputs = tokenizer(inputcommentaire, return_tensors="pt", truncation=True, padding=True)

            for s in range(1,min(inputs['input_ids'].shape[1]+1,fenetre_occ_max+1)):
                tokens,attrib=interpretabilite_occlusion(model,inputcommentaire,predictions.numpy()[0] + 1,sliding_window_shapes=(s,),show_progress=False)

                html_content=show_texte_color(tokens[1:-1],attrib[1:-1]) # avec le slicing on retire les tokens de début et fin de phrase (<s> et </s> qui en plus font l'affichage barré)
                st.html(html_content)

elif page=="Performance des modèles":
    st.write("Comparaison sur un jeu de xx commentaire de l'acc / aobo de LGBM, Camembert réentrainé, un LLM éventuellement")
    st.write("avec option pour échantillon stratifié ?")
    st.write("champ mot de passe pour mettre une clé LLM")

elif page=="Simulation LLM":

    if "mistral_api_key" not in st.session_state:
        st.session_state["mistral_api_key"] = ""

    with st.popover("Paramètres LLM"):
        st.markdown("Saisissez ici une clé pour l'API de Mistral AI.")
        mistral_api_key = st.text_input("Mistral AI API Key",type='password',value=st.session_state["mistral_api_key"])
        if mistral_api_key != st.session_state["mistral_api_key"]:
            st.session_state["mistral_api_key"] = mistral_api_key

    st.write('## Saisssez un commentaire à analyser avec le LLM')
    # zone de saisie du commentaire à tester
    inputcommentaire=st.text_input("Commentaire à analyser:","Super produit !")
    # bouton de validation
    if st.button("Analyser"):
        st.divider()
        
    st.write("Approche structured output avec note, mots clés, thème. résumé ?")
    st.write("inclus une écriture de la réponse ?")



elif page=="Feature Engineering":
    st.write("Comparaison Stemming, Lemming NLTK / Spacy (large et grand model) avec un commentaire bien choisie en exemple et la possibilité de taper un commentaire")
    st.write("Montrer aussi la tokenisation/vectorisation BoW, TFIDF (avec la formule et les ngrammes), tiktoken")    

    st.write('## Saissez un commentaire:')

    # zone de saisie du commentaire à tester
    inputcommentaire=st.text_input("Commentaire à analyser:","Super produit !")

    # bouton de validation
    if st.button("Simuler les Feature Engineering"):
        st.divider()

        # mise en minuscule, on garde le commentaire initial dans inputcommentaire
        commentaire=inputcommentaire.lower()

        # suppression des chiffres
        import re
        numbers=re.compile('[0-9]+')
        commentaire=numbers.sub('',commentaire)
 
        # suppression des smileys
        import emoji
        commentaire= emoji.demojize(commentaire, language="fr")

        # Stemming
        from nltk.stem.snowball import FrenchStemmer
        stemmer=FrenchStemmer()

        mots = commentaire.split()
        mots_stem = [stemmer.stem(mot) for mot in mots]
        commentaire_stem = ' '.join(mots_stem)

        # Lemmingisation avec NLTK
        from nltk.stem import WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()
        mots_lemm = [wordnet_lemmatizer.lemmatize(mot) for mot in mots]
        commentaire_lemm = ' '.join(mots_lemm)

        # Lemmatisation avec Spacy
        with st.spinner("Calcul des features..."):
            import spacy
            nlp_sm=spacy.load('fr_core_news_sm')
            nlp_lg=spacy.load('fr_core_news_lg')

            def lemmatisation_spacy(texte,model_spacy) :
                doc = model_spacy(texte)
                return ' '.join([token.lemma_ for token in doc])
              
            commentaire_spacy_sm=lemmatisation_spacy(commentaire,nlp_sm)    
            commentaire_spacy_lg=lemmatisation_spacy(commentaire,nlp_lg)  
        

        dict_feature={
            "Commentaire brut": inputcommentaire,
            "Longueur du commentaire": len(inputcommentaire),
            "Commentaire en minuscule": commentaire,
            "Commentaire sans chiffres": commentaire,
            "Commentaire sans smileys": commentaire,
            "Commentaire après stemming NLTK": commentaire_stem,
            "Commentaire après lemming NLTK": commentaire_lemm,
            "Commentaire après lemmatisation Spacy (modèle fr_core_news_sm)": commentaire_spacy_sm,
            "Commentaire après lemmatisation Spacy (modèle fr_core_news_lg)": commentaire_spacy_lg
        }
        import pandas as pd
        df_feature=pd.DataFrame(dict_feature.items(),columns=["Etape","Texte"])


        st.dataframe(data=df_feature,hide_index=True,use_container_width=True)  