import streamlit as st

title='Feature Engineering'
sidebar_name = "Feature Engineering"


def run():
    text_button = "Simuler les Feature Engineering"
    commentaire_defaut='très bonnes expériences avec showroomprivé : sérieux , choix , qualité , prix et rapidité de livraison.Très satisfaite aussi du service client : retours et remboursements .'
    
    if "c1" not in st.session_state:
        st.session_state["c1"] = commentaire_defaut

    st.markdown(
        f"""
        Visualiser les transformations apportées sur un commentaire brut de notre jeu de données en le sélectionnant dans la liste déroulante puis en appuyant sur le bouton "{text_button}".
        """
    )
    
    st.write('## Sélection du commentaire')

    # Commentaires proposés dans la liste déroulante
    commentaires = ["Veuillez sélectionner...", commentaire_defaut, "c est des voleur j ais commande des albums photo et jamais recus les codes , conclusion e dans l os , merci voleur prive ,", 
                    "Madame , Monsieur , MA COMMANDE NE M'EST PAS PARVENUE ET JE SUIS EXTREMEMENT MECONTENTE . J'AI FAIT A CET EGARD 2 RECLAMATIONS AUPRES DE VOS SERVICES . MERCI DE FAIRE LE NECESSAIRE",
                    "Vente-privee des voleur à fuir vous faites pas avoir comme moi ServiceClient lamentable 👎 😡😡😡😡😡😡😡😡😡😡😡😡😡😡😡😡 💩 💩 💩 💩"]

    # Initialisation des valeurs par défaut si elles n'existent pas encore
    if "select_value" not in st.session_state:
        st.session_state.select_value = commentaires[1]
    if "free_value" not in st.session_state:
        st.session_state.free_value = ""

    # Fonction de vérification des champs commentaire
    def on_free_text_change():
        if st.session_state.free_value != "":
            st.session_state.select_value = commentaires[0]

    def on_select_change():
        if st.session_state.select_value != commentaires[0]:
           st.session_state.free_value = ""

           
    # liste déroulante de commentaires du jeu de données
    selected_comment = st.selectbox("Choix d'un commentaire extrait de notre jeu de données par menu déroulant", options=commentaires, key="select_value", on_change=on_select_change)

    # zone de saisie libre du commentaire à tester
    free_comment=st.text_input("ou commentaire libre à saisir ici :",key="free_value", on_change=on_free_text_change)
    
    # inputcommentaire = commentaire à analyser --> init par défaut au commentaire de la liste déroulante
    inputcommentaire = selected_comment
    # check sur le champ de saisie
    if free_comment != "":
        inputcommentaire = free_comment

    # Condition pour afficher le message d'alerte au clic sur le bouton
    text_filled = st.session_state.free_value.strip() != ""
    select_chosen = st.session_state.select_value != commentaires[0]
    form_valid = text_filled or select_chosen

    launch = st.button(text_button)

    # Message d'alerte si le formulaire est invalide et bouton cliqué
    if not form_valid and launch:
        st.warning("Veuillez sélectionner ou saisir un commentaire pour pouvoir continuer.")


    # analyse only if button and form_valid
    if (launch and form_valid): 
        st.divider()

        # init du 2ème commentaire pour comparaison
        commentaire_2 = commentaire_defaut
        if inputcommentaire == commentaire_defaut:
            commentaire_2 = commentaires[2]

        # calcul features numériques
        import string
        upper_letters = sum(1 for char in inputcommentaire if char.isupper())
        punct_count = sum(1 for char in inputcommentaire if char in string.punctuation)
        
        # mise en minuscule, on garde le commentaire initial dans inputcommentaire
        commentaire=inputcommentaire.lower()
        commentaire_2=commentaire_2.lower()

        # suppression des chiffres
        import re
        numbers=re.compile('[0-9]+')
        commentaire=numbers.sub('',commentaire)
        commentaire_2=numbers.sub('',commentaire_2)
 
        # suppression des smileys
        import emoji
        commentaire= emoji.demojize(commentaire, language="fr")
        commentaire_2= emoji.demojize(commentaire_2, language="fr")

        # Stemming
        from nltk.stem.snowball import FrenchStemmer
        stemmer=FrenchStemmer()

        mots = commentaire.split()
        mots_stem = [stemmer.stem(mot) for mot in mots]
        commentaire_stem = ' '.join(mots_stem)

        # Lemmingisation avec NLTK
        import nltk
        nltk.download('wordnet')
        nltk.download('stopwords')
        from nltk.stem import WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()
        mots_lemm = [wordnet_lemmatizer.lemmatize(mot) for mot in mots]
        commentaire_lemm = ' '.join(mots_lemm)

        # fonction en cache
        import spacy
        @st.cache_resource
        def spacy_load_sm():
            return spacy.load('fr_core_news_sm')

        @st.cache_resource
        def spacy_load_lg():
            return spacy.load('fr_core_news_lg')
        
        # Lemmatisation avec Spacy
        with st.spinner("Calcul des features..."):
            
            nlp_sm=spacy_load_sm()
            nlp_lg=spacy_load_lg()
            #spacy.load('fr_core_news_lg')

            def lemmatisation_spacy(texte,model_spacy) :
                doc = model_spacy(texte)
                return ' '.join([token.lemma_ for token in doc])
              
            commentaire_spacy_sm=lemmatisation_spacy(commentaire,nlp_sm)    
            commentaire_spacy_lg=lemmatisation_spacy(commentaire,nlp_lg)  

            commentaire_spacy_sm_2=lemmatisation_spacy(commentaire_2,nlp_sm) 
        
        st.markdown("<p style='font-size:16px; color:#1f77b4'>Résultat des prétraitements et de l'extraction des features sur le commentaire sélectionné ci-dessus</p>", unsafe_allow_html=True)
        dict_feature={
            "Commentaire brut": inputcommentaire,
            "Longueur du commentaire": len(inputcommentaire),
            "Nombre de majuscules": upper_letters,
            "Nombre de ponctuations": punct_count,
            "Commentaire sans smileys": commentaire,
            "Commentaire après stemming NLTK": commentaire_stem,
            "Commentaire après lemming NLTK": commentaire_lemm,
            "Commentaire après lemmatisation Spacy (modèle fr_core_news_sm)": commentaire_spacy_sm,
            "Commentaire après lemmatisation Spacy (modèle fr_core_news_lg)": commentaire_spacy_lg
        }
        import pandas as pd
        df_feature=pd.DataFrame(dict_feature.items(),columns=["Etape","Texte"])

        st.dataframe(data=df_feature,hide_index=True,use_container_width=True)  

        st.divider()
        st.write("## Vectorisation (basée sur le commentaire traité avec Spacy fr_core_news_sm)")
        st.markdown(f"<p style='font-size:16px; color:#1f77b4'>A des fins de comparaison, les vectorisations ci-dessous sont présentées pour le commentaire sélectionné plus haut (1ère ligne) et ce deuxième commentaire extrait de notre jeu de données (2ème ligne): \"{commentaire_2}\"</p>", unsafe_allow_html=True)

        # BoW
        from sklearn.feature_extraction.text import CountVectorizer
        from nltk.corpus import stopwords
        
        stop_words=set(stopwords.words('french'))
        stop_words.update(['a','j\'ai','car','a','c\'est','veepee','showroom'])

        import unicodedata

        def normalize(text):
            text = text.lower()
            text = unicodedata.normalize('NFD', text)
            text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])  # enlève les accents
            return text

        # Appliquer la normalisation aux stop words
        stop_words = [normalize(w) for w in stop_words]

        BoW=CountVectorizer(strip_accents='unicode',stop_words=list(stop_words)) # on supprime les accents
        BoW.fit([commentaire_spacy_sm,commentaire_spacy_sm_2])
        result_bow=BoW.transform([commentaire_spacy_sm,commentaire_spacy_sm_2])
        st.write("### BoW - Représentation creuse")
        st.dataframe(pd.DataFrame(result_bow.todense(),columns=BoW.get_feature_names_out()),hide_index=True)

        # TFIDF
        from sklearn.feature_extraction.text import TfidfVectorizer 

        tfidf=TfidfVectorizer(strip_accents='unicode',stop_words=list(stop_words)) # on supprime les accents
        tfidf.fit([commentaire_spacy_sm,commentaire_spacy_sm_2])
        result_tfidf=tfidf.transform([commentaire_spacy_sm,commentaire_spacy_sm_2])
        st.write("### TF-IDF (valeur par défaut pour ngram_range=(1,1)) - Représentation creuse")
        st.dataframe(pd.DataFrame(result_tfidf.todense(),columns=tfidf.get_feature_names_out()),hide_index=True)

        # TFIDF et ngrames
        tfidf=TfidfVectorizer(strip_accents='unicode',stop_words=list(stop_words),ngram_range=(1,2)) # on supprime les accents
        tfidf.fit([commentaire_spacy_sm,commentaire_spacy_sm_2])
        result_tfidf=tfidf.transform([commentaire_spacy_sm,commentaire_spacy_sm_2])
        st.write("### TF-IDF (ngram_range=(1,2))  - Représentation creuse")
        st.dataframe(pd.DataFrame(result_tfidf.todense(),columns=tfidf.get_feature_names_out()),hide_index=True)

        # Tiktoken
        import tiktoken
        tiktoken=tiktoken.get_encoding("cl100k_base")
        tiktoken_tokens = tiktoken.encode(commentaire_spacy_sm)
        tiktoken_tokens_2=tiktoken.encode(commentaire_spacy_sm_2)
        st.write("### Tiktoken - Tokenisation selon un vocabulaire fixe")

        #col1,col2=st.columns(2)
        #col1.write(tiktoken_tokens)
        #col2.write(tiktoken_tokens_2)     

        dict_tiktoken={}
        for i, token in enumerate(tiktoken_tokens):
            dict_tiktoken[token]=tiktoken.decode([token])

        dict_tiktoken_2={}
        for i, token in enumerate(tiktoken_tokens_2):
            dict_tiktoken_2[token]=tiktoken.decode([token])

        #col1,col2=st.columns(2)
        #col1.write(dict_tiktoken)
        #col2.write(dict_tiktoken_2)

        #essai de représentation sous forme de dataframe des dictionnaires tiktoken
        df_1=pd.DataFrame(list(dict_tiktoken.items()), columns=["Vecteur_1", "Token_1"])
        df_2=pd.DataFrame(list(dict_tiktoken_2.items()), columns=["Vecteur_2", "Token_2"])

        st.dataframe(pd.concat((df_1,df_2),axis=1),hide_index=False)

        st.write("### Modèles BERT")

        @st.cache_resource(ttl=86400)
        def load_model(model_path):
            from transformers import AutoModelForSequenceClassification
            return AutoModelForSequenceClassification.from_pretrained(model_path)

        @st.cache_resource(ttl=86400)
        def load_tokenizer(model_path):
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(model_path,use_fast=False)#,local_files_only=True)
        
        # chargement du tokenizer et du modèle
        with st.spinner("Chargement du modèle..."):

            
            import torch

            import os
            if "STREAMLIT_SERVER_RUN_ON_SAVE" in os.environ:
                #st.write("Exécution sur Streamlit Cloud")
                MODE = "cloud"
                model_path="Microbug/camembert-base-reviewfr"
            else:
                #st.write("Exécution locale")
                MODE = "local"
                model_path='./../_camembert/'
            

            # chargement du tokenizer    
            #tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False,local_files_only=True)
            tokenizer=load_tokenizer(model_path)
            # tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)

            # chargement du modèle
            # model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model=load_model(model_path)

        with st.spinner("Tokenisation..."):
            encodings = tokenizer([inputcommentaire,commentaire_2], truncation=True, padding=True, max_length=128, return_tensors="pt")


        # Afficher le token
        import pandas as pd
        st.write("Commentaire après passage dans le tokenizer Bert")
        st.dataframe(pd.DataFrame(encodings.input_ids))
        st.write("Attention mask")
        st.dataframe(pd.DataFrame(encodings.attention_mask))

        # Vectorisation
        model.eval()
        with torch.no_grad():
            outputs2 = model.roberta(**encodings) # la partie roberta donne l'encoding et l'embedding, juste avant la partie classification
            last_hidden_state = outputs2.last_hidden_state  # (1, seq_len, hidden_size)
            cls_vector = last_hidden_state[:, 0, :]
        st.write("Vecteur [CLS] passé à la tête de classification")
        st.dataframe(pd.DataFrame(cls_vector))