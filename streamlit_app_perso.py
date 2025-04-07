import streamlit as st

st.title("Analyse des commentaires clients")

# Structure des pages
st.sidebar.title("Sommaire")
pages=["Présentation du projet","Exploration", "Feature Engineering", "DataVisualisation", "Simulation"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.divider()
st.sidebar.write("Sébastien S")

if page=="Présentation du projet":
    st.write("Présentation du projet")

elif page=="Exploration":
    st.write("Exploration des données")

elif page=="DataVisualisation":    
    st.write ("Datavisualisation")

elif page=="Simulation":
    st.write('Saisssez un commentaire à analyser avec le modèle')

    # zone de saisie du commentaire à tester
    inputcommentaire=st.text_input("Commentaire à analyser:","Super produit !")

    # bouton de validation
    if st.button("Analyser"):
        st.divider()
        st.write("Commentaire:")
        st.write(inputcommentaire)
        st.write("Longueur du commentaire:",len(inputcommentaire))

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

        st.write("Commentaire avec lemmatisation spacy:")
        st.write(commentaire)

        # à ce stade le commentaire est pré-processer à l'identique

        #stopwords
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        stop_words=set(stopwords.words('french'))
        stop_words.update(['a','j\'ai','car','a','c\'est','veepee','showroom'])
        
        # Vectorisation tf-idf: chargement du vocabulaire
        from sklearn.feature_extraction.text import TfidfVectorizer 
        vectorizer=TfidfVectorizer(strip_accents='unicode',stop_words=list(stop_words)) # on supprime les accents
        
        # todo: charger le vocabulaire TODO
        # appliquer
        vectorizer.transform()

        # min max sur la longueur
        from sklearn.preprocessing import MinMaxScaler
        scaler_length=MinMaxScaler()
        # charger min et max TODO
        # appliquer
        scaler_length.transform()

        # concaténation tfidf et longueur du commentaire


        # chargement du modèle et de ses paramètres

        # processing de l'exemple