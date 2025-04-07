import streamlit as st

st.title("Analyse des commentaires clients")

# Structure des pages
st.sidebar.title("Sommaire")
pages=["Présentation","Exploration", "Feature Engineering", "DataVisualisation", "Simulation"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.markdown("---")
st.sidebar.write("Sébastien S")

if page=="Présentation":
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
        st.markdown("---")
        st.write("Commentaire:")
        st.write(inputcommentaire)
        st.write("Longueur:")
        st.write(len(inputcommentaire))