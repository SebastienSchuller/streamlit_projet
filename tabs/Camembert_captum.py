import streamlit as st
from func import afficher_etoiles
#from transformers import AutoModelForSequenceClassification, AutoTokenizer

sidebar_name = "Inférence CamemBERT + Captum"

@st.cache_resource(ttl=86400,show_spinner=False)
def load_model(model_path):
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(model_path)

@st.cache_resource(ttl=86400,show_spinner=False)
def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path,use_fast=False)#,local_files_only=True)


def run():
   
    valeur_defaut = st.session_state.get("c1", "")

    # zone de saisie du commentaire à tester
    inputcommentaire=st.text_input("Commentaire à analyser",key="free_input_BERT",value= valeur_defaut)
    # update c1
    st.session_state["c1"] = inputcommentaire
    fenetre_occ_max=st.slider("Taille max de la fenêtre d'occlusion (! au temps de calcul) :",1,20,10,1,None,None,"De 1 à ...")

    do_analysis =  st.session_state.get("bert_done") and st.session_state.get("bert_analysed_comment") == inputcommentaire

    # bouton de validation
    if (st.button("Analyser") or do_analysis):
        st.divider()

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

        new_comments = inputcommentaire#[inputcommentaire]
        
        with st.spinner("Tokenisation..."):
            encodings = tokenizer(new_comments, truncation=True, padding=True, max_length=128, return_tensors="pt")


        # Faire des prédictions
        with st.spinner("Evaluation..."):
            model.eval()
            with torch.no_grad():
                outputs = model(**encodings)
                predictions = torch.argmax(outputs.logits, dim=1)
                # st.write(predictions)
                st.write("Notation du modèle CamemBERT réentrainé :",predictions.to("cpu").numpy()[0] + 1)  # Revenir à la notation initiale (1-5)
                st.markdown(afficher_etoiles(predictions.to("cpu").numpy()[0] + 1), unsafe_allow_html=True)

        
        # Interprétabilité par SHAP
        st.divider()
        st.markdown("<p style='font-size:18px; color:#1f77b4'>Interprétabilité avec SHAP</p>", unsafe_allow_html=True)


        import shap
        import numpy as np
        
        def predict(texts):
            # Correction : si numpy array, convertir en liste de str
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            if isinstance(texts, str):
                texts = [texts]
            # Vérification supplémentaire (optionnelle)
            assert isinstance(texts, list) and all(isinstance(x, str) for x in texts), f"Entrée inattendue : {type(texts)} / {type(texts[0])}"
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            return probs.detach().cpu().numpy()
        
        custom_labels = ["⭐☆☆☆☆", "⭐⭐☆☆☆", "⭐⭐⭐☆☆", "⭐⭐⭐⭐☆", "⭐⭐⭐⭐⭐"]
        # Créez un masker qui segmente sur les espaces
        masker = shap.maskers.Text(r"\s")

        # Interprétabilité avec shap
        @st.cache_resource(ttl=86400, show_spinner=False)
        def get_shap_explainer_values(comment):
            explainer=shap.Explainer(predict, masker)
            shap_values = explainer([comment])
            return shap_values


        with st.spinner("Calcul de l'interprétabilité avec SHAP..."):
            # Utilisez ce masker dans l'explainer

            shap_values=  get_shap_explainer_values(new_comments)
            shap_values.output_names = custom_labels
            
            from streamlit_shap import st_shap
            st_shap(shap.plots.text(shap_values[0]), height=200)

        
        # interprétabilité par Occlusion avec captum
        from captum.attr import Occlusion

        def forward_func(inputs):
            inputs=inputs.to(torch.int64) #indispensable pour IntegratedGradient qui envoie un tableau de tensor modifiés, les remettre au bon type
            outputs = model(inputs)
            # print(torch.softmax(outputs.logits, dim=-1))
            return outputs[0] # on ne renvoie que les logits l'objet complet SequenceClassifierOutput
            #return torch.softmax(outputs.logits, dim=-1)[:, 1]  # Probabilité pour la classe positive       
        
        occlusion = Occlusion(forward_func)

        @st.cache_resource(ttl=86400,show_spinner=False)        
        def interpretabilite_occlusion(_model,x,y,sliding_window_shapes=(1,),show_progress=True):
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
                token=token.replace("\u2581","") # suppression du caractère de séparation de BERT
                html_content += f'<span style="background-color:{color}; padding:2px; margin:1px; border-radius:4px;">{token}</span> '

            return html_content


        # appel de l'occlusion avec fenetre glissante
        st.divider()

        with st.spinner("Calcul de l'Occlusion..."):
            # calcul du nombre de token
            inputs = tokenizer(inputcommentaire, return_tensors="pt", truncation=True, padding=True)

            #st.write(f"Interprétabilité Captum par occlusion, taille de fenêtre = {fenetre_occ_max}")
            st.markdown(f"<p style='font-size:18px; color:#1f77b4'>Interprétabilité Captum par occlusion, taille de fenêtre = {fenetre_occ_max}</p>", unsafe_allow_html=True)

            for s in range(1,min(inputs['input_ids'].shape[1]+1,fenetre_occ_max+1)):
                tokens,attrib=interpretabilite_occlusion(model,inputcommentaire,predictions.numpy()[0] + 1,sliding_window_shapes=(s,),show_progress=False)

                html_content=show_texte_color(tokens[1:-1],attrib[1:-1]) # avec le slicing on retire les tokens de début et fin de phrase (<s> et </s> qui en plus font l'affichage barré)
                st.html(html_content)
        
        # save session state
        st.session_state["bert_analysed_comment"] = inputcommentaire
        st.session_state["bert_done"] = True