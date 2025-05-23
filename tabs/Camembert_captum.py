import streamlit as st
from func import afficher_etoiles
#from transformers import AutoModelForSequenceClassification, AutoTokenizer

sidebar_name = "Simulation Camembert + Captum"

@st.cache_resource(ttl=86400)
def load_model(model_path):
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(model_path)

@st.cache_resource(ttl=86400)
def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path,use_fast=False)#,local_files_only=True)


def run():
    st.write('## Sasissez un commentaire à analyser avec le modèle Camembert')
    commentaire_defaut='très bonnes expériences avec showroomprivé : sérieux , choix , qualité , prix et rapidité de livraison.Très satisfaite aussi du service client : retours et remboursements .'

    if "c1" not in st.session_state:
        st.session_state["c1"] = commentaire_defaut

    # zone de saisie du commentaire à tester
    inputcommentaire=st.text_input("Commentaire à analyser:",key="c1")#commentaire_defaut)
    fenetre_occ_max=st.slider("Taille max de la fenêtre d'occlusion (! au temps de calcul):",1,20,3,1,None,None,"De 1 à ...")
    # bouton de validation
    if st.button("Analyser"):
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
                st.write("Notation du modèle Camembert réentrainé:",predictions.to("cpu").numpy()[0] + 1)  # Revenir à la notation initiale (1-5)

                st.markdown(afficher_etoiles(predictions.to("cpu").numpy()[0] + 1), unsafe_allow_html=True)

        # interprétabilité par Occlusion avec captum
        from captum.attr import Occlusion

        def forward_func(inputs):
            inputs=inputs.to(torch.int64) #indispensable pour IntegratedGradient qui envoie un tableau de tensor modifiés, les remettre au bon type
            outputs = model(inputs)
            # print(torch.softmax(outputs.logits, dim=-1))
            return outputs[0] # on ne renvoie que les logits l'objet complet SequenceClassifierOutput
            #return torch.softmax(outputs.logits, dim=-1)[:, 1]  # Probabilité pour la classe positive       
        
        occlusion = Occlusion(forward_func)

        @st.cache_resource(ttl=86400)        
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

            for s in range(1,min(inputs['input_ids'].shape[1]+1,fenetre_occ_max+1)):
                tokens,attrib=interpretabilite_occlusion(model,inputcommentaire,predictions.numpy()[0] + 1,sliding_window_shapes=(s,),show_progress=False)

                html_content=show_texte_color(tokens[1:-1],attrib[1:-1]) # avec le slicing on retire les tokens de début et fin de phrase (<s> et </s> qui en plus font l'affichage barré)
                st.html(html_content)