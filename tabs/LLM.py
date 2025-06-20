import streamlit as st
from func import afficher_etoiles
import os 

sidebar_name = "Inférence LLM Mistral AI"


def run():
    commentaire_defaut='très bonnes expériences avec showroomprivé : sérieux , choix , qualité , prix et rapidité de livraison.Très satisfaite aussi du service client : retours et remboursements .'

    if "mistral_api_key" not in st.session_state:
        st.session_state["mistral_api_key"] = ""

    if "proxy_config" not in st.session_state:
        st.session_state["proxy_config"] = ""
        
    with st.popover("Paramètres LLM"):
        st.markdown("Saisissez ici une clé pour l'API de Mistral AI.")
        mistral_api_key = st.text_input("Mistral AI API Key",type='password',value=st.session_state["mistral_api_key"])
        if mistral_api_key != st.session_state["mistral_api_key"]:
            st.session_state["mistral_api_key"] = mistral_api_key

        st.markdown("Saisissez ici une configuration de proxy si nécessaire.")
        proxy_config = st.text_input("Proxy configuration",value=st.session_state["proxy_config"])
        if proxy_config != st.session_state["proxy_config"]:
            st.session_state["proxy_config"] = proxy_config

        st.markdown("Choisissez un modèle")
        models=['mistral-small-2503','magistral-small-2506','mistral-large-latest']
        selected_models = st.selectbox("Modèle", options=models, key="select_model")

    st.markdown("<p style='font-size:0.875rem; font-weight: bold; margin-top:10px; margin-bottom:-50px'>Commentaire à analyser avec le LLM</p>", unsafe_allow_html=True)
    # zone de saisie du commentaire à tester
    valeur_defaut = st.session_state.get("c1", "")
    inputcommentaire=st.text_input("Commentaire",key="free_input_LLM",value=valeur_defaut,label_visibility='hidden')
    # update c1
    st.session_state["c1"] = inputcommentaire

    st.markdown("<p style='font-size:0.875rem; font-weight: bold; margin-top:10px; margin-bottom:-50px'>Prompt pour le LLM</p>", unsafe_allow_html=True)
    prompt=st.text_area("Prompt",value="Analyse le commentaire suivant et donne une note de 1 à 5 étoiles. Explique ta note et donne des mots clés associés au commentaire. (5 au maximum)",height=68,label_visibility='hidden')         
    
    from pydantic import BaseModel, Field, create_model
    from typing import List, get_args, get_origin, get_type_hints

    class Eval_commentaire(BaseModel):
        '''Commentaire évalué avec ton et mots clés'''
        star: int = Field(description="Note du commentaire entre 1 et 5", ge=1, le=5)
        ton: str = Field(description="Ton du message", list=["positif", "négatif", "neutre"])
        keywords: List[str] = Field(..., description="Liste de mots clés associés au commentaire (5 mots maximum)")
        topic: str = Field(description="Sujet du commentaire")

    st.markdown("<p style='font-size:0.875rem; font-weight: bold; margin-top:10px; margin-bottom:-20px'>Champs du Structured Output</p>", unsafe_allow_html=True)

    # Liste des champs disponibles    
    options = ["star", "ton", "keywords", "topic"]
    cols = st.columns([0.1,0.1,0.1,0.1,0.6])  # Crée une colonne par option

    selected_fields=[]
    for i,option in enumerate(options):
        if cols[i].checkbox(option, key=option,value=True):
            selected_fields.append(option)


    st.markdown("<p style='font-size:0.875rem; font-weight: bold; margin-top:10px; margin-bottom:-20px'>Choix de la température</p>", unsafe_allow_html=True)
    #st.write("## Température") 
    temperature = st.slider(
        'Température',
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        label_visibility='hidden'
    )
    
    # bouton de validation
    if st.button("Analyser et répondre au commentaire"):
        if mistral_api_key=="":
            st.error("Veuillez saisir une clé API Mistral AI dans le popover en haut à gauche de l'écran.")
        else:

            # gestion du proxy
            if proxy_config=="":
                # pas de proxy
                if "HTTPS_PROXY" in os.environ:
                    del os.environ["HTTPS_PROXY"]
                if "HTTP_PROXY" in os.environ:
                    del os.environ["HTTP_PROXY"]
                if "GRPC_PROXY" in os.environ:
                    del os.environ["GRPC_PROXY"]          
            else:
                # proxy
                os.environ["HTTPS_PROXY"] = proxy_config
                os.environ["HTTP_PROXY"] = proxy_config
                os.environ["GRPC_PROXY"] = proxy_config

            st.divider()

            st.write("## Utilisation d'un modèle Mistral AI avec une structured output")
            st.write("Modèle : ",selected_models, " Température: ",str(temperature))

            original_fields = Eval_commentaire.model_fields
            original_types = get_type_hints(Eval_commentaire)


            dynamic_fields = {
            field: (
                original_types[field],
                original_fields[field].default if original_fields[field].default is not None else ...
            )
            for field in selected_fields
         }

            # Crée un nouveau modèle basé sur la sélection
            CustomModel = create_model("CustomEval", **dynamic_fields)

            from langchain_mistralai import ChatMistralAI
            from langchain_core.prompts import ChatPromptTemplate   
            from langchain.prompts import FewShotPromptTemplate
            from langchain.prompts.prompt import PromptTemplate

            # LLM with function call
            llm = ChatMistralAI(model=selected_models,api_key=mistral_api_key,temperature=temperature)
            structured_llm_evaluateur = llm.with_structured_output(CustomModel)#(Eval_commentaire)
            
            eval_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                ("human", "{query}"),
            ]
            )

            retrieval_grader = eval_prompt | structured_llm_evaluateur

            def eval(commentaire):
                question="Evalue ce commentaire : "+commentaire
                docs = retrieval_grader.invoke({"query": question})
                return docs
            
            retour=eval(inputcommentaire)
            st.write("## Résultat de l'évaluation du LLM :") 
            # st.write(retour)

            for k in retour.model_dump().keys():
                if k=="star":
                    st.write("### Note du commentaire :",retour.star)
                    st.markdown(afficher_etoiles(retour.star), unsafe_allow_html=True)
                elif k=="ton":
                    st.write("### Ton du commentaire :",retour.ton)  
                elif k=="keywords":
                    st.write("### Mots clés associés au commentaire :",retour.keywords)
                elif k=="topic":
                    st.write("### Sujet du commentaire :",retour.topic)

            #### début partie réponse au commentaire
            st.write("## Réponse au commentaire avec une approche few shot example :")
            auto_examples=[{'input': 'Nom client: nan Commentaire:Je ne recommande pas Showroomprivé en '
                        'Belgique . Service client : 0 ! En fonction des marques , il '
                        'arrive que les produits arrivent abîmés , que des erreurs sur le '
                        'produit soient faites . Surtout consultez bien leurs conditions '
                        'générales avant d ’ acheter , elles mettent à votre charge les '
                        'frais de retour de votre colis , notamment si le produit ne vous '
                        'plaît pas par exemple et aussi dans d ’ autres cas .',
                'output': 'Bonjour , Je suis sincèrement désolé de lire votre ressenti quant '
                            'aux anomalies relatives à vos commande . Je vous présente mes '
                            'excuses au nom de shwroomprive.com.Je vous informe avoir pris en '
                            'charge votre réclamation et vous confirme que vous serez '
                            "recontactée afin de m'entretenir avec vous de vive voix.Ayoub"},
                {'input': 'Nom client: Vinciane Denne Commentaire:Je commande des capsules de '
                        'cafe Dolce Gusto au prix de 20,66€/ boite + fdp . Dans mon panier '
                        "le total est juste . J'effectue le payement sans trop me poser de "
                        "questions puis je vais directement voir ce que j'ai payé car "
                        "quelque chose me semble louche . Je constate que j'ai payé "
                        "23,69€/boite . Je fais directement un mail auquel j'ai eu réponse "
                        'dans les 24h pour me dire que le prix est bien de 23,69 mais sur '
                        'le site il est toujours affichés 20,66€ . Je ne trouve pas ca '
                        "normal d'afficher un prix et de pas le respecter.Normalement prix "
                        'affichés = prix a payé même si ils se sont trompé .',
                'output': 'Bonjour , Je fais suite à votre message concernant les prix de '
                            'vos capsules de café proposés sur notre site.Je vous informe '
                            'avoir pris en charge votre réclamation et vous confirme que vous '
                            'serez recontactée afin de faire suite à votre requête.Ayoub'},
                {'input': "Nom client: nan Commentaire:Le vêtement m ' a énormément déçu . Le "
                        "tissu ne correspond pas à ce que j'espérais .",
                'output': "Bonjour , Je suis sincèrement navré d'apprendre que votre "
                            'commande ne vous aient pas apportée entière satisfaction . Je '
                            'vous présente mes excuses pour ce désagrément . Je vous informe '
                            'avoir pris en charge votre réclamation et vous confirme que vous '
                            "serez recontactée aujourd'hui afin de m'entretenir avec vous de "
                            'vive voix sur ce sujet . Ayoub'},
                {'input': 'Nom client: nan Commentaire:Vêtements de très mauvaises qualités '
                        '... dommage',
                'output': "Bonjour , Je suis sincèrement navré d'apprendre que vos vêtements "
                            'ne vous aient pas apportée entière satisfaction . Je vous '
                            'présente mes excuses pour ce désagrément . Je vous informe avoir '
                            'pris en charge votre réclamation et vous confirme que vous serez '
                            "recontactée aujourd'hui afin de m'entretenir avec vous de vive "
                            'voix sur ce sujet . Ayoub'},
                {'input': 'Nom client: Jacqueline L . Commentaire:Marchandise conforme à la '
                        'commande livraison plus longue que prévu',
                'output': 'Bonjour , Un grand merci pour votre avis ! Bonne journée de la '
                            "part de toute l'équipe.Ayoub"}]
            
            with st.expander("Visualiser les exemples de réponse fournies au LLM"):
                st.dataframe(auto_examples)
            
            # Création de la classe de sortie structurée
            class Reponse_commentaire(BaseModel):
                '''Réponse à un commentaire'''
                reponse: str = Field(description="Réponse au commentaire")

            # Prompt
            system = """Tu es un professionnel du service client après-vente, qui analyse et répond à des commentaires laissés par des clients suite à une commande.\n
                Tu t'inspires pour les réponses des exemples fournis le plus possible.       
                """
   

            structured_llm_evaluateur_with_example = llm.with_structured_output(Reponse_commentaire)

            examples=auto_examples

            # Template pour afficher chaque exemple
            example_template = """
            Input: {input}
            Output: {output}
            """

            prompt_template = FewShotPromptTemplate(
                examples=examples,
                example_prompt=PromptTemplate(input_variables=["input", "output"], template=example_template),
                prefix=system,
                suffix="Input: {input}\nOutput:",
                input_variables=["input"]
            )

            retrieval_grader_with_example = prompt_template | structured_llm_evaluateur_with_example

            def reponse_with_example(commentaire,prenom):
                question="Réponds à ce commentaire : "+ commentaire + ". Nom :" + str(prenom)
                docs = retrieval_grader_with_example.invoke({"input": question})
                return docs
            
            # sleep pour gérer les limitations de l'API free Mistral AI
            import time
            with st.spinner("Attente 1s pour le LLM..."):
                time.sleep(1)
            retour=reponse_with_example(inputcommentaire,"")
            st.write("Prompt : Tu es un professionnel du service client après-vente, qui analyse et répond à des commentaires laissés par des clients suite à une commande.\n Tu t'inspires pour les réponses des exemples fournis le plus possible.")
            st.write("Réponse générée :")
            st.info(retour.reponse)

            ## Réponse sans fewshot ?
            st.write("## Réponse au commentaire sans l'approche few shot example :")
            st.write("Prompt : Tu es un professionnel du service client après-vente, qui analyse et répond à des commentaires laissés par des clients suite à une commande.")
            structured_llm_without_example=llm.with_structured_output(Reponse_commentaire)
            # Prompt
            system = """Tu es un professionnel du service client après-vente, qui analyse et répond à des commentaires laissés par des clients suite à une commande.\n
                """

            eval_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    ("human", "{input}"),
                ]
            )

            retrieval_grader_without_example = eval_prompt | structured_llm_without_example

            def reponse_without_example(commentaire,prenom):
                question="Répond à ce commentaire : "+ commentaire + ". Nom :" + str(prenom)
                docs = retrieval_grader_without_example.invoke({"input": question})
                return docs
            with st.spinner("Attente 1s pour le LLM..."):
                time.sleep(1)
            retour=reponse_without_example(inputcommentaire,"")

            st.write("Réponse générée :")
            st.info(retour.reponse)