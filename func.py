# ETOILE
def afficher_etoiles(note: float, max_etoiles: int = 5):
    """
    Affiche une note sous forme d'étoiles remplies et vides.
    
    :param note: Note sur max_etoiles (ex: 3.5 sur 5)
    :param max_etoiles: Nombre maximum d'étoiles (par défaut 5)
    """
    pleine = "⭐"
    vide = "☆"
    
    # Nombre d'étoiles pleines
    nb_pleines = int(note)  

    # Nombre d'étoiles vides
    nb_vides = max_etoiles - nb_pleines

    # Construction de l'affichage
    etoiles = "★" * nb_pleines  # Étoiles pleines
    etoiles += "☆" * nb_vides  # Étoiles vides

    html_code = f"""
    <div style="font-size: 32px; color: gold;">
        {etoiles}
    </div>
    """
    return html_code