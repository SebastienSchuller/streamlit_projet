"""

Config file for Streamlit App

"""

from member import Member


TITLE = "My Awesome App"

TEAM_MEMBERS = [
    Member(
        name="Mariem Abdellatif",
        #linkedin_url="https://www.linkedin.com/in/charlessuttonprofile/",
        #github_url="https://github.com/charlessutton",
    ),
    Member("Valérie Gautier Turbin"),
    Member("Sana Nasri"),
    Member("Sébastien Schuller")
]

PROMOTION = "Promotion Data Scientist Orange - Juin 2025"
