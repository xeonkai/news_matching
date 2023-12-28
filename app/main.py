from st_pages import Page, Section, add_page_title, show_pages

"## Please click on Topic Discovery Tool in the sidebar to continue."

show_pages(
    [
        Section(name="ARTICLE INDEXING", icon="📰"),
        # TODO: Change name to Upload raw data
        Page("app/Home.py", "Upload Data", "🖥️"),
        Page("app/pages/1__Data_Selection_&_Article_Indexing.py", "Data Selection & Article Indexing", "🔎"),

        # Section(name="UPLOAD DATA TO CENTRAL TOOL", icon="📰"),
        # Page("app/pages/3__Upload_Data_to_Central_Tool.py", "Upload Data", "🖥️"),

        # Section(name="ANALYTICS", icon="📰"),
        # Page("app/pages/4__Traction_Analytics.py", "Traction Analytics Interface Demo", "🖥️"),

        # Section(name="REFERENCE", icon="📰"),
        # Page("app/pages/5__Taxonomy_Explorer.py", "Taxonomy Explorer", "🔮"),
        # Page("app/pages/6__Model_Performance.py", "Model Performance", "🔎"),
    ]
)
add_page_title(layout="wide")
