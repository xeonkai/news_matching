from st_pages import Page, Section, add_page_title, show_pages

show_pages(
    [
        Section(name="ARTICLE INDEXING", icon="📰"),
        Page("app/Instructions.py", "Instructions", "📰"),
        Page("app/Home.py", "Topic Discovery Tool", "🖥️"),
        Page("app/pages/1__Data_Selection_&_Article_Indexing.py", "Data Selection & Article Indexing", "🔎"),
        Page("app/pages/2__Preview_and_Download_Indexed_Data.py", "Download Indexed Data", "📊"),

        Section(name="UPLOAD DATA TO CENTRAL TOOL", icon="📰"),
        Page("app/pages/3__Upload_Data_to_Central_Tool.py", "Upload Data", "🖥️"),

        Section(name="ANALYTICS", icon="📰"),
        Page("app/pages/4__Traction_Analytics.py", "Traction Analytics Interface Demo", "🖥️"),

        Section(name="REFERENCE", icon="📰"),
        Page("app/pages/5__Taxonomy_Explorer.py", "Taxonomy Explorer", "🔮"),
        Page("app/pages/6__Model_Performance.py", "Model Performance", "🔎"),
    ]
)