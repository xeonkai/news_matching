from st_pages import Page, add_page_title, show_pages

"## Please click on relevant tabs in the sidebar to continue."

show_pages(
    [
        # Section(name="ARTICLE INDEXING", icon="📰"),
        Page("app/Home.py", "Upload", "🖥️"),
        Page(
            "app/pages/1__Data_Selection_&_Article_Indexing.py",
            "Select & Label",
            "🔎",
        ),
        Page("app/pages/2__Summary_Metrics.py", "Aggregation", "🖥️"),
        # Section(name="ANALYTICS", icon="📰"),
        Page("app/pages/4__Traction_Analytics.py", "Traction Analytics", "🖥️"),
        # Section(name="REFERENCE", icon="📰"),
        # Page("app/pages/6__Model_Performance.py", "Model Performance", "🔎"),
    ]
)
add_page_title(layout="wide")
