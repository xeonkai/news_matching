import streamlit as st


def align_text(text, alignment):
    """Function to align text to column width. Allowed alignments: "justify", "center", "right", "left"
    Args:
        text (str): Text to be aligned
        alignment (str): Alignment of text

    Returns:
        str: Aligned text
    """

    return st.markdown(
        f'<div style="text-align: {alignment};">{text}</div>', unsafe_allow_html=True
    )


def horizontal_line():
    """Insert Horizontal Line
    Returns:
        str: Horizontal Line
    """

    return st.markdown("""---""")
