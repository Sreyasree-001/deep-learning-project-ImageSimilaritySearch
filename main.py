import streamlit as st
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

from auth import init_db, create_user, check_user
from similarity_utils import find_similar_images


st.set_page_config(
    page_title="Image Similarity System",
    layout="wide"
)

init_db()


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.role = None

if "theme" not in st.session_state:
    st.session_state.theme = "Light"


def apply_theme():
    if st.session_state.theme == "Dark":
        st.markdown(
            """
            <style>
            body { background-color: #0e1117; color: white; }
            .stApp { background-color: #0e1117; }
            </style>
            """,
            unsafe_allow_html=True
        )



def login_page():
    st.title("🔐 Login")

    st.markdown("<br>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Login", use_container_width=True):
        role = check_user(username, password)
        if role:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.session_state.role = role
            st.rerun()
        else:
            st.error("Invalid credentials")



def signup_page():
    st.title("📝 Sign Up")

    st.markdown("<br>", unsafe_allow_html=True)

    username = st.text_input("Choose Username")
    password = st.text_input("Choose Password", type="password")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Create Account", use_container_width=True):
        if create_user(username, password):
            st.success("Account created. Please login.")
        else:
            st.error("Username already exists")



def admin_panel():
    st.title("🛠 Admin Panel")

    vectors = np.load("embeddings_20epochs/image_vectors.npy")
    paths = np.load("embeddings_20epochs/image_paths.npy")

    col1, col2 = st.columns(2)
    col1.metric("Total Images", len(paths))
    col2.metric("Embedding Dimension", vectors.shape[1])

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Dataset Preview")

    cols = st.columns(5)
    for col, img_path in zip(cols, paths[:5]):
        col.image(img_path, width=140)

    st.info("Embeddings are precomputed for faster similarity search.")



def dashboard():
    apply_theme()

    st.sidebar.title("Controls")
    st.sidebar.write(f"👤 {st.session_state.user}")
    st.sidebar.write(f"🔑 {st.session_state.role}")

    theme_choice = st.sidebar.radio(
        "Theme",
        ["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1
    )
    st.session_state.theme = theme_choice

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.role = None
        st.rerun()

    if st.session_state.role == "admin":
        if st.sidebar.button("Admin Panel"):
            admin_panel()
            return

    st.title("🔍 Image Similarity Search")
    st.markdown("<hr>", unsafe_allow_html=True)

    uploaded_img = st.file_uploader(
        "Upload a query image",
        type=["jpg", "jpeg", "png"]
    )

    top_k = st.slider("Number of similar images", 3, 10, 5)

    if uploaded_img:
        query_img = Image.open(uploaded_img).convert("RGB")

        col1, col2 = st.columns([1, 2])
        col1.image(query_img, width=250)
        col2.write("Click the button below to find similar images.")

        if col2.button("Find Similar Images"):
            with st.spinner("Searching..."):
                start = time.time()
                results = find_similar_images(query_img, top_k)
                elapsed = time.time() - start

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Results")

            img_cols = st.columns(top_k)
            scores = []

            for col, (path, score) in zip(img_cols, results):
                col.image(path, width=200)
                col.caption(f"{score:.3f}")
                scores.append(score)



if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        login_page()
    with tab2:
        signup_page()
else:
    dashboard()