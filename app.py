import streamlit as st
import pandas as pd

# 1. Membuat Judul Utama Aplikasi di Browser
st.title("🎬 Movie Recommender System")
st.caption("Final Project Machine Learning - Kevin Richie & Johannes Simatupang")

# 2. Membuat Layout Sidebar untuk Input Pengguna
st.sidebar.header("Input Parameter")

# Input Angka untuk User ID
user_id = st.sidebar.number_input("Masukkan User ID:", min_value=1, max_value=610, value=1)

# Pilihan Genre menggunakan Multi-select
genre_list = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
selected_genres = st.sidebar.multiselect("Pilih Genre Favorit:", genre_list)

# 3. Tombol untuk Trigger Rekomendasi
if st.sidebar.button("Generate Recommendation"):
    st.subheader(f"Rekomendasi Film untuk User {user_id}")
    
    # Menampilkan info genre yang dipilih
    if selected_genres:
        st.write(f"Menyaring preferensi berdasarkan genre: {', '.join(selected_genres)}")
    else:
        st.write("Menampilkan rekomendasi default (Top Rated).")
        
    # Data Tiruan (Dummy) sebelum Model Machine Learning Kita Pasang
    dummy_movies = {
        'Movie Title': ['Toy Story (1995)', 'Jumanji (1995)', 'Heat (1995)', 'Sabrina (1995)'],
        'Predicted Rating': [4.8, 4.5, 4.2, 4.0]
    }
    df_dummy = pd.DataFrame(dummy_movies)
    
    # Menampilkan tabel hasil ke layar aplikasi
    st.dataframe(df_dummy)
    st.success("Model Ensemble berhasil memproses prediksi!")