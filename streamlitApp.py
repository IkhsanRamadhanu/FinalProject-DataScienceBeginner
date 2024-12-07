import streamlit as st

st.title('HULU Content Recommendation')
st.markdown('Type your favourite show title below, and we will recommend you 5 more shows that suits you!')
userInput = st.text_input("Show Title", placeholder='Insert your favorite show in Hulu!')

import pandas as pd
df = pd.read_csv('data_hulu.csv')
df.info()

df = df.dropna()
df = df.drop_duplicates('imdbId')

df_id = df['imdbId'].tolist()
df_title = df['title'].tolist()
df_genre = df['genres'].tolist()
df_type = df['type'].tolist()

contentData = pd.DataFrame({
    'id': df_id,
    'title': df_title,
    'type': df_type,
    'genres': df_genre
})

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
tf.fit(contentData['genres'])
tfidf_matrix = tf.fit_transform(contentData['genres'])
tfidf_matrix.todense()

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tf.get_feature_names_out(),
    index=contentData.title
).sample(22, axis=1).sample(10, axis=0)

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix)

cosine_sim_df = pd.DataFrame(cosine_sim, index=contentData['title'], columns=contentData['title'])

def show_recommendations(showName, similarity_data=cosine_sim_df, items=contentData[['title', 'genres', 'type']], k=5):
    # Buat salinan eksplisit untuk menghindari SettingWithCopyWarning
    items = items.copy()

    # Simpan versi asli judul untuk ditampilkan nanti
    items['title_original'] = items['title']

    # Normalize the case for input and DataFrame columns
    showName = showName.lower()
    similarity_data.columns = similarity_data.columns.str.lower()
    items['title'] = items['title'].str.lower()

    # Handle the case where the input title might not exist
    if showName not in similarity_data.columns:
        # return f'"{showName}" tidak ditemukan di dalam dataset. Tolong periksa ulang dan masukan nama content yang valid.'
        return False

    # Find recommendations
    index = similarity_data.loc[:, showName].to_numpy().argpartition(range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(showName, errors='ignore')

    # Return recommendations by merging with items
    recommendations = pd.DataFrame(closest, columns=['title']).merge(items, on='title').head(k)
    recommendations['rank'] = recommendations.index + 1
    recommendations = recommendations[['rank'] + [col for col in recommendations.columns if col != 'rank']]
    return recommendations[['rank','title_original', 'genres', 'type']].rename(columns={'title_original': 'title'})


if userInput:
    df = show_recommendations(userInput)
    st.divider()

    
    if df is False:
        # Jika input tidak valid, tampilkan pesan error
        st.error(f'"{userInput}" not found in the dataset. Please check again and enter a valid show title.')
    else:
        # Jika input valid, tampilkan hasil rekomendasi
        st.write("Here are our show recommendations based on your favorite show. Enjoy watching!")
        st.dataframe(
            df,
            column_config={
                "rank": "Rank",
                "title": "Title",
                "genres": 'Genres',
                "type": 'Type',
            },
            hide_index=True,
            use_container_width=True,
        )
