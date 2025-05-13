# app.py

import streamlit as st
from concordance import preprocess, search_concordance_with_lemma, format_concordance, get_word_frequencies, find_collocations, get_clusters
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Armenian Text Analyzer", layout="wide")
st.title("📚 Armenian Text Analyzer")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Concordance", "📊 Word Frequency", "🔗 Collocation Analysis", "🔗 N-gram Clusters"])

# Concordance tab
with tab1:
    st.header("🔍 Concordance Search")
    st.markdown("Upload an Armenian novel or text file and search word usage contextually.")

    # Upload .txt file
    uploaded_file = st.file_uploader("📁 Upload an Armenian .txt file", type=["txt"])

    # Keyword input
    keyword = st.text_input("🔍 Enter search keyword (lemma form):")

    # Window size
    window = st.slider("🧩 Number of words before/after keyword", min_value=1, max_value=6, value=3)

    text = ""
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")

    if st.button("🔎 Search") and text and keyword:
        progress_bar = st.progress(0, text="🔄 Starting...")

        def update_progress(current, total):
            progress = current / total
            progress_bar.progress(progress, text=f"🔄 Lemmatizing sentence {current}/{total}")

        sentences = preprocess(text, on_progress=update_progress)

        progress_bar.empty()

        with st.spinner("🔍 Searching for matches..."):
            results = search_concordance_with_lemma(sentences, keyword, window=window)
            formatted = format_concordance(results, window=window)

        if formatted:
            st.success(f"Found {len(formatted)} match(es):")
            with st.container():
                st.markdown(
                    f"""
                    <div style='
                        border:1px solid #444;
                        padding:10px;
                        height:300px;
                        overflow-y:scroll;
                        background-color:#111;
                        color:#ccc;
                    '>
                        {'<br>'.join(formatted)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            st.info("No matches found.")


# Word Frequency tab
with tab2:
    st.header("📊 Word Frequency Analysis")

    uploaded_file = st.file_uploader("📄 Upload a TXT file for frequency analysis", type=["txt"], key="freq_file")

    if uploaded_file:
        raw_text = uploaded_file.read().decode("utf-8")
        
    col1, col2, col3 = st.columns(3)

    use_lemmas = col1.selectbox("Use Lemmatization", ["Yes", "No"]) == "Yes"
    remove_stop = col2.selectbox("Remove Stopwords", ["Yes", "No"]) == "Yes"

    show_top_n = col3.selectbox("Show", ["Top N Words", "All Words"])
    if show_top_n == "Top N Words":
        top_n = st.number_input("Show Top N words", min_value=1, value=50, step=5)
    else:
        top_n = None

    if st.button("📊 Show Frequencies"):
        with st.spinner("Analyzing..."):
            freqs = get_word_frequencies(raw_text, use_lemmas, remove_stop)

            if top_n:
                freqs = dict(freqs.most_common(top_n))  # Top N frequencies
            else:
                freqs = dict(sorted(freqs.items(), key=lambda x: -x[1]))  # Sort in descending order

            # Convert the frequencies to a DataFrame
            freq_df = pd.DataFrame(list(freqs.items()), columns=["Word", "Frequency"])

            # Create an interactive bar chart using Plotly
            fig = px.bar(freq_df, x="Word", y="Frequency", 
                        title="Word Frequency Chart", 
                        labels={"Word": "Words", "Frequency": "Frequency"})

            # Update the layout for better readability
            fig.update_layout(
                xaxis_tickangle=-45,  # Rotate the x-axis labels
                xaxis_title="Words",
                yaxis_title="Frequency",
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
                paper_bgcolor="rgba(0,0,0,0)"  # Transparent background
            )

            # Display the Plotly chart in Streamlit
            st.plotly_chart(fig)

            st.subheader("📝 Word List")
            scrollable_box = "\n".join([f"{word}: {count}" for word, count in freqs.items()])
            st.text_area("Frequencies", scrollable_box, height=300)
            
with tab3:
    st.header("🔗 Collocation Analysis")

    uploaded_file = st.file_uploader("📄 Upload a TXT file for collocation analysis", type=["txt"], key="colloc_file")

    if uploaded_file:
        raw_text = uploaded_file.read().decode("utf-8")

    col1, col2, col3 = st.columns(3)
    keyword = st.text_input("Enter keyword for collocation analysis").strip().lower()
    measure = col1.selectbox("Collocation Measure", ["PMI", "T-Score"])
    use_lemmas = col2.selectbox("Use Lemmatization", ["Yes", "No"], key="lemma_colloc") == "Yes"
    ordering = col3.selectbox("Order by",["Frequency", "Frequency(L)", "Frequency(R)", "Score"], key="ordering")
    window_size = st.slider("Window size (words before and after)", min_value=1, max_value=10, value=5)

    if st.button("🔍 Find Collocations"):
        with st.spinner("Processing..."):
            colloc_df = find_collocations(text, keyword, window, measure, use_lemmas=use_lemmas, ordering=ordering)
            st.subheader("📋 Collocation Results")
            st.dataframe(colloc_df)

with tab4:
    st.header("🔗 Cluster Analysis (N-grams)")

    uploaded_file = st.file_uploader("📄 Upload a TXT file for cluster analysis", type=["txt"], key="cluster_file")

    if uploaded_file:
        raw_text = uploaded_file.read().decode("utf-8")
    else:
        st.stop()

    col1, col2, col3 = st.columns(3)
    use_lemmas = col1.selectbox("Use Lemmatization", ["No", "Yes"], key="cluster_lemmas") == "Yes"
    remove_stop = col2.selectbox("Remove Stopwords", ["No", "Yes"], key="cluster_stop") == "Yes"
    ngram_size = col3.number_input("N-gram Size", min_value=2, max_value=5, value=3)

    top_n = st.number_input("Show Top N Clusters", min_value=1, value=50, step=5)

    if st.button("🔍 Analyze Clusters"):
        with st.spinner("Processing clusters..."):
            cluster_data = get_clusters(raw_text, use_lemmas, remove_stop, ngram_size)
            sorted_clusters = sorted(cluster_data.items(), key=lambda x: -x[1]['frequency'])[:top_n]

            table_data = []
            for rank, (cluster, stats) in enumerate(sorted_clusters, start=1):
                table_data.append({
                    "Rank": rank,
                    "Cluster": cluster,
                    "Frequency": stats["frequency"],
                    "Range": stats["range"],
                    "Size": ngram_size
                })

            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
