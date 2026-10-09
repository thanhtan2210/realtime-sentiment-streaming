import streamlit as st
import pandas as pd
from pymongo import MongoClient
import time
import plotly.express as px

from src.common import config

st.set_page_config(page_title="Real-time Sentiment Monitor", layout="wide")

st.title("Real-time Sentiment Analysis Dashboard")
st.markdown("Monitoring user emotions from social media (Kafka + Spark + MongoDB)")

# MongoDB Connection
@st.cache_resource
def get_db_connection():
    client = MongoClient(config.MONGO_URI)
    return client[config.DB_NAME][config.COLLECTION_NAME]

db_collection = get_db_connection()

placeholder_stats = st.empty()

def load_data():
    data = list(db_collection.find().sort("_id", -1).limit(1000))
    if data:
        df = pd.DataFrame(data)
        df["_id"] = df["_id"].astype(str)
        return df
    return pd.DataFrame()

while True:
    df = load_data()
    with placeholder_stats.container():
        if not df.empty:
            total_msgs = len(df)
            st.metric("Total Messages (Last 1000 records)", total_msgs)
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Common Emotions")
                sentiment_counts = df['sentiment'].value_counts()
                fig_pie = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, hole=0.3)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with c2:
                st.subheader("Activity by Country")
                country_counts = df['country'].value_counts().head(10)
                fig_bar = px.bar(x=country_counts.index, y=country_counts.values, labels={'x':'Country', 'y':'Count'})
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.subheader("Latest Detailed Data")
            st.dataframe(df[['text', 'sentiment', 'country', 'platform']].head(10), use_container_width=True)
        else:
            st.warning("Waiting for data from Spark Streaming...")

    time.sleep(3)
