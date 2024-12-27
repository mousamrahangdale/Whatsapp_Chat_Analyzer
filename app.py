import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from textblob import TextBlob
import functions

st.title('WhatsApp Chat Analyzer')
file = st.file_uploader("Choose a file")

if file:
    df = functions.generateDataFrame(file)
    try:
        dayfirst = st.radio("Select Date Format in text file:", ('dd-mm-yy', 'mm-dd-yy'))
        if dayfirst == 'dd-mm-yy':
            dayfirst = True
        else:
            dayfirst = False
        users = functions.getUsers(df)
        users_s = st.sidebar.selectbox("Select User to View Analysis", users)
        selected_user = ""

        if st.sidebar.button("Show Analysis"):
            selected_user = users_s

            st.title("Showing Results for : " + selected_user)
            df = functions.PreProcess(df, dayfirst)
            if selected_user != "Everyone":
                df = df[df['User'] == selected_user]
            df, media_cnt, deleted_msgs_cnt, links_cnt, word_count, msg_count = functions.getStats(df)
            st.title("Chat Statistics")
            stats_c = ["Messages", "Total Words", "Media Shared", "Links Shared", "Messages Deleted"]
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.subheader(stats_c[0])
                st.title(msg_count)
            with c2:
                st.subheader(stats_c[1])
                st.title(word_count)
            with c3:
                st.subheader(stats_c[2])
                st.title(media_cnt)
            with c4:
                st.subheader(stats_c[3])
                st.title(links_cnt)
            with c5:
                st.subheader(stats_c[4])
                st.title(deleted_msgs_cnt)

            # User Activity Count
            if selected_user == 'Everyone':
                x = df['User'].value_counts().head()
                name = x.index
                count = x.values
                st.title("Messaging Frequency")
                st.subheader('Messaging Percentage Count of Users')
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(round((df['User'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
                        columns={'User': 'name', 'count': 'percent'}))
                with col2:
                    fig, ax = plt.subplots()
                    ax.bar(name, count,color='red')
                    ax.set_xlabel("Users")
                    ax.set_ylabel("Message Sent")
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)


            # Sentiment Analysis
            st.title("Sentiment Analysis")
            def analyze_sentiment(message):
                sentiment = TextBlob(message).sentiment
                return sentiment.polarity, sentiment.subjectivity

            df[['sentiment_polarity', 'sentiment_subjectivity']] = df['Message'].apply(
                lambda x: pd.Series(analyze_sentiment(x)))
            positive = len(df[df['sentiment_polarity'] > 0])
            negative = len(df[df['sentiment_polarity'] < 0])
            neutral = len(df[df['sentiment_polarity'] == 0])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.header("Positive")
                st.title(positive)
            with col2:
                st.header("Negative")
                st.title(negative)
            with col3:
                st.header("Neutral")
                st.title(neutral)

            # Sentiment Distribution Plot (Scatter)
            st.title("Sentiment Analysis Overview")
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df['sentiment_polarity'], df['sentiment_subjectivity'], 
                                  c=df['sentiment_polarity'], cmap='Accent', alpha=0.9)
            ax.set_title("Sentiment Polarity vs Subjectivity", fontsize=16)
            ax.set_xlabel("Polarity", fontsize=12)
            ax.set_ylabel("Subjectivity", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.colorbar(scatter, label='Polarity')
            st.pyplot(fig)

            # Sentiment Word List with Subjectivity
            st.title("Sentiment Word List")
            df['sentiment_word'] = df['sentiment_polarity'].apply(
                lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
            st.dataframe(df[['User', 'Message', 'sentiment_word', 'sentiment_polarity', 'sentiment_subjectivity']])  

            # Emoji Analysis
            emojiDF = functions.getEmoji(df)
            st.title("Emoji Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emojiDF)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emojiDF[1].head(), labels=emojiDF[0].head(), autopct="%0.2f")
                plt.legend()
                st.pyplot(fig)  

            # Monthly Timeline
            timeline = functions.getMonthlyTimeline(df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['Message'])
            ax.set_xlabel("Month")
            ax.set_ylabel("Messages Sent")
            plt.xticks(rotation='vertical')
            st.title('Monthly Timeline')
            st.pyplot(fig)

            # Daily Timeline
            functions.dailytimeline(df)

            st.title('Most Busy Days')
            functions.WeekAct(df)
            st.title('Most Busy Months')
            functions.MonthAct(df)

            # Weekly Activity Map
            st.title("Weekly Activity Map")
            user_heatmap = functions.activity_heatmap(df)
            fig, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)

            # WordCloud
            st.title("Wordcloud")
            df_wc = functions.create_wordcloud(df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

            # Common Words
            commonWord = functions.MostCommonWords(df)
            fig, ax = plt.subplots()
            colors = sns.color_palette("coolwarm", len(commonWord))
            ax.barh(commonWord[0], commonWord[1], color=colors)
            ax.set_xlabel("Words")
            ax.set_ylabel("Frequency")
            plt.xticks(rotation='vertical')
            st.title('Most Frequent Words Used In Chat')
            st.pyplot(fig)

    except Exception as e:
        st.subheader("Unable to Process Your Request")
        st.error(str(e))
