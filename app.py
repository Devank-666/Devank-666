
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    data = pd.read_csv('loksabhaelection.csv')
    return data

data = load_data()


st.title("Lok Sabha Election Results 2019")

# Overview
st.header("Overview")
st.write("""
This application provides a detailed visualization of the 2019 Lok Sabha election results.
Explore the data through various charts and gain insights into the election outcome.
""")


st.subheader("Data Preview")
st.write(data.head())


st.subheader("State-wise Winner Distribution")
state_winner_count = data.groupby('State Name')['Winning Margin %'].count().reset_index()
state_winner_chart = alt.Chart(state_winner_count).mark_bar().encode(
    x='Winning Margin %:Q',
    y=alt.Y('State Name:N', sort='-x'),
    tooltip=['State Name', 'Winning Margin %']
).properties(
    width=800,
    height=600
)
st.altair_chart(state_winner_chart)


st.subheader("Party-wise Vote Share")
party_vote_share = data.groupby('Party')['Candidate Votes'].sum().reset_index()
party_vote_share['Percentage'] = (party_vote_share['Candidate Votes'] / party_vote_share['Candidate Votes'].sum()) * 100
party_vote_pie = alt.Chart(party_vote_share).mark_arc().encode(
    theta=alt.Theta(field="Percentage", type="quantitative"),
    color=alt.Color(field="Party", type="nominal"),
    tooltip=['Party', 'Percentage']
).properties(
    width=400,
    height=400
)
st.altair_chart(party_vote_pie)


st.subheader("Constituency-wise Results")
constituency = st.selectbox("Select Constituency", data['Constituency Name'].unique())
constituency_data = data[data['Constituency Name'] == constituency]
st.write(constituency_data)


st.subheader("State-wise Party Distribution")
selected_state = st.selectbox("Select State", data['State Name'].unique())
state_data = data[data['State Name'] == selected_state]
state_party_dist = state_data.groupby('Party')['Winning Margin %'].count().reset_index()
state_party_bar = alt.Chart(state_party_dist).mark_bar().encode(
    x='Party:N',
    y='Winning Margin %:Q',
    tooltip=['Party', 'Winning Margin %']
).properties(
    width=800,
    height=400
)
st.altair_chart(state_party_bar)

party_filter = st.sidebar.multiselect('Select Parties', options=data['Party'].unique(), default=data['Party'].unique())

filtered_data = data[data['Party'].isin(party_filter)] 

states = filtered_data['State Name'].unique()

st.subheader('Comparison Between Two States or Constituencies')
comparison_type = st.selectbox('Select Comparison Type', ['States', 'Constituencies'])

if comparison_type == 'States':
    state1 = st.selectbox('Select first state', states, key='state1')
    state2 = st.selectbox('Select second state', states, key='state2')
    
    if state1 and state2:
        state1_data = filtered_data[filtered_data['State Name'] == state1]
        state2_data = filtered_data[filtered_data['State Name'] == state2]

        state1_seats_won = state1_data['Party'].value_counts().reset_index()
        state1_seats_won.columns = ['Party', 'Seats']
        state2_seats_won = state2_data['Party'].value_counts().reset_index()
        state2_seats_won.columns = ['Party', 'Seats']

        fig, ax = plt.subplots()
        sns.barplot(x='Party', y='Seats', data=state1_seats_won, color='blue', label=state1, ax=ax)
        sns.barplot(x='Party', y='Seats', data=state2_seats_won, color='orange', label=state2, ax=ax)
        ax.set_title(f'Seats Won Comparison: {state1} vs {state2}')
        ax.legend()
        st.pyplot(fig)

elif comparison_type == 'Constituencies':
    constituencies = filtered_data['Constituency Name'].unique()
    constituency1 = st.selectbox('Select first constituency', constituencies, key='constituency1')
    constituency2 = st.selectbox('Select second constituency', constituencies, key='constituency2')

    if constituency1 and constituency2:
        constituency1_data = filtered_data[filtered_data['Constituency Name'] == constituency1]
        constituency2_data = filtered_data[filtered_data['Constituency Name'] == constituency2]

        fig, ax = plt.subplots()
        sns.barplot(x='Candidate Name', y='Candidate Votes', data=constituency1_data, color='blue', label=constituency1, ax=ax)
        sns.barplot(x='Candidate Name', y='Candidate Votes', data=constituency2_data, color='orange', label=constituency2, ax=ax)
        ax.set_title(f'Votes Distribution Comparison: {constituency1} vs {constituency2}')
        ax.set_xlabel('Candidate Name')
        ax.set_ylabel('Candidate Votes')
        ax.legend()
        st.pyplot(fig)



st.subheader("Party Performance Comparison")
parties = st.multiselect("Select Parties to Compare", data['Party'].unique())
if parties:
    comparison_data = data[data['Party'].isin(parties)]
    party_performance = comparison_data.groupby(['State Name', 'Party'])['Candidate Votes'].sum().reset_index()
    performance_chart = alt.Chart(party_performance).mark_bar().encode(
        x='Candidate Votes:Q',
        y=alt.Y('State Name:N', sort='-x'),
        color='Party:N',
        tooltip=['State Name', 'Party', 'Candidate Votes']
    ).properties(
        width=800,
        height=600
    )
    st.altair_chart(performance_chart)


st.header("Summary and Insights")
st.write("""
This visualization tool allows you to explore the 2019 Lok Sabha election results in a detailed and interactive manner.
Use the various charts and filters to gain insights into the election outcome and understand the distribution of votes and winners across different states and parties.
""")


st.write("© 2024 Lok Sabha Election Analysis")

