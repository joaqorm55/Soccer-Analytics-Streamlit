import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from todofcpy.visualization.graph import Heatmap
from todofcpy.visualization.graph import Field
import math
import numpy as np
from streamlit_metrics import metric, metric_row
import plotly.graph_objs as go

df = pd.read_csv('goal_detail.csv')
df2 = pd.read_csv('foulcommit_detail.csv')
df3 = pd.read_csv('shoton_detail.csv')

league_list = df['LeagueName'].unique().tolist()
season_list = df['Season'].unique().tolist()

def get_grid_size(n_elements):
    """Calculate the size of a grid to fit a given number of elements."""
    n_rows = (n_elements + 2) // 3
    n_cols = 3
    return n_rows

def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Home", "Projects", "Contact"],  # required
                icons=["house", "book", "envelope"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Goals", "Shots", "Fouls"],  # required
            icons=["bi bi-dribbble", "bi bi-bullseye", "bi bi-exclamation-triangle"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Leagues", "Teams", "Players"],  # required
            icons=["bi bi-person-badge-fill", "book", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected

st.set_page_config(page_title='European League Visualization', page_icon=':soccer:', initial_sidebar_state='expanded', layout='wide')
selected = streamlit_menu(example=2)

if selected == "Goals":

    st.sidebar.markdown('## Select League ')
    menu_league = st.sidebar.selectbox('select', league_list)

    df = df[df['LeagueName'].isin([menu_league])]
    
    st.sidebar.markdown('## Select Season ')
    menu_season = st.sidebar.selectbox('select', season_list)

    selection = df[df['Season'].isin([menu_season])]

    TeamGoals = selection['team'].value_counts()
    PlayerGoals = selection['player1'].value_counts().nlargest(15)
    GameGoals = selection['match_id'].value_counts()
    GoalsperSeason = round(TeamGoals.mean(),2)
    GoalsperGame = round(GameGoals.mean(),2)
    MaxGoalperGame = max(GameGoals)
    MaxPlayerGoal = max(PlayerGoals)

    TypeGoals = selection['subtype'].value_counts().nlargest(6)

    TotalTypeGoals = selection['subtype'].value_counts()
    
    grouped_data = selection.groupby("team")


    # st.markdown('#### Basic Game Stats -  {} '.format(menu_league))
    st.markdown('<h4 style="text-align: center; color: gray;">Basic Game Stats - {}</h4>'.format(menu_league), unsafe_allow_html=True)

    
# 
    # Display the lines
    st.write('---')
    

    col1, col2, col3, col4= st.columns([1,1,1,1])
    with col1:
        st.metric(label="Average Goals in {} per team".format(menu_season), value=GoalsperSeason)
    with col2:
        st.metric(label="Average Goals per {}".format("games"), value=GoalsperGame)
    with col3:
        st.metric(label="Highest Goals in a {}".format("game"), value=MaxGoalperGame)
    with col4:
        st.metric(label="Highest Goals from Player", value=MaxPlayerGoal)
    col1.width = 0
    col2.width = 0
    col3.width = 0


    st.write('---')

    st.markdown('<h4 style="text-align: center; color: gray;">Top Goal Scorer in - {} - {} </h4>'.format(menu_league, menu_season), unsafe_allow_html=True)

    # Display the lines
    st.write('---')

    col1, col2, = st.columns([1,3])
    with col1:
        st.dataframe(PlayerGoals)
    with col2:
        st.bar_chart(PlayerGoals)

    st.write('---')
    

    st.markdown('<h4 style="text-align: center;color: gray;">Goal Types - {}</h4>'.format(menu_league), unsafe_allow_html=True)
    st.write('---')

    

    col1, col2, = st.columns([3,1])
    with col1:
        fig = go.Figure(go.Pie(labels=TypeGoals.index, values=TypeGoals.values, showlegend=True))
        st.plotly_chart(fig)
    with col2:
        st.dataframe(TotalTypeGoals)

    st.write('---')
    st.markdown('<h4 style="text-align: center;color: gray;">Goal by Time - {}</h4>'.format(menu_league), unsafe_allow_html=True)
    st.write('---')

    fig = go.Figure()

    fig.add_trace(go.Histogram(x=selection['elapsed'], nbinsx=20, opacity=0.9, name='Time'))

    # Update the layout
    fig.update_layout(title='Distribution of Goals based on Time',
                    xaxis_title='Time',
                    yaxis_title='Frequency',
                    xaxis_tickvals=list(range(0, 90, 10)))
    st.plotly_chart(fig)

    st.write('---')

    st.markdown('<h4 style="text-align: center;color: gray;">Location of Goals - {}</h4>'.format(menu_league), unsafe_allow_html=True)

    st.write('---')


    fig, ax = plt.subplots(figsize=(10,4))

    # plot the histograms
    ax.hist(selection['pos_y'], bins=100, alpha=0.9, label='GoaltoGoal')
    ax.hist(selection['pos_x'], bins=100, alpha=0.9, label='SidetoSide')

    # add a vertical line at the mean of each data set
    ax.axvline(selection['pos_y'].mean(), color='b', linestyle='dashed', linewidth=2)
    ax.axvline(selection['pos_x'].mean(), color='r', linestyle='dashed', linewidth=2)

    # add labels and legend
    ax.set_xlabel('Coordinate Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Goals based on Location')
    ax.legend()

    # display the plot using Streamlit
    st.pyplot(fig)

    st.write('---')
    st.markdown('<h4 style="text-align: center;color: gray;">Goals by width per team - {}</h4>'.format(menu_league), unsafe_allow_html=True)
    st.write('---')


    num_groups = len(grouped_data)
    num_cols = 3  # set the number of columns to 3
    num_rows = num_groups // num_cols
    if num_groups % num_cols != 0:
        num_rows += 1

    # create the subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

    # loop through the groups and create a histogram for each group
    for i, (group_name, group_data) in enumerate(grouped_data):
        row_idx = i // num_cols
        col_idx = i % num_cols
        axs[row_idx, col_idx].hist(group_data['pos_x'], bins=10)
        axs[row_idx, col_idx].set_title(f'Team {group_name}')

    # adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.5)

    # show the plot
    plt.show()

    col1, col2 = st.columns([3,1])
    with col1:
        st.pyplot(fig)
    with col2:
        st.table(TeamGoals)
    col1.width = 0
    col2.width = 0



if selected == "Shots":

    st.sidebar.markdown('## Select League ')
    menu_league = st.sidebar.selectbox('select', league_list)

    df3 = df3[df3['LeagueName'].isin([menu_league])]
    
    st.sidebar.markdown('## Select Season ')
    menu_season = st.sidebar.selectbox('select', season_list)

    selection = df3[df3['Season'].isin([menu_season])]

    TeamGoals = selection['team'].value_counts()
    PlayerGoals = selection['player1'].value_counts().nlargest(15)
    GameGoals = selection['match_id'].value_counts()
    GoalsperSeason = round(TeamGoals.mean(),2)
    GoalsperGame = round(GameGoals.mean(),2)
    MaxGoalperGame = max(GameGoals)
    MaxPlayerGoal = max(PlayerGoals)

    TypeGoals = selection['subtype'].value_counts().nlargest(6)

    TotalTypeGoals = selection['subtype'].value_counts()
    
    grouped_data = selection.groupby("team")


    # st.markdown('#### Basic Game Stats -  {} '.format(menu_league))
    st.markdown('<h4 style="text-align: center; color: gray;">Basic Game Stats - {}</h4>'.format(menu_league), unsafe_allow_html=True)

    
# 
    # Display the lines
    st.write('---')
    

    col1, col2, col3, col4= st.columns([1,1,1,1])
    with col1:
        st.metric(label="Average Shots in {} per team".format(menu_season), value=GoalsperSeason)
    with col2:
        st.metric(label="Average Shots per {}".format("games"), value=GoalsperGame)
    with col3:
        st.metric(label="Highest Shots in a {}".format("game"), value=MaxGoalperGame)
    with col4:
        st.metric(label="Highest Shots from Player", value=MaxPlayerGoal)
    col1.width = 0
    col2.width = 0
    col3.width = 0


    st.write('---')

    st.markdown('<h4 style="text-align: center; color: gray;">Top Shots Attempts per Player in - {} - {} </h4>'.format(menu_league, menu_season), unsafe_allow_html=True)

    # Display the lines
    st.write('---')

    col1, col2, = st.columns([1,3])
    with col1:
        st.dataframe(PlayerGoals)
    with col2:
        st.bar_chart(PlayerGoals)

    st.write('---')
    

    st.markdown('<h4 style="text-align: center;color: gray;">Shot Types - {}</h4>'.format(menu_league), unsafe_allow_html=True)
    st.write('---')

    

    col1, col2, = st.columns([3,1])
    with col1:
        fig = go.Figure(go.Pie(labels=TypeGoals.index, values=TypeGoals.values, showlegend=True))
        st.plotly_chart(fig)
    with col2:
        st.dataframe(TotalTypeGoals)

    st.write('---')
    st.markdown('<h4 style="text-align: center;color: gray;">Goal by Time - {}</h4>'.format(menu_league), unsafe_allow_html=True)
    st.write('---')

    fig = go.Figure()

    fig.add_trace(go.Histogram(x=selection['elapsed'], nbinsx=20, opacity=0.9, name='Time'))

    # Update the layout
    fig.update_layout(title='Distribution of Shots based on Time',
                    xaxis_title='Time',
                    yaxis_title='Frequency',
                    xaxis_tickvals=list(range(0, 90, 10)))
    st.plotly_chart(fig)

    st.write('---')

    st.markdown('<h4 style="text-align: center;color: gray;">Location of Shots - {}</h4>'.format(menu_league), unsafe_allow_html=True)

    st.write('---')


    fig, ax = plt.subplots(figsize=(10,4))

    # plot the histograms
    ax.hist(selection['pos_y'], bins=100, alpha=0.9, label='GoaltoGoal')
    ax.hist(selection['pos_x'], bins=100, alpha=0.9, label='SidetoSide')

    # add a vertical line at the mean of each data set
    ax.axvline(selection['pos_y'].mean(), color='b', linestyle='dashed', linewidth=2)
    ax.axvline(selection['pos_x'].mean(), color='r', linestyle='dashed', linewidth=2)

    # add labels and legend
    ax.set_xlabel('Coordinate Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Shots based on Location')
    ax.legend()

    # display the plot using Streamlit
    st.pyplot(fig)

    st.write('---')
    st.markdown('<h4 style="text-align: center;color: gray;">Shots by width per team - {}</h4>'.format(menu_league), unsafe_allow_html=True)
    st.write('---')


    num_groups = len(grouped_data)
    num_cols = 3  # set the number of columns to 3
    num_rows = num_groups // num_cols
    if num_groups % num_cols != 0:
        num_rows += 1

    # create the subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

    # loop through the groups and create a histogram for each group
    for i, (group_name, group_data) in enumerate(grouped_data):
        row_idx = i // num_cols
        col_idx = i % num_cols
        axs[row_idx, col_idx].hist(group_data['pos_x'], bins=10)
        axs[row_idx, col_idx].set_title(f'Team {group_name}')

    # adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.5)

    # show the plot
    plt.show()

    col1, col2 = st.columns([3,1])
    with col1:
        st.pyplot(fig)
    with col2:
        st.table(TeamGoals)
    col1.width = 0
    col2.width = 0


if selected == "Fouls":


    st.sidebar.markdown('## Select League ')
    menu_league = st.sidebar.selectbox('select', league_list)

    df2 = df2[df2['LeagueName'].isin([menu_league])]
    
    st.sidebar.markdown('## Select Season ')
    menu_season = st.sidebar.selectbox('select', season_list)

    selection = df2[df2['Season'].isin([menu_season])]

    TeamGoals = selection['team'].value_counts()
    PlayerGoals = selection['player1'].value_counts().nlargest(15)
    GameGoals = selection['match_id'].value_counts()
    GoalsperSeason = round(TeamGoals.mean(),2)
    GoalsperGame = round(GameGoals.mean(),2)
    MaxGoalperGame = max(GameGoals)
    MaxPlayerGoal = max(PlayerGoals)

    TypeGoals = selection['subtype'].value_counts().nlargest(6)

    TotalTypeGoals = selection['subtype'].value_counts()
    
    grouped_data = selection.groupby("team")


    # st.markdown('#### Basic Game Stats -  {} '.format(menu_league))
    st.markdown('<h4 style="text-align: center; color: gray;">Basic Game Stats - {}</h4>'.format(menu_league), unsafe_allow_html=True)

    
# 
    # Display the lines
    st.write('---')
    

    col1, col2, col3, col4= st.columns([1,1,1,1])
    with col1:
        st.metric(label="Average Fouls in {} per team".format(menu_season), value=GoalsperSeason)
    with col2:
        st.metric(label="Average Fouls per {}".format("games"), value=GoalsperGame)
    with col3:
        st.metric(label="Highest Fouls in a {}".format("game"), value=MaxGoalperGame)
    with col4:
        st.metric(label="Highest Fouls from Player", value=MaxPlayerGoal)
    col1.width = 0
    col2.width = 0
    col3.width = 0


    st.write('---')

    st.markdown('<h4 style="text-align: center; color: gray;">Top Fouls per Players in - {} - {} </h4>'.format(menu_league, menu_season), unsafe_allow_html=True)

    # Display the lines
    st.write('---')

    col1, col2, = st.columns([1,3])
    with col1:
        st.dataframe(PlayerGoals)
    with col2:
        st.bar_chart(PlayerGoals)

    st.write('---')
    

    st.markdown('<h4 style="text-align: center;color: gray;">Fouls Types - {}</h4>'.format(menu_league), unsafe_allow_html=True)
    st.write('---')

    

    col1, col2, = st.columns([3,1])
    with col1:
        fig = go.Figure(go.Pie(labels=TypeGoals.index, values=TypeGoals.values, showlegend=True))
        st.plotly_chart(fig)
    with col2:
        st.dataframe(TotalTypeGoals)

    st.write('---')
    st.markdown('<h4 style="text-align: center;color: gray;">Fouls by Time - {}</h4>'.format(menu_league), unsafe_allow_html=True)
    st.write('---')

    fig = go.Figure()

    fig.add_trace(go.Histogram(x=selection['elapsed'], nbinsx=20, opacity=0.9, name='Time'))

    # Update the layout
    fig.update_layout(title='Distribution of Fouls based on Time',
                    xaxis_title='Time',
                    yaxis_title='Frequency',
                    xaxis_tickvals=list(range(0, 90, 10)))
    st.plotly_chart(fig)

    st.write('---')

    st.markdown('<h4 style="text-align: center;color: gray;">Location of Fouls - {}</h4>'.format(menu_league), unsafe_allow_html=True)

    st.write('---')


    fig, ax = plt.subplots(figsize=(10,4))

    # plot the histograms
    ax.hist(selection['pos_y'], bins=100, alpha=0.9, label='GoaltoGoal')
    ax.hist(selection['pos_x'], bins=100, alpha=0.9, label='SidetoSide')

    # add a vertical line at the mean of each data set
    ax.axvline(selection['pos_y'].mean(), color='b', linestyle='dashed', linewidth=2)
    ax.axvline(selection['pos_x'].mean(), color='r', linestyle='dashed', linewidth=2)

    # add labels and legend
    ax.set_xlabel('Coordinate Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Fouls based on Location')
    ax.legend()

    # display the plot using Streamlit
    st.pyplot(fig)

    st.write('---')
    st.markdown('<h4 style="text-align: center;color: gray;">Fouls by width per team - {}</h4>'.format(menu_league), unsafe_allow_html=True)
    st.write('---')


    num_groups = len(grouped_data)
    num_cols = 3  # set the number of columns to 3
    num_rows = num_groups // num_cols
    if num_groups % num_cols != 0:
        num_rows += 1

    # create the subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

    # loop through the groups and create a histogram for each group
    for i, (group_name, group_data) in enumerate(grouped_data):
        row_idx = i // num_cols
        col_idx = i % num_cols
        axs[row_idx, col_idx].hist(group_data['pos_x'], bins=10)
        axs[row_idx, col_idx].set_title(f'Team {group_name}')

    # adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.5)

    # show the plot
    plt.show()

    col1, col2 = st.columns([3,1])
    with col1:
        st.pyplot(fig)
    with col2:
        st.table(TeamGoals)
    col1.width = 0
    col2.width = 0




# field = Field()
# plt = field.create_field(grass='#FFFFFF', lines = '#C8102E')
# plt.scatter((Team_Goal['pos_y']/69)*120,(Team_Goal['pos_x']/44)*80,c = '#012169')
# plt.show()

# df = Player_Goal[['pos_x','pos_y']]
# dat = df.to_numpy()
# thm = Heatmap(data=dat)
# thm.set_colors(color='plasma')
# plt = thm.create_heatmap_plot()
# plt.show()