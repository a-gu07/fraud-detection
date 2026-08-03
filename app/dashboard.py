import httpx
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import altair as alt

def api_get(path, params=None, timeout=30):
    """helper method that ensures we give a clean error message if any of the API calls time out or error"""
    try:
        response = httpx.get(f'{API_URL}{path}', params=params, timeout=timeout)
    except httpx.RequestError:
        st.error('API: Unreachable')
        st.stop()
    if response.status_code != 200:
        st.error(f'API: Error ({response.status_code})')
        st.stop()
    return response

st.session_state.setdefault('last_id', 0)
st.session_state.setdefault('start_time', None)
st.session_state.setdefault('bucket_scores', {})

API_URL = st.secrets.get('API_URL', 'http://127.0.0.1:8000')

st.set_page_config(page_title='Fraud Detection Dashboard', layout='wide')

st_autorefresh(interval=2000, key='refresh_timer')

st.title('Fraud Detection — Live Monitor')
st.badge('Made by Alan Gu :)', color='violet')
st.info(
    "This is a simulated live feed: transactions are replayed from a "
    "pre-downloaded and labeled dataset (Kaggle ULB credit card fraud "
    "dataset). The true fraud label for each transaction is known ahead of time. "
    "However, the labels are used only to compute the accuracy metrics below, "
    "never as an input to the "
    "scoring itself."
)
api_get('/health', timeout=65)
st.success("API: Connected")

st.caption(
    "Each transaction is scored on how statistically unusual it looks, using both "
    "a global baseline of normal transaction behavior and an EWMA model that shifts "
    "its baseline as transactions come in. A higher score means the transaction is "
    "more likely to be fraudulent. The slider sets the score threshold: transactions "
    "scoring at or above the score are flagged as alerts. "
)
threshold = st.slider(label='Score Threshold',min_value=0.0, max_value=100.0, value=19.08, step=0.5)

stats = api_get('/stats', params={'threshold': threshold}).json()

st.caption(
    "Precision is the percentage of flagged transactions that were actually fraud. "
    "Recall is the percentage of fraud transactions that got flagged. FPR(false positive rate) is the share of "
    "legitimate transactions wrongly flagged. All three are computed based on "
    "the selected threshold. Note: the default score of 19.08 "
    "is the threshold chosen to capture most of the available recall while keeping false positives to a fixed, low rate. "
)

st.caption('The time displayed is in the UTC zone.')


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric('Total Processed', stats['total'])
col2.metric('Alerts', stats['alerts'])
col3.metric('Precision', f"{stats['precision']:.2%}" if stats['precision'] is not None else '—')
col4.metric('Recall', f"{stats['recall']:.2%}" if stats['recall'] is not None else '—')
col5.metric('FPR', f"{stats['fpr']:.2%}" if stats['fpr'] is not None else '—')

transactions = api_get('/transactions').json()

if not transactions:
    # transactions is empty so the table ws dropped, we need to clear accumulated session state
    st.session_state['last_id'] = 0
    st.session_state['start_time'] = None
    st.session_state['bucket_scores'] = {}
    st.info('Waiting for data...')
    st.stop()

transactions = pd.DataFrame(transactions)
# necessary otherwise processed_at stays as strings
transactions['processed_at'] = pd.to_datetime(transactions['processed_at'])
#create an alert column that displays which transactions are alerted
transactions['Alert'] = transactions['Score'] >= threshold
transactions = transactions.set_index("id", drop=True)

def highlight_alerts(row):
    if row['Alert']:
        return ['background-color: #ffcccc'] * len(row)
    return [''] * len(row)

styled = transactions.style.apply(highlight_alerts, axis=1)
st.dataframe(styled)

if transactions.index.max() < st.session_state['last_id']:
    st.session_state['last_id'] = 0
    st.session_state['start_time'] = None
    st.session_state['bucket_scores'] = {}


new_transactions = api_get(
    '/transactions',
    params={'after_id': st.session_state['last_id'], 'limit': 5000},
).json()

#runs if new_transactions isnt empty
if new_transactions:
    new_transactions = pd.DataFrame(new_transactions)
    new_transactions['processed_at'] = pd.to_datetime(new_transactions['processed_at'])

    if st.session_state['start_time'] is None:
        st.session_state['start_time'] = new_transactions['processed_at'].min()

    elapsed = (new_transactions['processed_at'] - st.session_state['start_time']).dt.total_seconds()
    new_transactions['bucket_time'] = st.session_state['start_time'] + pd.to_timedelta((elapsed // 2) * 2, unit='s')

    for bucket_time, group in new_transactions.groupby('bucket_time')['Score']:
        if group.empty:
            continue
        st.session_state['bucket_scores'].setdefault(bucket_time, []).extend(group.tolist())

    st.session_state['last_id'] = int(new_transactions['id'].max())

bucket_times = sorted(st.session_state['bucket_scores'].keys())
rates = [
    sum(1 for s in st.session_state['bucket_scores'][bt] if s >= threshold) / len(st.session_state['bucket_scores'][bt])
    for bt in bucket_times
]
display_rates = pd.Series(rates, index=pd.DatetimeIndex(bucket_times))

x_min = st.session_state['start_time']
x_max = display_rates.index.max()

if display_rates.empty:
    st.info("Waiting for data...")
else:
    chart_df = display_rates.reset_index()
    chart_df.columns = ['Processed Time', 'Alert Rate']

    chart = (
        alt.Chart(chart_df)
        .mark_line(color='red')
        .encode(
            x=alt.X("Processed Time:T", scale=alt.Scale(domain=[x_min, x_max]), axis=alt.Axis(format="%H:%M:%S")),
            y=alt.Y("Alert Rate:Q", scale=alt.Scale(domain=[0, 1])),
        )
    )
    st.caption(
        'Note that the chart shows the alert rates in 2 second windows, NOT the cumulative alert rate. '
    )
    st.altair_chart(chart, width='stretch')

