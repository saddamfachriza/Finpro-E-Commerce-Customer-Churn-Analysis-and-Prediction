import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ============================================================
# PAGE CONFIG — HARUS PALING ATAS, SEBELUM APAPUN
# ============================================================

st.set_page_config(
    page_title = 'Cartify Churn Predictor',
    page_icon  = '🛒',
    layout     = 'wide'
)

# ============================================================
# LOAD MODEL — SATU FUNGSI SAJA, PAKAI PATH AMAN
# ============================================================

@st.cache_resource
def load_model():
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'best_model.sav')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

model          = load_model()
BEST_THRESHOLD = 0.38

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict_churn(input_df):
    proba      = model.predict_proba(input_df)[:, 1]
    prediction = (proba >= BEST_THRESHOLD).astype(int)
    return proba, prediction

def get_risk_segment(proba):
    if proba >= 0.9727:
        return 'High Risk', '#F44336'
    elif proba >= 0.8976:
        return 'Medium Risk', '#FF9800'
    else:
        return 'Low Risk', '#4CAF50'

def get_retention_action(segment):
    actions = {
        'High Risk'  : 'Personal outreach segera, diskon eksklusif, dedicated customer success manager',
        'Medium Risk': 'Email campaign personal, loyalty points, survei kepuasan',
        'Low Risk'   : 'Newsletter berkala, general promo, push notification re-engagement'
    }
    return actions.get(segment, '-')

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        text-align: center;
        padding: 1rem 0 0.2rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #AAAAAA;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .churn-alert {
        background: #FFEBEE;
        border-left: 5px solid #F44336;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    .safe-alert {
        background: #E8F5E9;
        border-left: 5px solid #4CAF50;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    .high-risk  { color: #F44336; font-weight: bold; }
    .med-risk   { color: #FF9800; font-weight: bold; }
    .low-risk   { color: #4CAF50; font-weight: bold; }
    .section-divider {
        border-top: 2px solid #dee2e6;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-header">🛒 Cartify Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistem Prediksi Customer Churn berbasis Machine Learning — XGBoost + No Resampling + Class Weight | Threshold: 0.38</div>', unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(['Single Prediction', 'Batch Prediction (CSV)'])

# ============================================================
# TAB 1: SINGLE PREDICTION
# ============================================================

with tab1:
    st.subheader('Prediksi Churn untuk Satu Customer')
    st.write('Isi data customer di bawah ini untuk memprediksi kemungkinan churn.')

    with st.form('single_prediction_form'):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('**Data Transaksi**')
            Tenure = st.number_input(
                'Tenure (bulan)', min_value=0, max_value=61, value=10,
                help='Lama customer bergabung dalam bulan'
            )
            WarehouseToHome = st.number_input(
                'Warehouse to Home (km)', min_value=5, max_value=36, value=15,
                help='Jarak dari warehouse ke rumah customer'
            )
            DaySinceLastOrder = st.number_input(
                'Day Since Last Order', min_value=0, max_value=46, value=3,
                help='Hari sejak order terakhir'
            )
            OrderCount = st.number_input(
                'Order Count', min_value=1, max_value=16, value=2,
                help='Jumlah order dalam periode tertentu'
            )
            CouponUsed = st.number_input(
                'Coupon Used', min_value=0, max_value=16, value=1,
                help='Jumlah kupon yang digunakan'
            )

        with col2:
            st.markdown('**Data Customer**')
            CityTier = st.selectbox(
                'City Tier', options=[1, 2, 3],
                help='Tier kota tempat tinggal customer'
            )
            NumberOfDeviceRegistered = st.number_input(
                'Number of Device Registered', min_value=1, max_value=6, value=3,
                help='Jumlah perangkat yang terdaftar'
            )
            NumberOfAddress = st.number_input(
                'Number of Address', min_value=1, max_value=22, value=3,
                help='Jumlah alamat pengiriman yang terdaftar'
            )
            SatisfactionScore = st.selectbox(
                'Satisfaction Score', options=[1, 2, 3, 4, 5],
                help='Skor kepuasan customer (1=sangat tidak puas, 5=sangat puas)'
            )
            HourSpendOnApp = st.selectbox(
                'Hour Spend on App', options=[0, 1, 2, 3, 4, 5],
                help='Jam yang dihabiskan di aplikasi per hari'
            )

        with col3:
            st.markdown('**Data Preferensi**')
            Gender = st.selectbox(
                'Gender', options=['Male', 'Female']
            )
            MaritalStatus = st.selectbox(
                'Marital Status', options=['Single', 'Married', 'Divorced']
            )
            PreferredLoginDevice = st.selectbox(
                'Preferred Login Device', options=['Mobile Phone', 'Computer']
            )
            PreferredPaymentMode = st.selectbox(
                'Preferred Payment Mode', options=['Debit Card', 'UPI', 'CC', 'COD', 'E wallet']
            )
            PreferedOrderCat = st.selectbox(
                'Preferred Order Category',
                options=['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others']
            )
            Complain = st.selectbox(
                'Complain', options=[0, 1],
                format_func=lambda x: 'Tidak Ada Komplain (0)' if x == 0 else 'Ada Komplain (1)',
                help='Apakah customer pernah mengajukan komplain'
            )
            OrderAmountHikeFromlastYear = st.number_input(
                'Order Amount Hike From Last Year (%)',
                min_value=11, max_value=26, value=15,
                help='Persentase kenaikan jumlah order dibanding tahun lalu'
            )
            CashbackAmount = st.number_input(
                'Cashback Amount ($)', min_value=0.0, max_value=325.0, value=150.0,
                help='Jumlah cashback yang diterima customer'
            )

        submitted = st.form_submit_button(
            'Prediksi Churn',
            use_container_width=True,
            type='primary'
        )

    # Hasil prediksi
    if submitted:
        input_data = pd.DataFrame([{
            'Tenure'                     : Tenure,
            'PreferredLoginDevice'       : PreferredLoginDevice,
            'CityTier'                   : CityTier,
            'WarehouseToHome'            : float(WarehouseToHome),
            'PreferredPaymentMode'       : PreferredPaymentMode,
            'Gender'                     : Gender,
            'HourSpendOnApp'             : float(HourSpendOnApp),
            'NumberOfDeviceRegistered'   : NumberOfDeviceRegistered,
            'PreferedOrderCat'           : PreferedOrderCat,
            'SatisfactionScore'          : SatisfactionScore,
            'MaritalStatus'              : MaritalStatus,
            'NumberOfAddress'            : NumberOfAddress,
            'Complain'                   : Complain,
            'OrderAmountHikeFromlastYear': float(OrderAmountHikeFromlastYear),
            'CouponUsed'                 : float(CouponUsed),
            'OrderCount'                 : float(OrderCount),
            'DaySinceLastOrder'          : float(DaySinceLastOrder),
            'CashbackAmount'             : CashbackAmount,
        }])

        proba, prediction = predict_churn(input_data)
        segment, seg_color = get_risk_segment(proba[0])
        action = get_retention_action(segment)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader('Hasil Prediksi')

        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label = 'Status Prediksi',
                value = 'CHURN' if prediction[0] == 1 else 'NOT CHURN'
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label = 'Churn Probability',
                value = f'{proba[0]*100:.2f}%'
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label = 'Risk Segment',
                value = segment
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Alert box
        if prediction[0] == 1:
            st.markdown(f"""
            <div class="churn-alert">
                <h4 style="color:#B71C1C; margin:0 0 0.8rem 0">
                    Customer Berisiko Churn!
                </h4>
                <p style="margin:0.4rem 0; color:#1a1a1a; font-size:0.95rem">
                    <b style="color:#1a1a1a">Segmen:</b>
                    <span style="color:{seg_color}; font-weight:700">
                        {segment}
                    </span>
                </p>
                <p style="margin:0.4rem 0; color:#1a1a1a; font-size:0.95rem">
                    <b style="color:#1a1a1a">Probabilitas:</b>
                    <span style="color:#B71C1C; font-weight:700">
                        {proba[0]*100:.2f}%
                    </span>
                </p>
                <p style="margin:0.4rem 0; color:#1a1a1a; font-size:0.95rem">
                    <b style="color:#1a1a1a">Rekomendasi Treatment:</b>
                    <span style="color:#424242">
                        {action}
                    </span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-alert">
                <h4 style="color:#1B5E20; margin:0 0 0.8rem 0">
                    Customer Tidak Berisiko Churn
                </h4>
                <p style="margin:0.4rem 0; color:#1a1a1a; font-size:0.95rem">
                    <b style="color:#1a1a1a">Probabilitas Churn:</b>
                    <span style="color:#2E7D32; font-weight:700">
                        {proba[0]*100:.2f}%
                    </span>
                </p>
                <p style="margin:0.4rem 0; color:#424242; font-size:0.95rem">
                    <b style="color:#1a1a1a">Status:</b>
                    Customer aman, tidak memerlukan treatment retensi khusus.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Data input summary
        with st.expander('Lihat Data Input Customer'):
            st.dataframe(input_data.T.rename(columns={0: 'Value'}))

# ============================================================
# TAB 2: BATCH PREDICTION
# ============================================================

with tab2:
    st.subheader('Prediksi Churn untuk Banyak Customer (Upload CSV)')
    st.write('Upload file CSV berisi data customer untuk prediksi secara batch.')

    # Template download
    st.markdown('**Download Template CSV:**')
    template_df = pd.DataFrame([{
        'Tenure'                     : 10,
        'PreferredLoginDevice'       : 'Mobile Phone',
        'CityTier'                   : 1,
        'WarehouseToHome'            : 15.0,
        'PreferredPaymentMode'       : 'Debit Card',
        'Gender'                     : 'Male',
        'HourSpendOnApp'             : 3.0,
        'NumberOfDeviceRegistered'   : 3,
        'PreferedOrderCat'           : 'Laptop & Accessory',
        'SatisfactionScore'          : 3,
        'MaritalStatus'              : 'Single',
        'NumberOfAddress'            : 3,
        'Complain'                   : 0,
        'OrderAmountHikeFromlastYear': 15.0,
        'CouponUsed'                 : 1.0,
        'OrderCount'                 : 2.0,
        'DaySinceLastOrder'          : 3.0,
        'CashbackAmount'             : 150.0,
    }])

    st.download_button(
        label     = 'Download Template CSV',
        data      = template_df.to_csv(index=False),
        file_name = 'template_churn_prediction.csv',
        mime      = 'text/csv'
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Upload CSV
    uploaded_file = st.file_uploader(
        'Upload CSV File',
        type = ['csv'],
        help = 'File CSV harus memiliki kolom yang sama dengan template di atas'
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f'File berhasil diupload: {len(df_upload)} baris data')

            with st.expander('Preview Data Upload (5 baris pertama)'):
                st.dataframe(df_upload.head())

            if st.button('Jalankan Prediksi Batch', type='primary', use_container_width=True):

                with st.spinner('Memproses prediksi...'):
                    proba_batch, pred_batch = predict_churn(df_upload)

                    df_result = df_upload.copy()
                    df_result['Churn_Probability'] = proba_batch.round(4)
                    df_result['Predicted_Churn']   = pred_batch
                    df_result['Predicted_Churn']   = df_result['Predicted_Churn'].map(
                        {1: 'Churn', 0: 'Not Churn'}
                    )

                    # Segmentasi hanya untuk yang diprediksi Churn
                    df_result['Risk_Segment'] = df_result.apply(
                        lambda row: get_risk_segment(row['Churn_Probability'])[0]
                        if row['Predicted_Churn'] == 'Churn' else '-',
                        axis=1
                    )

                # Summary metrics
                total         = len(df_result)
                total_churn   = (df_result['Predicted_Churn'] == 'Churn').sum()
                total_safe    = (df_result['Predicted_Churn'] == 'Not Churn').sum()
                high_risk     = (df_result['Risk_Segment'] == 'High Risk').sum()
                medium_risk   = (df_result['Risk_Segment'] == 'Medium Risk').sum()
                low_risk      = (df_result['Risk_Segment'] == 'Low Risk').sum()

                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.subheader('Hasil Prediksi Batch')

                col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                col_m1.metric('Total Customer', total)
                col_m2.metric('Diprediksi Churn', total_churn)
                col_m3.metric('High Risk', high_risk)
                col_m4.metric('Medium Risk', medium_risk)
                col_m5.metric('Low Risk', low_risk)

                # Tabel hasil
                st.dataframe(
                    df_result[['Churn_Probability', 'Predicted_Churn', 'Risk_Segment']
                              + list(df_upload.columns)],
                    use_container_width=True
                )

                # Download hasil
                st.download_button(
                    label     = 'Download Hasil Prediksi CSV',
                    data      = df_result.to_csv(index=False),
                    file_name = 'hasil_prediksi_churn.csv',
                    mime      = 'text/csv',
                    type      = 'primary'
                )

        except Exception as e:
            st.error(f'Error saat memproses file: {str(e)}')
            st.write('Pastikan format CSV sesuai dengan template yang disediakan.')

# ============================================================
# FOOTER
# ============================================================

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.85rem;">
    Cartify Churn Predictor | Model: XGBoost + No Resampling + Class Weight |
    Threshold: 0.38 | F2-Score: 0.9192 | Recall: 93.45%
</div>
""", unsafe_allow_html=True)