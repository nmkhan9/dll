from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from joblib import load

AIRLINES = [
    '0N', '2A', '2E', '2G', '2J', '2M', '2T', '2W', '2Z', '3F', '3H', '3K', '3L', '3M', '3O', '3R', '3T', '3U'
    '4A', '4B', '4N', '4Y', '4Z', '5E', '5F', '5J', '5N', '5O', '5R', '5T', '5U', '5V', '5W', '5Z', '6D', '6E',
    '6H', '6J', '6R', '7C', '7G', '7H', '7P', '7R', '7S', '7V', '8B', '8E', '8F', '8L', '8M', '8V', '9C', '9H',
    '9I', '9J', '9K', '9N', '9R', '9X', 'A3', 'A6', 'A8', 'A9', 'AC', 'AD', 'AE', 'AF', 'AH', 'AI', 'AK', 'AM',
    'AQ', 'AR', 'AS', 'AT', 'AV', 'AW', 'AY', 'AZ', 'B3', 'B6', 'B7', 'B9', 'BA', 'BC', 'BG', 'BI', 'BJ', 'BK',
    'BP', 'BR', 'BS', 'BT', 'BU', 'BV', 'BW', 'BX', 'BY', 'BZ', 'C7', 'CA', 'CC', 'CD', 'CE', 'CG', 'CI', 'CM',
    'CQ', 'CX', 'CY', 'CZ', 'D2', 'D7', 'D8', 'DD', 'DE', 'DG', 'DI', 'DJ', 'DK', 'DL', 'DM', 'DO', 'DR', 'DT',
    'DV', 'DX', 'DY', 'DZ', 'E5', 'E9', 'EI', 'EK', 'EL', 'EN', 'EO', 'EP', 'ER', 'ES', 'ET', 'EU', 'EW', 'EY',
    'F2', 'F3', 'F8', 'F9', 'FA', 'FB', 'FD', 'FG', 'FH', 'FI', 'FJ', 'FM', 'FN', 'FO', 'FR', 'FU', 'FW', 'FY',
    'FZ', 'G3', 'G4', 'G5', 'G9', 'GA', 'GD', 'GF', 'GJ', 'GK', 'GL', 'GM', 'GQ', 'GS', 'GV', 'GX', 'GY', 'H2',
    'H4', 'H5', 'H7', 'H9', 'HA', 'HC', 'HD', 'HE', 'HF', 'HH', 'HJ', 'HK', 'HM', 'HO', 'HU', 'HV', 'HW', 'HX',
    'HY', 'HZ', 'I2', 'I4', 'I5', 'IA', 'IB', 'ID', 'IE', 'IK', 'IN', 'IO', 'IR', 'IT', 'IU', 'IW', 'IX', 'IY',
    'IZ', 'J2', 'J4', 'J5', 'J7', 'J9', 'JA', 'JD', 'JH', 'JL', 'JM', 'JQ', 'JT', 'JU', 'JX', 'JY', 'K3', 'K6',
    'K7', 'KB', 'KC', 'KE', 'KG', 'KL', 'KM', 'KN', 'KP', 'KQ', 'KR', 'KU', 'KX', 'KY', 'L6', 'LA', 'LC', 'LF',
    'LG', 'LH', 'LJ', 'LM', 'LN', 'LO', 'LS', 'LT', 'LX', 'LY', 'M0', 'ME', 'MF', 'MH', 'MK', 'MM', 'MO', 'MS',
    'MU', 'MX', 'N2', 'N3', 'N4', 'N9', 'NE', 'NF', 'NH', 'NK', 'NO', 'NP', 'NS', 'NT', 'NU', 'NX', 'NZ', 'OA',
    'OB', 'OC', 'OD', 'OG', 'OJ', 'OK', 'OP', 'OQ', 'OR', 'OS', 'OU', 'OV', 'OW', 'OZ', 'P2', 'P4', 'P5', 'PA',
    'PB', 'PC', 'PD', 'PG', 'PI', 'PJ', 'PK', 'PM', 'PN', 'PR', 'PU', 'PV', 'PW', 'PX', 'PY', 'Q2', 'Q4', 'QB',
    'QC', 'QF', 'QG', 'QH', 'QI', 'QN', 'QP', 'QQ', 'QR', 'QS', 'QV', 'QW', 'QZ', 'R3', 'RA', 'RC', 'RF', 'RJ',
    'RN', 'RO', 'RQ', 'RS', 'RT', 'RY', 'S0', 'S4', 'S5', 'S6', 'S7', 'S8', 'SA', 'SB', 'SC', 'SF', 'SG', 'SI',
    'SJ', 'SK', 'SL', 'SM', 'SN', 'SP', 'SQ', 'SR', 'SS', 'SU', 'SV', 'SY', 'SZ', 'T5', 'T6', 'TB', 'TC', 'TF',
    'TG', 'TJ', 'TK', 'TL', 'TM', 'TO', 'TP', 'TR', 'TS', 'TU', 'TV', 'TW', 'TX', 'TY', 'TZ', 'U2', 'U4', 'U6',
    'UA', 'UD', 'UG', 'UI', 'UK', 'UL', 'UN', 'UO', 'UP', 'UQ', 'UR', 'UT', 'UU', 'UX', 'V0', 'V5', 'V7', 'V8',
    'VA', 'VB', 'VE', 'VJ', 'VM', 'VN', 'VP', 'VS', 'VT', 'VY', 'VZ', 'W1', 'W2', 'W4', 'W5', 'W6', 'W9', 'WB',
    'WF', 'WG', 'WJ', 'WK', 'WM', 'WN', 'WS', 'WT', 'WU', 'WY', 'X3', 'X4', 'XC', 'XE', 'XJ', 'XK', 'XP', 'XQ',
    'XR', 'XY', 'XZ', 'Y4', 'Y8', 'Y9', 'YB', 'YC', 'YK', 'YL', 'YN', 'YQ', 'YS', 'YT', 'Z0', 'Z2', 'ZA', 'ZE',
    'ZG', 'ZH', 'ZL', 'ZN', 'ZP', 'Y2', 'DN', 'IF', '9V', 'QL', 'FT', 'N0', 'IP', '2I', '8D', '6I', 'VQ', 'VU',
    'ZD', 'B0', '7E', 'OM', 'JR', 'ON', 'D3', 'P0', 'R4', 'IQ', 'AN', 'H8', 'RW', '6G', 'BF', 'Y7', 'U8', 'GZ',
    'A1', 'RZ', 'W3', 'BB', 'E4', '8U', '9P', 'KV', 'UM']



AIRPORTS = ['YFH', 'CGP', 'DAC', 'HOM', 'SOV', 'PGM', 'BTK', 'IKT', 'KCK',
        'LFW', 'BKO', 'ABJ', 'INB', 'CUK', 'TZA', 'PLJ', 'DGA', 'SPR',
       'BZE', 'HPN', 'BDA', 'MAD', 'RAO', 'CGH', 'POA', 'GRU', 'REC',
       'FOR', 'EVN', 'YTQ', 'YPJ', 'YVP', 'YGW', 'YUL', 'SIN', 'PEN',
       'CGK', 'HAK', 'MNL', 'BAH', 'AUH', 'CCJ', 'TRV', 'CCU', 'EIS',
       'SJU', 'STT', 'FLL', 'TPA', 'EYW', 'MCO', 'MHH', 'TLH', 'PBI',
       'ANU', 'RAK', 'AGA', 'TTU', 'LYS', 'CMN', 'BRU', 'TNG', 'CUR',
       'JED', 'TFU', 'NNG', 'LJG', 'KWL', 'XIY', 'YIC', 'GZG', 'LHW',
       'XUZ', 'SHE', 'CGQ', 'DLU', 'CAN', 'KHN', 'CSX', 'WUH', 'WXN',
       'HKG', 'KMG', 'HFE', 'CKG', 'SZX', 'BHY', 'TNA', 'KWE', 'HGH',
       'YLX', 'NGB', 'WUX', 'CTU', 'HUZ', 'LZY', 'CGO', 'TSA', 'ZUH',
       'PEK', 'LXA', 'DNH', 'NKG', 'SYX', 'ZHY', 'HRB', 'XIC', 'CZX',
       'PVG', 'MIG', 'HKT', 'KRY', 'URC', 'YNT', 'LIM', 'BOS', 'MSS',
       'YYC', 'FRA', 'PMI', 'LPA', 'WDH', 'JNB', 'CPT', 'MQP', 'DUR',
       'LUD', 'OND', 'KIM', 'PLZ', 'ERS', 'HRE', 'UTN', 'ELS', 'LUN',
       'HDS', 'DAL', 'RMO', 'ILO', 'CYZ', 'CEB', 'KLO', 'TUG', 'TAC',
       'BCD', 'TAG', 'PPS', 'GES', 'CGY', 'BXU', 'PAG', 'MFM', 'DVO',
       'BKK', 'ICN', 'LED', 'ALG', 'CCS', 'PZO', 'YXP', 'YZF', 'YFB',
       'YBB', 'YCB', 'YEG', 'YEV', 'GUA', 'FRS', 'SAP', 'FAI', 'ARC',
       'TAS', 'BFN', 'ADB', 'AYT', 'LKO', 'IXC', 'DEL', 'MAA', 'VGA',
       'HYD', 'BLR', 'IXE', 'TRZ', 'GOX', 'UDR', 'DIU', 'BOM', 'COK',
       'DIB', 'RJA', 'STV', 'HBX', 'IXM', 'HSR', 'ATQ', 'TIR', 'GOI',
       'JAI', 'AMD', 'IXB', 'IXG', 'JLR', 'IDR', 'GAU', 'PNQ', 'JDH',
       'VTZ', 'TCR', 'IXJ', 'NAG', 'BBI', 'IXL', 'RPR', 'JRH', 'BHO',
       'PAT', 'CNN', 'CJB', 'IXR', 'SAG', 'RDP', 'DED', 'VNS', 'IXZ',
       'DXB', 'AJL', 'MCT', 'DMM', 'DOH', 'TLV', 'PRG', 'KMI', 'OKA',
       'OIT', 'HND', 'KMJ', 'UKB', 'PYJ', 'OVB', 'PUS', 'CJU', 'FUK',
       'MYJ', 'CTS', 'CRK', 'NGO', 'KKJ', 'ANC', 'ANI', 'PAC', 'URS',
       'VKO', 'ARH', 'ULV', 'MRV', 'WBB', 'KLG', 'OOK', 'KGX', 'UNK',
       'LDZ', 'YIA', 'DPS', 'SMK', 'TLA', 'ORV', 'GLV', 'BKC', 'WTK',
       'KKA', 'OME', 'OTZ', 'ELI', 'ABL', 'SHH', 'PHO', 'TMS', 'JMJ',
       'DLC', 'KOW', 'JJN', 'TYN', 'YNZ', 'RGN', 'MDL', 'KUL', 'BTT',
       'NUI', 'AET', 'RBY', 'DOY', 'XMN', 'ZHA', 'ZYI', 'NNY', 'SJW',
       'PKX', 'SHA', 'HIA', 'IQN', 'GYS', 'SWA', 'SHF', 'DMK', 'FOC',
       'YIN', 'CNX', 'KUU', 'SHL', 'IXS', 'QOW', 'LOS', 'VQS', 'MVY',
       'EWB', 'ACK', 'CZH', 'PND', 'UIB', 'BGA', 'CLO', 'BOG', 'MKK',
       'OGG', 'HNL', 'STL', 'BRL', 'LUP', 'ORD', 'IAD', 'DFW', 'IPL',
       'BFD', 'ELD', 'ATH', 'KGS', 'JTR', 'KVA', 'SKG', 'TIA', 'CFU',
       'RHO', 'IST', 'LCA', 'MLA', 'MUC', 'VCE', 'STR', 'GVA', 'VNO',
       'DUS', 'NCL', 'LHR', 'CDG', 'ARN', 'HET', 'XNN', 'KHX', 'YVR',
       'YYJ', 'YXU', 'SEA', 'YYZ', 'YLW', 'YCG', 'PDX', 'YTZ', 'YQB',
       'YQU', 'YOW', 'YXT', 'YQX', 'YHZ', 'YYT', 'LGA', 'JFK', 'EWR',
       'YSJ', 'RDU', 'CLT', 'DTW', 'YWG', 'LAS', 'MIA', 'YXE', 'IAH',
       'RSW', 'AUS', 'CCC', 'MEX', 'DEN', 'PVR', 'LAX', 'FDF', 'MBJ',
       'SAN', 'SFO', 'PHX', 'BGI', 'LIS', 'MXP', 'VIE', 'CNF', 'IPN',
       'MRS', 'RNS', 'CFE', 'DUB', 'BLQ', 'BIA', 'BER', 'CPH', 'NTE',
       'FLR', 'BIQ', 'WAW', 'BUD', 'OSL', 'AGP', 'TUN', 'HEL', 'OTP',
       'LAD', 'NKC', 'CKY', 'PTP', 'CUN', 'KIX', 'AAE', 'ELU', 'HME',
       'ORN', 'TIN', 'QSF', 'LGW', 'CZL', 'DSS', 'IXA', 'IMF', 'JGA',
       'KWI', 'LGK', 'SBW', 'SDK', 'BKI']


def generate_options(items, selected=None):
    return '\n'.join(
        f'<option value="{code}"{" selected" if code == selected else ""}>{code}</option>'
        for code in items
    )

app = Flask(__name__)

MODEL_PATH = r"E:\Pycode\Thuchanhbaocao2\model\linear_regression.joblib"
ENCODER_PATH = r"E:\Pycode\Thuchanhbaocao2\model\onehot_encoder.pkl"

lr_model = load(MODEL_PATH)
ohe = load(ENCODER_PATH)

categorical_cols = ['season','carrier', 'origin', 'destination', 'year','quarter','month', 'day','day_of_week','trend_by_month']

def preprocess_input(data):
    df = pd.DataFrame([data])
    df['departure_date'] = pd.to_datetime(df['departure_date'], errors='coerce')
    if pd.isna(df['departure_date'].iloc[0]):
        raise ValueError("Ngày không hợp lệ")

    df['year'] = df['departure_date'].dt.year
    df['month'] = df['departure_date'].dt.month
    df['day'] = df['departure_date'].dt.day
    df['day_of_week'] = df['departure_date'].dt.dayofweek
    df['quarter'] = df['departure_date'].dt.quarter

    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_dayofweek'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    def get_season(m): 
        return ["winter", "spring", "summer", "autumn"][(m % 12) // 3]
    df['season'] = df['month'].apply(get_season)
    def trend_by_month(month):
        if 4 < month <= 9:
            return "on_dinh"
        elif month in [10, 11, 12]:
            return "tang"
        elif month in [1, 2]:
            return "ngang"
        else:
            return "giam"
    df['trend_by_month'] = df['month'].apply(trend_by_month)

    X_num = df[['sin_month', 'cos_month', 'sin_dayofweek', 'cos_dayofweek']]
    X_cat = ohe.transform(df[categorical_cols])
    X_cat_df = pd.DataFrame(X_cat, columns=ohe.get_feature_names_out(categorical_cols), index=df.index)

    return pd.concat([X_cat_df, X_num], axis=1)

@app.route('/')
def home():
    return render_template(
        'index.html',
        airline_options=generate_options(AIRLINES),
        origin_options=generate_options(AIRPORTS),
        destination_options=generate_options(AIRPORTS)
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict() if request.content_type != 'application/json' else request.get_json()
        required = ['carrier', 'departure_date', 'origin', 'destination']
        for f in required:
            if f not in input_data or not input_data[f]:
                raise ValueError(f"Thiếu: {f}")

        if input_data['origin'] == input_data['destination']:
            raise ValueError("Nơi đi và nơi đến không được trùng!")

        X = preprocess_input(input_data)
        pred = round(float(lr_model.predict(X)[0]), 4)

        if request.content_type == 'application/json':
            return jsonify({"prediction_gram_co2": pred, "message": "Thành công!"})
        return render_template('index.html', prediction=pred,
                               airline_options=generate_options(AIRLINES, input_data.get('carrier')),
                               origin_options=generate_options(AIRPORTS, input_data.get('origin')),
                               destination_options=generate_options(AIRPORTS, input_data.get('destination')))
    except Exception as e:
        err = str(e)
        if request.content_type == 'application/json':
            return jsonify({"error": err}), 400
        return render_template('index.html', error=err,
                               airline_options=generate_options(AIRLINES),
                               origin_options=generate_options(AIRPORTS),
                               destination_options=generate_options(AIRPORTS))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    return predict()

if __name__ == '__main__':
    app.run(debug=True)
