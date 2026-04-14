import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Cấu hình hiển thị
plt.rcParams['figure.figsize'] = [12, 6]

def smart_read(file_name):
    df = pd.read_csv(file_name)
    # Tự động tìm cột có chữ 'date' hoặc 'time' (không phân biệt hoa thường)
    date_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_col:
        df[date_col[0]] = pd.to_datetime(df[date_col[0]])
        df.set_index(date_col[0], inplace=True)
    else:
        # Nếu không tìm thấy, ép cột đầu tiên làm index
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
    return df

# --- BÀI 1: DOANH THU SIÊU THỊ ---
print("--- Đang xử lý Bài 1 ---")
df1 = smart_read('ITA105_Lab_5_Supermarket.csv')
df1.iloc[:, 0] = df1.iloc[:, 0].fillna(method='ffill') # Cột đầu tiên là doanh thu
df1['rolling_mean'] = df1.iloc[:, 0].rolling(window=7).mean()
df1[[df1.columns[0], 'rolling_mean']].plot(title="Bai 1: Doanh thu Sieu thi")
plt.show()

# --- BÀI 2: LƯU LƯỢNG TRUY CẬP WEBSITE ---
print("--- Đang xử lý Bài 2 ---")
df2 = smart_read('ITA105_Lab_5_Web_traffic.csv')
df2 = df2.asfreq('H').interpolate()
df2['hour'] = df2.index.hour
sns.lineplot(data=df2, x='hour', y=df2.columns[0])
plt.title("Bai 2: Luu luong theo gio")
plt.show()

# --- BÀI 3: GIÁ CỔ PHIẾU ---
print("--- Đang xử lý Bài 3 ---")
df3 = smart_read('ITA105_Lab_5_Stock.csv')
df3.iloc[:, 0] = df3.iloc[:, 0].fillna(method='ffill')
df3['MA7'] = df3.iloc[:, 0].rolling(7).mean()
df3['MA30'] = df3.iloc[:, 0].rolling(30).mean()
df3[[df3.columns[0], 'MA7', 'MA30']].plot(title="Bai 3: Gia Co phieu")
plt.show()

# --- BÀI 4: SẢN XUẤT CÔNG NGHIỆP ---
print("--- Đang xử lý Bài 4 ---")
df4 = smart_read('ITA105_Lab_5_Production.csv')
data_clean = df4.iloc[:, 0].fillna(method='ffill')
result = seasonal_decompose(data_clean, model='additive', period=12)
result.plot()
plt.show()