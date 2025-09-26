import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

year = [1992, 1993, 1994, 1995]
yVal = [293, 246, 231, 282, 301, 252, 227, 291, 304, 259, 239, 296, 306, 265, 240, 300]
n = len(yVal)

# 4-Quarter Moving Total
moving_total = [sum(yVal[i:i+4]) for i in range(n-3)]

# 4-Quarter Moving Average
moving_total_avg = [tot/4 for tot in moving_total]

# 4-Quarter Centered Moving Average
centered_moving_avg = [(moving_total_avg[i] + moving_total_avg[i+1])/2
                       for i in range(len(moving_total_avg)-1)]

# Percentage of actual to moving average
percentage = [round(yVal[i+2]/centered_moving_avg[i]*100, 3)
              for i in range(len(centered_moving_avg))]

# finding seasonal index
first, second, third, fourth = [], [], [], []
i, j, k, l = 2, 3, 0, 1
while i<len(percentage) and j<len(percentage) and k<len(percentage) and l<len(percentage):
    first.append(percentage[i])
    second.append(percentage[j])
    third.append(percentage[k])
    fourth.append(percentage[l])
    i+=4
    j+=4
    k+=4
    l+=4

quarters=[first, second, third, fourth]

# Modified sum
modified_sum = [round(sum(each)-min(each)-max(each), 3) for each in quarters]

# Modified mean
modified_mean = [each/2 for each in modified_sum]

# Adjusting factor
adj_factor = round(400/sum(modified_mean), 4)

# Seasonal indices
seasonal_indices = [round(each*adj_factor, 3) for each in modified_mean]

# Deseasonalized data
deseasonalized_data = [round(yVal[i]/(seasonal_indices[i%4]/100), 3) for i in range(len(yVal))]

# Least square regression for trend
half = n / 2
XBy2 = [(-half + 0.5) + i for i in range(n)]
X = [x*2 for x in XBy2]
XY = [round(x*y, 3) for x, y in zip(X, deseasonalized_data)]
X2 = [x**2 for x in X]
Y_mean = sum(deseasonalized_data)/n
b = sum(XY) / sum(X2)
a = Y_mean

# Cyclic variation
Y_pred = [round(a+(b*each),4) for each in X]
cyclic_variation = [round((y/y_pred)*100, 4) for y, y_pred in zip(deseasonalized_data, Y_pred)]

# ---- DataFrames instead of PrettyTable ---- #

# Table 1: First 4 steps
year_display = []
for each in year:
    year_display.append(each)
    year_display.extend(['-', '-', '-'])

table1 = pd.DataFrame({
    "Year(1)": year_display,
    "Quarter(2)": ['I', 'II', 'III', 'IV']*len(year),
    "Actual Value(3)": yVal,
    "Moving Total(4)": ['-']*2 + moving_total + ['-'],
    "Moving Average(5)=(4)/4": ['-']*2 + moving_total_avg + ['-'],
    "Centered Moving Average(6)": ['-']*2 + centered_moving_avg + ['-']*2,
    "Percentage of Actual to Moving Average(7)": ['-']*2 + percentage + ['-']*2
})
print("\nCalculation of the first 4 steps to compute seasonal index")
print(table1)

# Table 2: Steps 5 & 6
table2 = pd.DataFrame({
    "Year": year+["Modified Sum"]+["Modified Mean"],
    "Quarter 1": ['-']+quarters[0]+[modified_sum[0]]+[modified_mean[0]],
    "Quarter 2": ['-']+quarters[1]+[modified_sum[1]]+[modified_mean[1]],
    "Quarter 3": quarters[2]+['-']+[modified_sum[2]]+[modified_mean[2]],
    "Quarter 4": quarters[3]+['-']+[modified_sum[3]]+[modified_mean[3]]
})
print("\nSteps 5 and 6 in computing the seasonal index")
print(table2)

print("\nAdjusting Factor : ", adj_factor, "\n")

table3 = pd.DataFrame({
    "Quarter": ['I', 'II', 'III', 'IV'],
    "Indices": modified_mean,
    "Seasonal Indices": seasonal_indices
})
print(table3)
print("\nSum of seasonal indices : ", sum(seasonal_indices))

# Table 4: Deseasonalized data
table4 = pd.DataFrame({
    "Year (1)": year_display,
    "Quarter (2)": ['I', 'II', 'III', 'IV']*len(year),
    "Actual Value(3)": yVal,
    "Seasonal index/100 (4)": [each/100 for each in seasonal_indices]*len(year),
    "Deseasonalized data (5)": deseasonalized_data
})
print("\nCalculation of deseasonalized time series values")
print(table4)

# Table 5: Trend component
table5 = pd.DataFrame({
    "Year (1)": year_display,
    "Quarter (2)": ['I', 'II', 'III', 'IV']*len(year),
    "Y-Deseasonalized data (3)": deseasonalized_data,
    "Translating or Coding Time Var (4)": XBy2,
    "X (5)=(4)*2": X,
    "XY (6)=(5)*(3)": XY,
    "X**2 (7)": X2
})
print("\nIdentifying the trend component")
print(table5)
print(f"\nTrend line : y = {a:.3f} + {b:.3f}x\n")

# Table 6: Cyclic variation
table6 = pd.DataFrame({
    "Year (1)": year_display,
    "Quarter (2)": ['I', 'II', 'III', 'IV']*len(year),
    "Y-Deseasonalized data (3)": deseasonalized_data,
    "a + bx = Y (4)": Y_pred,
    "Percent of Trend (5)": cyclic_variation
})
print("\nIdentifying the cyclic variation")
print(table6)

# ---- Plotting ---- #
labels = []
for yr in year:
    labels += [f"{yr} Q1", f"{yr} Q2", f"{yr} Q3", f"{yr} Q4"]

plt.figure(figsize=(10, 6))
plt.plot(yVal, label="Actual data")
plt.scatter(range(len(labels)), yVal)
plt.plot(Y_pred, label="Trend line")
plt.scatter(range(len(labels)), Y_pred)
plt.plot(deseasonalized_data, label="Deseasonalized data")
plt.scatter(range(len(labels)), deseasonalized_data)
plt.plot([np.nan, np.nan] + centered_moving_avg + [np.nan, np.nan], label="Centered mov.avg")
plt.scatter(range(len(labels)), [np.nan, np.nan] + centered_moving_avg + [np.nan, np.nan])

plt.xticks(range(len(labels)), labels, rotation=45)
plt.xlabel("Year and Quarter")
plt.ylabel("Values")
plt.title("Time Series with Trend Line")
plt.legend()
plt.show()




==================================================================




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Given data as pandas Series ---
y = pd.Series([
    29,20,25,29,31,33,34,27,26,30,
    29,28,28,26,27,26,30,28,26,30,
    31,30,37,30,33,31,27,33,37,29,
    28,30,29,34,30,20,17,23,24,34,
    36,35,33,29,25,27,30,29,28,32
])

# Parameters
y_mean = y.mean()
time_lag = 25
n = len(y)

# --- ACF calculation ---
cov_list = []
for i in range(time_lag + 1):
    covariance = ((y - y_mean) * (y.shift(i) - y_mean)).sum() / n
    cov_list.append(covariance)

rho = [c / cov_list[0] for c in cov_list]  # ACF values

# --- PACF calculation ---
def calculate_pacf(y, lags, rho):
    pacf_vals = [1.0]  # PACF(0) = 1
    for k in range(1, lags+1):
        P_k = np.array([[rho[abs(i-j)] for j in range(k)] for i in range(k)])
        rho_k = np.array(rho[1:k+1])
        phi_k = np.linalg.solve(P_k, rho_k)  # Yule-Walker
        pacf_vals.append(phi_k[-1])
    return np.array(pacf_vals)

pacf_vals = calculate_pacf(y, time_lag, rho)

# --- Print ACF + PACF table ---
print(" Time Lag |    Covariance    |    Rho (ACF)   |    PACF")
print("----------|------------------|----------------|----------------")
for i in range(time_lag + 1):
    print(f"{i:<9} | {cov_list[i]:<16.6f} | {rho[i]:<14.6f} | {pacf_vals[i]:<14.6f}")

# --- Significance level ---
conf_level = 2 / np.sqrt(n)

# --- ACF significance testing ---
print("\nACF VALUES WITH SIGNIFICANCE TESTING")
print("=" * 65)
print(f"{'Lag':<6} {'ACF':<10} {'Significant?':<12} {'Decision':<40}")
print("-" * 65)

for lag, val in enumerate(rho):
    if lag == 0:
        significant, decision = "N/A", "ACF(0) = 1 (by definition)"
    else:
        if abs(val) > conf_level:
            significant, decision = "Yes", "Reject H0: Significant autocorrelation"
        else:
            significant, decision = "No", "Fail to reject H0: Not significant"
    print(f"{lag:<6} {val:<10.4f} {significant:<12} {decision:<40}")

# --- PACF significance testing ---
print("\nPACF VALUES WITH SIGNIFICANCE TESTING")
print("=" * 65)
print(f"{'Lag':<6} {'PACF':<10} {'Significant?':<12} {'Decision':<40}")
print("-" * 65)

for lag, val in enumerate(pacf_vals):
    if lag == 0:
        significant, decision = "N/A", "PACF(0) = 1 (by definition)"
    else:
        if abs(val) > conf_level:
            significant, decision = "Yes", "Reject H0: Significant partial autocorrelation"
        else:
            significant, decision = "No", "Fail to reject H0: Not significant"
    print(f"{lag:<6} {val:<10.4f} {significant:<12} {decision:<40}")

# --- Plot ACF ---
plt.figure(figsize=(8, 6))
plt.stem(range(len(rho)-1), rho[1:])
plt.axhline(0, color='black')
plt.axhline(conf_level, color='red', linestyle='--', linewidth=0.8, label='95% CI')
plt.axhline(-conf_level, color='red', linestyle='--', linewidth=0.8)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Time Lag')
plt.ylabel('ACF (rho)')
plt.legend()
plt.show()

# --- Plot PACF ---
plt.figure(figsize=(8, 6))
plt.stem(range(len(pacf_vals)), pacf_vals)
plt.axhline(0, color='black')
plt.axhline(conf_level, color='red', linestyle='--', label='95% CI')
plt.axhline(-conf_level, color='red', linestyle='--')
plt.title("Partial Autocorrelation Function (PACF)")
plt.xlabel("Lag")
plt.ylabel("PACF")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()






==================================================================




# EXAMPLE 1 (Exercise problem 4.8)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t

# Dataset
periods = np.array(list(range(1, 25)))
yt = [
    315, 195, 310, 316, 325, 335, 318, 355, 420, 410, 485, 420,
    460, 395, 390, 450, 458, 570, 520, 400, 420, 580, 475, 560
]

# -------------------------------
# First Order Exponential Smoothing
# -------------------------------
def exponential_smoothing(data, lambda_, y0):
    smoothed = np.zeros(len(data))
    smoothed[0] = y0
    for t in range(1, len(data)):
        smoothed[t] = lambda_ * data[t] + (1 - lambda_) * smoothed[t - 1]
    return smoothed

# -------------------------------
# Second Order Exponential Smoothing
# -------------------------------
def second_order_smoothing(data, lambda_, y0):
    n = len(data)
    smoothed = np.zeros(n)
    smoothed[0] = y0
    smooth_2nd = np.zeros(n)
    smooth_2nd[0] = smoothed[0]

    for t in range(1, n):
        smoothed[t] = lambda_ * data[t] + (1 - lambda_) * smoothed[t-1]
        smooth_2nd[t] = lambda_ * smoothed[t] + (1 - lambda_) * smooth_2nd[t-1]

    # Final forecast values (double smoothed)
    final = 2 * smoothed - smooth_2nd
    return smoothed, smooth_2nd, final

# Parameters
lambda_02 = 0.2
lambda_04 = 0.4

smoothed_02 = exponential_smoothing(yt, lambda_02, yt[0])
smoothed_04 = exponential_smoothing(yt, lambda_04, yt[0])

sm1_02, sm2_02, final_02 = second_order_smoothing(yt, lambda_02, yt[0])
sm1_04, sm2_04, final_04 = second_order_smoothing(yt, lambda_04, yt[0])

# -------------------------------
# Data Table
# -------------------------------
data = {
    'Period': periods,
    'Original': yt,
    'Smoothed (λ=0.2)': smoothed_02,
    'Smoothed (λ=0.4)': smoothed_04,
    '2nd Order Final (λ=0.2)': final_02,
    '2nd Order Final (λ=0.4)': final_04
}
df = pd.DataFrame(data)
print(df)

# -------------------------------
# Plots
# -------------------------------
# First-order smoothing
plt.plot(periods, yt, label='Original Data', marker='o', color='black')
plt.plot(periods, smoothed_02, label='1st Order (λ=0.2)', marker='o', color='blue')
plt.plot(periods, smoothed_04, label='1st Order (λ=0.4)', marker='o', color='red')
plt.xlabel('Period')
plt.ylabel('yt')
plt.title('First Order Exponential Smoothing')
plt.legend()
plt.grid(True)
plt.xticks(periods)
plt.tight_layout()
plt.show()

# Second-order smoothing (Final)
plt.plot(periods, yt, label='Original Data', marker='o', color='black')
plt.plot(periods, final_02, label='2nd Order Final (λ=0.2)', marker='o', color='blue')
plt.plot(periods, final_04, label='2nd Order Final (λ=0.4)', marker='o', color='red')
plt.xlabel('Period')
plt.ylabel('yt')
plt.title('Second Order Exponential Smoothing (Final)')
plt.legend()
plt.grid(True)
plt.xticks(periods)
plt.tight_layout()
plt.show()

# -------------------------------
# T-Tests for 1st and 2nd Order Smoothing
# -------------------------------
def perform_ttest(original, smoothed, label, alpha=0.05):
    d = original - smoothed
    d_mean = np.mean(d)
    d_std = np.std(d, ddof=1)
    n = len(d)
    t_stat = d_mean / (d_std / np.sqrt(n))
    p_val = 2 * t.sf(np.abs(t_stat), df=n-1)

    print(f"\nTest for {label}")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
    if p_val < alpha:
        print(f"Reject H0 at α={alpha}: Significant difference between original and {label}.")
    else:
        print(f"Fail to Reject H0 at α={alpha}: No significant difference between original and {label}.")

# First Order Tests
perform_ttest(yt, smoothed_02, "1st Order (λ=0.2)")
perform_ttest(yt, smoothed_04, "1st Order (λ=0.4)")

# Second Order Tests (Final series)
perform_ttest(yt, final_02, "2nd Order Final (λ=0.2)")
perform_ttest(yt, final_04, "2nd Order Final (λ=0.4)")





============================================================





import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# -----------------------------
# Fetch Tesla Data
# -----------------------------

start_date = dt.datetime(2024, 1, 1)
end_date   = dt.datetime(2024, 12, 31)

tesla = yf.Ticker("TSLA")
data = tesla.history(start=start_date, end=end_date)
data["Rate of Return"] = (data["Close"] - data["Open"]) / data["Open"] * 100

data = data.sample(n=150, random_state=42).sort_index()
data["DOY"] = data.index.dayofyear

# -----------------------------
# Helper Functions
# -----------------------------
def coded_variable(n):
    if n % 2 == 0:
        return np.arange(-n+1, n, 2)[:n].astype(float)
    else:
        return np.arange(-(n//2), n//2 + 1).astype(float)

def fit_linear(y, x):
    a = np.mean(y)
    b = np.sum(x*y) / np.sum(x**2)
    y_pred = a + b*x
    return (a, b), y_pred

def fit_quadratic(y, x):
    n = len(y)
    sum_x2 = np.sum(x**2)
    sum_x4 = np.sum(x**4)
    sum_y  = np.sum(y)
    sum_x2y = np.sum(x**2 * y)

    b = np.sum(x*y) / sum_x2
    A = np.array([[n, sum_x2],[sum_x2, sum_x4]])
    B = np.array([sum_y, sum_x2y])
    a, c = np.linalg.solve(A, B)

    y_pred = a + b*x + c*(x**2)
    return (a, b, c), y_pred

def fit_cubic(y, x):
    X = np.vstack([np.ones_like(x), x, x**2, x**3]).T
    coeffs = np.linalg.inv(X.T @ X) @ (X.T @ y)
    y_pred = X @ coeffs
    return tuple(coeffs), y_pred

def error_analysis(y, y_pred):
    resid = y - y_pred
    rmse = np.sqrt(np.mean(resid**2))
    mape = np.mean(np.abs(resid / y)) * 100
    r2 = 1 - np.sum(resid**2) / np.sum((y - np.mean(y))**2)
    return rmse, mape, r2

# -----------------------------
# Variables and Trends
# -----------------------------
variables = {
    "Opening Stock": data["Open"].values,
    "Closing Stock": data["Close"].values,
    "Rate of Return": data["Rate of Return"].values
}

trend_types = ["Linear", "Quadratic", "Cubic"]
results_table = []

# -----------------------------
# Plot (all three regressions together)
# -----------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 12))
fig.suptitle("Secular Trend Fits (All Regressions Together)", fontsize=16)

for row, (var_name, y) in enumerate(variables.items()):
    x = coded_variable(len(y))
    ax = axes[row]
    ax.plot(range(len(y)), y, marker="o", markersize=3, color="black", label="Actual")

    # Fit all trends
    for trend in trend_types:
        if trend == "Linear":
            coeffs, y_pred = fit_linear(y, x)
        elif trend == "Quadratic":
            coeffs, y_pred = fit_quadratic(y, x)
        else:
            coeffs, y_pred = fit_cubic(y, x)

        rmse, mape, r2 = error_analysis(y, y_pred)
        results_table.append({
            "Variable": var_name,
            "Trend": trend,
            "Coefficients": np.round(coeffs, 4),
            "RMSE": round(rmse, 4),
        })

        # Add regression line
        ax.plot(range(len(y)), y_pred, lw=1.5, label=f"{trend} Fit")

    ax.set_title(var_name)
    ax.set_xlabel("Observation Index")
    ax.set_ylabel(var_name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# -----------------------------
# Results Table
# -----------------------------
results_df = pd.DataFrame(results_table)
print("Secular Trend Results:")
print(results_df)








===========================================================================





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def auto_corr_func(data, lags):
    n = len(data['ROR'])
    c_k = []
    Y = np.mean(data['ROR'])
    for k in range(lags+1):
        sum_product = 0
        for t in range(n - k):
            p1 = data['ROR'][t] - Y
            p2 = data['ROR'][t + k] - Y
            sum_product += p1 * p2
        c_k.append(sum_product / n)

    P_k = np.array(c_k[1:]) / c_k[0] if c_k[0] != 0 else np.zeros(len(c_k) - 1)
    return P_k

# ===================== LOAD DATA =====================
fp = "/content/drive/MyDrive/Sem - 9/Mathematical Modelling/PS6/data_bajaj.csv"
data = pd.read_csv(fp)

# Calculate ROR
ror = ((data['Close Price'] - data['Open Price']) / data['Open Price']) * 100
data['ROR'] = ror
print(data.head())

# ===================== AUTOCORRELATION =====================
auto = auto_corr_func(data, 25)
print(auto)

n1 = len(auto)
v = 2 / np.sqrt(n1)
print("Significance threshold (±v):", v)

# Plotting ACF
plt.figure(figsize=(8, 8))
plt.stem(auto)
plt.axhline(v, color='red', linestyle='--')
plt.axhline(-v, color='red', linestyle='--')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title('Auto Correlation Function')
plt.show()

# Significance table
for lag, val in enumerate(auto):
    if lag == 0:
        significant = "N/A"
        decision = "PACF(0) = 1 (by definition)"
    else:
        if abs(val) > v:
            significant = "Yes"
            decision = "Reject H0: Significant autocorrelation"
        else:
            significant = "No"
            decision = "Fail to reject H0: Not significant"
    print(f"{lag:<6} {val:<10.4f} {significant:<12} {decision:<40}")

# ===================== DURBIN-WATSON =====================
# Residuals = deviation from mean
residuals = data['ROR'] - np.mean(data['ROR'])
diff_res = np.diff(residuals)  # e_t - e_{t-1}
dw_stat = np.sum(diff_res**2) / np.sum(residuals**2)
print("\nDurbin-Watson statistic:", dw_stat)

if dw_stat < 2:
    print("Positive autocorrelation")
elif dw_stat > 2:
    print("Negative autocorrelation")
else:
    print("No autocorrelation")







============================================================================================



import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, solve
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.weightstats import ztest

x = [i for i in range(1, 51)]
y = [
    29,20,25,29,31,33,34,27,26,30,
    29,28,28,26,27,26,30,28,26,30,
    31,30,37,30,33,31,27,33,37,29,
    28,30,29,34,30,20,17,23,24,34,
    36,35,33,29,25,27,30,29,28,32
]
n = len(y)

x_2 = [each**2 for each in x]
x_3 = [each**3 for each in x]
x_4 = [each**4 for each in x]
x_5 = [each**5 for each in x]
x_6 = [each**6 for each in x]
xy = [val*y[i] for i, val in enumerate(x)]
x2y = [val*y[i] for i, val in enumerate(x_2)]
x3y = [val*y[i] for i, val in enumerate(x_3)]
x4y = [val*y[i] for i, val in enumerate(x_4)]

lin_b = ((n*sum(xy)) - (sum(x)*sum(y)))/((n*sum(x_2)) - (sum(x)**2))
lin_a = np.mean(y) - (lin_b * np.mean(x))
y_lin = [lin_a+(lin_b*each) for each in x]

a, b, c = symbols('a b c')
eq1 = n*a + sum(x)*b + sum(x_2)*c - sum(y)
eq2 = sum(x)*a + sum(x_2)*b + sum(x_3)*c - sum(xy)
eq3 = sum(x_2)*a + sum(x_3)*b + sum(x_4)*c - sum(x2y)
sols = solve((eq1, eq2, eq3), (a, b, c))
quad_a = float(sols[a])
quad_b = float(sols[b])
quad_c = float(sols[c])
y_quad = [quad_a+(quad_b*each)+(quad_c*(each**2)) for each in x]

a, b, c, d = symbols('a b c d')
eq1 = n*a + sum(x)*b + sum(x_2)*c + sum(x_3)*d - sum(y)
eq2 = sum(x)*a + sum(x_2)*b + sum(x_3)*c + sum(x_4)*d - sum(xy)
eq3 = sum(x_2)*a + sum(x_3)*b + sum(x_4)*c + sum(x_5)*d - sum(x2y)
eq4 = sum(x_3)*a + sum(x_4)*b + sum(x_5)*c + sum(x_6)*d - sum(x3y)
sols1 = solve((eq1, eq2, eq3, eq4), (a, b, c, d))
cubic_a = float(sols1[a])
cubic_c = float(sols1[c])
cubic_b = float(sols1[b])
cubic_d = float(sols1[d])

y_cubic = [cubic_a+(cubic_b*each)+(cubic_c*(each**2))+(cubic_d*(each**3)) for each in x]

df = pd.DataFrame({
    "X" : x,
    "codedx":x,
    "linear":y_lin,
    "quadratic":y_quad,
    "cubic":y_cubic
    })
             
print(df)       

plt.plot(x, y_lin, color='red', label='Linear Regression')
plt.plot(x, y_quad, color='green', label='Quadratic Regression')
plt.plot(x, y_cubic, color='blue', label='Cubic Regression')                                  
plt.legend()
plt.show()

# Convert predictions to float numpy arrays
y_lin_arr = np.array(y_lin, dtype=float)
y_quad_arr = np.array(y_quad, dtype=float)
y_cubic_arr = np.array(y_cubic, dtype=float)
y_arr = np.array(y, dtype=float)

# t-test Linear vs Actual
t_stat, p_val = ttest_ind(y_lin_arr, y_arr, equal_var=False)
print(f"\nLinear vs Actual -> t = {t_stat:.5f}, p = {p_val:.5f}")
print("Result:", "Significant difference" if p_val < 0.05 else "No significant difference")

# t-test Quadratic vs Actual
t_stat1, p_val1 = ttest_ind(y_quad_arr, y_arr, equal_var=False)
print(f"\nQuadratic vs Actual -> t = {t_stat1:.5f}, p = {p_val1:.5f}")
print("Result:", "Significant difference" if p_val1 < 0.05 else "No significant difference")

# t-test Cubic vs Actual
t_stat2, p_val2 = ttest_ind(y_cubic_arr, y_arr, equal_var=False)
print(f"\nCubic vs Actual -> t = {t_stat2:.5f}, p = {p_val2:.5f}")
print("Result:", "Significant difference" if p_val2 < 0.05 else "No significant difference")

# Convert predictions and actuals to arrays
y_lin_arr = np.array(y_lin, dtype=float)
y_quad_arr = np.array(y_quad, dtype=float)
y_cubic_arr = np.array(y_cubic, dtype=float)
y_arr = np.array(y, dtype=float)

# One-sample z-test: test if mean(predicted - actual) = 0
# Linear
z_stat_lin, p_val_lin = ztest(y_lin_arr, y_arr)
print(f"\nLinear vs Actual -> z = {z_stat_lin:.5f}, p = {p_val_lin:.5f}")
print("Result:", "Significant difference" if p_val_lin < 0.05 else "No significant difference")

# Quadratic
z_stat_quad, p_val_quad = ztest(y_quad_arr, y_arr)
print(f"\nQuadratic vs Actual -> z = {z_stat_quad:.5f}, p = {p_val_quad:.5f}")
print("Result:", "Significant difference" if p_val_quad < 0.05 else "No significant difference")

# Cubic
z_stat_cubic, p_val_cubic = ztest(y_cubic_arr, y_arr)
print(f"\nCubic vs Actual -> z = {z_stat_cubic:.5f}, p = {p_val_cubic:.5f}")
print("Result:", "Significant difference" if p_val_cubic < 0.05 else "No significant difference")

                                                      


===============================================================



import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, solve
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import t
from statsmodels.stats.weightstats import ztest

x = [i for i in range(1, 51)]
y = [
    29,20,25,29,31,33,34,27,26,30,
    29,28,28,26,27,26,30,28,26,30,
    31,30,37,30,33,31,27,33,37,29,
    28,30,29,34,30,20,17,23,24,34,
    36,35,33,29,25,27,30,29,28,32
]
n = len(y)
x_coded = [2*int(val-np.mean(x)) for val in x] if len(x)%2==0 else [int(val-np.mean(x)) for val in x]

x_2 = [each**2 for each in x_coded]
x_3 = [each**3 for each in x_coded]
x_4 = [each**4 for each in x_coded]
x_5 = [each**5 for each in x_coded]
x_6 = [each**6 for each in x_coded]
xy = [val*y[i] for i, val in enumerate(x_coded)]
x2y = [val*y[i] for i, val in enumerate(x_2)]
x3y = [val*y[i] for i, val in enumerate(x_3)]
x4y = [val*y[i] for i, val in enumerate(x_4)]

lin_a = np.mean(y)
lin_b = sum(xy)/sum(x_2)
y_lin = [lin_a+(lin_b*each) for each in x_coded]

a, c = symbols('a c')
quad_b = sum(xy)/sum(x_2)
eq1 = n*a + sum(x_2)*c - sum(y)
eq2 = sum(x_2)*a + sum(x_4)*c - sum(x2y)
sols = solve((eq1, eq2), (a, c))
quad_a = sols[a]
quad_c = sols[c]
y_quad = [quad_a+(quad_b*each)+(quad_c*(each**2)) for each in x_coded]

a, b, c, d = symbols('a b c d')
eq1 = n*a + sum(x_2)*c - sum(y)
eq2 = sum(x_2)*a + sum(x_4)*c - sum(x2y)
eq3 = sum(x_2)*b + sum(x_4)*d - sum(xy)
eq4 = sum(x_4)*b + sum(x_6)*d - sum(x3y)
sols1 = solve((eq1, eq2), (a, c))
sols2 = solve((eq3, eq4), (b, d))
cubic_a = float(sols1[a])
cubic_c = float(sols1[c])
cubic_b = float(sols2[b])
cubic_d = float(sols2[d])

y_cubic = [cubic_a+(cubic_b*each)+(cubic_c*(each**2))+(cubic_d*(each**3)) for each in x_coded]

df = pd.DataFrame({
    "X" : x,
    "codedx":x_coded,
    "linear":y_lin,
    "quadratic":y_quad,
    "cubic":y_cubic
    })
             
print(df)       

plt.plot(x_coded, y_lin, color='red', label='Linear Regression')
plt.plot(x_coded, y_quad, color='green', label='Quadratic Regression')
plt.plot(x_coded, y_cubic, color='blue', label='Cubic Regression')                                  
plt.legend()
plt.show()
                                         
# Convert predictions to float numpy arrays
y_lin_arr = np.array(y_lin, dtype=float)
y_quad_arr = np.array(y_quad, dtype=float)
y_cubic_arr = np.array(y_cubic, dtype=float)
y_arr = np.array(y, dtype=float)

# t-test Linear vs Actual
t_stat, p_val = ttest_ind(y_lin_arr, y_arr)
print(f"\nLinear vs Actual -> t = {t_stat:.5f}, p = {p_val:.5f}")
print("Result:", "Significant difference" if p_val < 0.05 else "No significant difference")

# t-test Quadratic vs Actual
t_stat1, p_val1 = ttest_ind(y_quad_arr, y_arr)
print(f"\nQuadratic vs Actual -> t = {t_stat1:.5f}, p = {p_val1:.5f}")
print("Result:", "Significant difference" if p_val1 < 0.05 else "No significant difference")

# t-test Cubic vs Actual
t_stat2, p_val2 = ttest_ind(y_cubic_arr, y_arr)
print(f"\nCubic vs Actual -> t = {t_stat2:.5f}, p = {p_val2:.5f}")
print("Result:", "Significant difference" if p_val2 < 0.05 else "No significant difference")
  

# Convert predictions and actuals to arrays
y_lin_arr = np.array(y_lin, dtype=float)
y_quad_arr = np.array(y_quad, dtype=float)
y_cubic_arr = np.array(y_cubic, dtype=float)
y_arr = np.array(y, dtype=float)

# One-sample z-test: test if mean(predicted - actual) = 0
# Linear
z_stat_lin, p_val_lin = ztest(y_lin_arr, y_arr)
print(f"\nLinear vs Actual -> z = {z_stat_lin:.5f}, p = {p_val_lin:.5f}")
print("Result:", "Significant difference" if p_val_lin < 0.05 else "No significant difference")

# Quadratic
z_stat_quad, p_val_quad = ztest(y_quad_arr, y_arr)
print(f"\nQuadratic vs Actual -> z = {z_stat_quad:.5f}, p = {p_val_quad:.5f}")
print("Result:", "Significant difference" if p_val_quad < 0.05 else "No significant difference")

# Cubic
z_stat_cubic, p_val_cubic = ztest(y_cubic_arr, y_arr)
print(f"\nCubic vs Actual -> z = {z_stat_cubic:.5f}, p = {p_val_cubic:.5f}")
print("Result:", "Significant difference" if p_val_cubic < 0.05 else "No significant difference")


