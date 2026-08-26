# =============================================================================
#   MACHINE LEARNING - Complete Notes aur Code
# =============================================================================

# ===========================================================================
# SECTION 1: MACHINE LEARNING KYA HAI?
# ===========================================================================
"""
================================================================================
                        MACHINE LEARNING (ML)
================================================================================

MACHINE LEARNING KYA HAI?
--------------------------
Machine Learning, Artificial Intelligence (AI) ka ek hissa hai jis mein
computer ko is tarah tayyar kiya jata hai keh woh data dekh kar khud seekh
sake aur baghair sarahatan program kiye, tajurbay se behtar hota jaye.

Sada alfaz mein:
  Jaise bacha baar baar gir kar chalna seekhta hai, waise hi ML mein
  computer data dekh kar patterns pehchaanna seekhta hai.

ML KE 3 AHEM HISSAY
--------------------
  1. DATA      -> Computer ko sikhane ka mawad
  2. ALGORITHM -> Riyazi ke usool jo data se seekhne mein madad karte hain
  3. MODEL     -> Woh nateeja jo training ke baad predictions karta hai

ML KYU ISTEMAL KAREN?
----------------------
  - Bade data se patterns dhoondne ke liye
  - Insani ghaltiyan kam karne ke liye
  - Khud-kaar faislay karne ke liye (spam detection, loan approval waghera)
"""

# ===========================================================================
# SECTION 2: TYPES OF MACHINE LEARNING - AQSAAM
# ===========================================================================
"""
================================================================================
                     ML KI 4 AHEM AQSAAM
================================================================================

--------------------------------------------------------------------------
1. SUPERVISED LEARNING (Nigrani wali seekh)
--------------------------------------------------------------------------
   Definition: Computer ko labeled data diya jata hai yani sawal aur jawab
               dono diye jate hain. Computer is se pattern seekh kar naye
               data ka jawab deta hai.

   Misaal:
     -> Aap ke paas gharon ki qeematein aur unki size ka data hai
     -> Model seekhta hai: bara ghar = zyada qeemat
     -> Phir naye ghar ki qeemat predict karta hai

   Aqsaam:
     a) REGRESSION      -> Number predict kare (masalan ghar ki qeemat, tankhwah)
     b) CLASSIFICATION  -> Category predict kare (spam/not spam, loan approved/rejected)

   Haqeeqi misaalein:
     - Email spam detection
     - House price prediction
     - Loan approval/rejection

--------------------------------------------------------------------------
2. UNSUPERVISED LEARNING (Baghair nigrani ke seekh)
--------------------------------------------------------------------------
   Definition: Computer ko sirf data diya jata hai, koi label (jawab) nahi
               hota. Computer khud hi data mein similarities aur patterns
               dhoondta hai.

   Misaal:
     -> Aap ke paas customers ka khareedari data hai
     -> Model khud mukhtalif groups bana leta hai (saste pasand karne
        wale, mehnge pasand karne wale)
     -> Aap ne koi label nahi diya, model ne khud seekha

   Aqsaam:
     a) CLUSTERING   -> Data ko groups mein taqseem karna
     b) ASSOCIATION  -> Relationships dhoondna (jo X khareede woh Y bhi
        khareedta hai)

   Haqeeqi misaalein:
     - Netflix ki recommendations
     - Customer segmentation
     - Market basket analysis

--------------------------------------------------------------------------
3. SEMI-SUPERVISED LEARNING
--------------------------------------------------------------------------
   Definition: Thora labeled data + bohat zyada unlabeled data. Pehle
               labeled se seekhta hai, phir unlabeled ko label karta hai.

   Misaal: 100 tasveeron mein 10 labeled (billi/kutta) baaqi 90 unlabeled.
           Model pehle 10 se seekhta hai phir 90 khud label karta hai.

   Aqsaam:
     a) Generative Models -> Unlabeled data se naye samples banata hai
     b) Self Training     -> Model khud labels karta hai aur dobara
        train hota hai

--------------------------------------------------------------------------
4. REINFORCEMENT LEARNING (Inaam wali seekh)
--------------------------------------------------------------------------
   Definition: Ek Agent hai jo Environment mein kaam karta hai. Sahi kaam
               par Reward milta hai, ghalat par Penalty. Agent khud
               seekhta hai keh kya karna chahiye.

   3 ahem hissay:
     1. Agent          -> Woh jo seekh raha hai (jaise robot ya insaan)
     2. Environment     -> Woh duniya/mahol jis mein kaam karta hai
     3. Reward/Penalty  -> Sahi kaam par inaam, ghalat par saza

   Misaal: Video game khelna seekhna - har sahi move par points milte hain

--------------------------------------------------------------------------
UNSUPERVISED KI TAFSEEL - Clustering & Association
--------------------------------------------------------------------------

  CLUSTERING:
    -> Data ko mukhtalif groups mein taqseem karna
    -> Algorithm khud groups banata hai data ki similarity ki bunyad par
    -> Misaal: Customers ko 3 groups mein baantna

  ASSOCIATION:
    -> Data mein relationships dhoondna
    -> Misaal: Jo roti khareede woh makhan bhi khareedta hai
    -> 100 mein se kitne customers ne yeh pattern follow kiya?
"""

# ===========================================================================
# SECTION 3: LINEAR REGRESSION KYA HAI?
# ===========================================================================
"""
================================================================================
                        LINEAR REGRESSION
================================================================================

LINEAR REGRESSION KYA HAI?
----------------------------
Linear Regression ek Supervised ML algorithm hai jo continuous/numerical
values predict karta hai. Yeh data mein ek seedhi lakeer (line) dhoondta
hai jo best fit ho.

FORMULA
-------
  y = mx + c

  Jahan:
    y = output/target (jo predict karna hai, masalan Package)
    x = input/feature (jo hamare paas hai, masalan CGPA)
    m = slope/coefficient (CGPA kitna Package barhata hai)
    c = intercept/constant (bunyadi raqam jab x=0 ho)

KYU ISTEMAL KAREN?
-------------------
  - Jab output ek number ho (qeemat, tankhwah, number)
  - Jab input aur output ka relationship linear ho
  - Aasan aur samajhne mein sab se aasan algorithm

Misaal:
  CGPA 3.4 hai -> Package kya hoga?
  Model: y = 0.5839 * 3.4 + c

STEPS YAAD RAKHEN
------------------
  1. Libraries import karen
  2. Data load karen (CSV)
  3. X (input) aur y (output) alag karen
  4. Train/Test split karen (80/20)
  5. Model banayen (LinearRegression)
  6. Model train karen (fit)
  7. Predictions nikalen (predict)
"""

# ===========================================================================
# SECTION 3A: LINEAR REGRESSION CODE - Jobs Dataset
# ===========================================================================
print("=" * 60)
print("    LINEAR REGRESSION - JOBS DATASET (CGPA -> Package)")
print("=" * 60)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- Step 1: Data Load ---
# Note: Aap ke paas Jobs.csv hona chahiye isi folder mein
# jobs = pd.read_csv('Jobs.csv')
# jobs.info()  # Data ki maloomat dekhne ke liye

# Demo ke liye synthetic data banate hain
np.random.seed(42)
n = 200
cgpa = np.random.uniform(2.0, 4.0, n)
package = 0.5839 * cgpa + 0.5 + np.random.normal(0, 0.3, n)
jobs = pd.DataFrame({'cgpa': cgpa, 'package': package})
print("\nData Sample:")
print(jobs.head())
print(f"\nData Shape: {jobs.shape}")

# --- Step 2: Features aur Target alag karen ---
X = jobs[['cgpa']]    # Input feature (2D array chahiye)
y = jobs[['package']] # Output/Target

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

# --- Step 3: Train/Test Split (80% Train, 20% Test) ---
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {x_train.shape}")
print(f"Test size:  {x_test.shape}")

# --- Step 4: Model banayen ---
lr = LinearRegression()

# --- Step 5: Model Train karen ---
lr.fit(x_train, y_train)
print("\nModel Successfully Trained!")

# --- Step 6: Predictions ---
y_pred = lr.predict(x_test)
print("\nFirst 5 Predictions vs Actual:")
for i in range(5):
    print(f"  CGPA: {x_test.iloc[i,0]:.2f} | Predicted: {y_pred[i,0]:.4f} | Actual: {y_test.iloc[i,0]:.2f}")

# --- Step 7: Coefficient aur Intercept ---
m = lr.coef_[0][0]   # slope
c = lr.intercept_[0] # intercept
print(f"\nFormula: y = {m:.4f}x + {c:.4f}")
print(f"Yani: Package = {m:.4f} x CGPA + {c:.4f}")

# --- Step 8: Nayi CGPA ke liye Prediction ---
cgpa_input = 3.4
a = np.array([[cgpa_input]])
point = lr.predict(a)
print(f"\nCGPA {cgpa_input} ke liye Package: {point[0][0]:.4f}")

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.scatter(x_test, y_test, color='blue', alpha=0.5, label='Actual Data')
plt.plot(x_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('CGPA')
plt.ylabel('Package')
plt.title('Linear Regression: CGPA vs Package')
plt.legend()
plt.tight_layout()
plt.savefig('linear_regression_plot.png', dpi=100)
plt.show()
print("\nPlot saved as 'linear_regression_plot.png'")


# ===========================================================================
# SECTION 4: EVALUATION METRICS KYA HAIN? - MAAP KE AUZAAR
# ===========================================================================
"""
================================================================================
                        EVALUATION METRICS
================================================================================

EVALUATION METRICS KYA HAIN?
------------------------------
Evaluation Metrics woh maap hain jin se hum dekhte hain keh hamara Model
kitna acha ya bura kaam kar raha hai.

Yeh y_test (actual) aur y_pred (predicted) ke darmiyan farq naapte hain.

--------------------------------------------------------------------------
1. MAE - Mean Absolute Error (Ausat mutlaq ghalti)
--------------------------------------------------------------------------
   Formula: MAE = Average of |actual - predicted|

   Kya karta hai: Har ghalti ka ausat nikalta hai (baghair square kiye)

   Misaal: 3 predictions
     Actual:    2, 4, 6    -> Mean = 4
     Predicted: 2.5, 3.5, 5.5
     Errors:    0.5, 0.5, 0.5
     MAE = 0.5

   Kab istemal: Jab badi ghaltiyan aur choti ghaltiyan barabar ahem hon

--------------------------------------------------------------------------
2. MSE - Mean Squared Error (Ausat murabba ghalti)
--------------------------------------------------------------------------
   Formula: MSE = Average of (actual - predicted)^2

   Kya karta hai: Ghaltiyon ko square kar ke ausat nikalta hai. Badi
   ghaltiyon ko zyada penalty deta hai

   Kab istemal: Jab badi ghaltiyan sangeen hon

--------------------------------------------------------------------------
3. RMSE - Root Mean Squared Error
--------------------------------------------------------------------------
   Formula: RMSE = sqrt(MSE)

   Kya karta hai: MSE ka square root nikalta hai. Asal unit mein wapis
   laata hai (samajhna aasan)

--------------------------------------------------------------------------
4. R2 SCORE (R-Squared) - Model ki taaqat
--------------------------------------------------------------------------
   Range: 0 se 1 (ya kabhi manfi)
   1 ke qareeb = behtareen model
   0 ke qareeb = bekaar model

--------------------------------------------------------------------------
SST, SSE, SSR KYA HAIN?
--------------------------------------------------------------------------
   SST (Total Sum of Squares):
     -> y ka mean se kul farq
     -> SST = sum(y - y_mean)^2
     -> Misaal: y = [2,4,6], mean=4
       SST = (2-4)^2 + (4-4)^2 + (6-4)^2 = 4+0+4 = 8

   SSE (Sum of Squared Errors):
     -> Actual aur predicted ka farq
     -> Model kitna ghalat hai

   SSR (Sum of Squared Regression):
     -> Model ne kitni variation explain ki

   R2 = 1 - (SSE/SST)   ya   R2 = SSR/SST
"""

print("\n" + "=" * 60)
print("    EVALUATION METRICS")
print("=" * 60)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# SST misaal (apke notes se)
y_example = np.array([2, 4, 6])
y_mean = np.mean(y_example)
sst = np.sum((y_example - y_mean) ** 2)
print(f"\nSST misaal: y=[2,4,6], mean={y_mean}")
print(f"SST = (2-{y_mean})^2 + (4-{y_mean})^2 + (6-{y_mean})^2 = {sst}")

# Actual Metrics on our model
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\nLinear Regression Model ki Performance:")
print(f"  MAE  (Mean Absolute Error)  : {mae:.4f}")
print(f"  MSE  (Mean Squared Error)   : {mse:.4f}")
print(f"  RMSE (Root MSE)             : {rmse:.4f}")
print(f"  R2   (Accuracy/Score)       : {r2:.4f}")
print(f"\n  R2={r2:.2f} matlab Model ne {r2*100:.1f}% variation explain ki")


# ===========================================================================
# SECTION 5: LABEL ENCODING vs ONE-HOT ENCODING - FARQ KYA HAI?
# ===========================================================================
"""
================================================================================
               LABEL ENCODING vs ONE-HOT ENCODING
================================================================================

Masla: ML models sirf numbers samajhte hain, text nahi
Hal: Categorical data ko numbers mein badalna

--------------------------------------------------------------------------
1. LABEL ENCODING
--------------------------------------------------------------------------
   Har category ko ek number do

   Kab istemal: Jab order/tarteeb ho (Ordinal data)
   Misaal:
     Size: Small=0, Medium=1, Large=2  <- Tarteeb maujood hai

   Misaalein (apke notes se):
     Customer Satisfaction:
       Unsatisfied=0, Neutral=1, Satisfied=2, Very Satisfied=3

     Education Level:
       Primary=0, High School=1, Bachelor's=2, Master's=3, PhD=4

     Temperature:
       Low=-1, Medium=-2, High=3

--------------------------------------------------------------------------
2. ONE-HOT ENCODING
--------------------------------------------------------------------------
   Har category ke liye nayi column banao (0 ya 1)

   Kab istemal: Jab koi order na ho (Nominal data)
   Misaal:
     Color: Red, Green, Blue <- Koi tarteeb nahi

     Color | Red | Green | Blue
     Red   |  1  |   0   |  0
     Green |  0  |   1   |  0
     Blue  |  0  |   0   |  1

   Farq:
     Label   -> Tarteeb wale data ke liye (kam columns)
     One-Hot -> Baghair tarteeb wale data ke liye (zyada columns, lekin
                ghalat order nahi)
"""

print("\n" + "=" * 60)
print("    LABEL ENCODING vs ONE-HOT ENCODING")
print("=" * 60)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# --- Label Encoding misaal ---
print("\n1. LABEL ENCODING:")
satisfaction = ['Unsatisfied', 'Neutral', 'Satisfied', 'Very Satisfied',
                'Satisfied', 'Unsatisfied']
le = LabelEncoder()
encoded = le.fit_transform(satisfaction)
print("   Original:", satisfaction[:4])
print("   Encoded: ", encoded[:4].tolist())

# Manual Mapping (apke notes ke mutabiq)
satisfaction_map = {'Unsatisfied': 0, 'Neutral': 1,
                    'Satisfied': 2, 'Very Satisfied': 3}
education_map = {'Primary': 0, 'High School': 1, 'Bachelor\'s': 2,
                 'Master\'s': 3, 'PhD': 4}
print("\n   Customer Satisfaction Mapping:", satisfaction_map)
print("   Education Level Mapping:", education_map)

# --- One-Hot Encoding misaal ---
print("\n2. ONE-HOT ENCODING:")
colors = ['Red', 'Green', 'Blue', 'Red', 'Green']
df_color = pd.DataFrame({'Color': colors})
ohe_result = pd.get_dummies(df_color, columns=['Color'])
print(df_color.head())
print("   After One-Hot Encoding:")
print(ohe_result)

# Weight-Height dataset misaal (apke notes se)
print("\n3. PRACTICAL EXAMPLE - Weight-Height Dataset:")
wh_data = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Height': [175, 162, 180, 155, 170],
    'Weight': [70, 55, 85, 50, 75]
})
print("Original:")
print(wh_data)

# Gender encode karen (Male=1, Female=0)
wh_data['Gender'] = wh_data['Gender'].replace({'Male': 1, 'Female': 0})
print("\nAfter Label Encoding:")
print(wh_data)


# ===========================================================================
# SECTION 6: MULTIPLE LINEAR REGRESSION + MIN-MAX SCALING KYA HAI?
# ===========================================================================
"""
================================================================================
             MULTIPLE LINEAR REGRESSION & MIN-MAX SCALING
================================================================================

MULTIPLE LINEAR REGRESSION KYA HAI?
--------------------------------------
Ek se zyada inputs (features) se output predict karna
  y = m1*x1 + m2*x2 + ... + c

MIN-MAX SCALING (NORMALIZATION) KYA HAI?
-------------------------------------------
Definition: Tamam features ko 0 se 1 ki range mein laana
Formula: scaled = (x - min) / (max - min)

Kyu zaroori?
  - Mukhtalif scale wale features ko barabar banana
  - Model zyada accurate hota hai
  - Training tez hoti hai

Misaal: Height (150-200) aur Weight (40-100)
  -> Height aur Weight ki range mukhtalif hai
  -> Scaling ke baad dono 0-1 mein aa jate hain
"""

print("\n" + "=" * 60)
print("    MULTIPLE LINEAR REGRESSION - Weight-Height Dataset")
print("=" * 60)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Data tayyar karen
np.random.seed(42)
n = 1000
gender = np.random.choice([0, 1], n)
height = np.random.uniform(150, 195, n)
weight = 0.5 * height + gender * 10 - 50 + np.random.normal(0, 5, n)

wh = pd.DataFrame({'Gender': gender, 'Height': height, 'Weight': weight})
print("\nData Sample:")
print(wh.head())

# X aur y
X_wh = wh[['Gender', 'Height']]
y_wh = wh['Weight']

# Train/Test Split
x_tr, x_te, y_tr, y_te = train_test_split(
    X_wh, y_wh, test_size=0.2, random_state=2
)

# Min-Max Scaling
print("\nMin-Max Scaling lagane se pehle:")
print(f"  Height range: {x_tr['Height'].min():.1f} - {x_tr['Height'].max():.1f}")

scaler = MinMaxScaler()
x_tr_scaled = scaler.fit_transform(x_tr)
x_te_scaled = scaler.transform(x_te)

print("Min-Max Scaling lagane ke baad:")
print(f"  Height range: {x_tr_scaled[:,1].min():.2f} - {x_tr_scaled[:,1].max():.2f}")

# Model Train
model_wh = LinearRegression()
model_wh.fit(x_tr_scaled, y_tr)
y_pred_wh = model_wh.predict(x_te_scaled)

mae  = mean_absolute_error(y_te, y_pred_wh)
mse  = mean_squared_error(y_te, y_pred_wh)
rmse = np.sqrt(mse)
r2   = r2_score(y_te, y_pred_wh)

print(f"\nMultiple Regression Results:")
print(f"  MAE:  {mae:.4f}")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R2:   {r2:.4f}")


# ===========================================================================
# SECTION 7: POLYNOMIAL FEATURES KYA HAIN?
# ===========================================================================
"""
================================================================================
                        POLYNOMIAL FEATURES
================================================================================

LINEAR REGRESSION vs POLYNOMIAL FEATURES
-------------------------------------------
  Linear:     y = mx + c           -> Seedhi lakeer
  Polynomial: y = ax^2 + bx + c    -> Curve

KAB ISTEMAL KAREN POLYNOMIAL?
--------------------------------
  1. Jab simple linear regression acha fit na de
  2. Jab data curve shape mein ho (asal relationship non-linear ho)

POLYNOMIAL FEATURES KAISE KAAM KARTA HAI?
--------------------------------------------
  Agar X = [Packets] to:
  Polynomial degree=2 -> X_poly = [Packets, Packets^2]
  Yani ek naya column ban jata hai X^2

NOTE (apke notes se):
  1. Linear Regression barah-e-raast data par train hoti hai magar sirf
     seedhi line bana sakti hai
  2. Polynomial Features mein Poly.fit_transform(X) se naye features
     bante hain, phir Linear Regression apply hoti hai -> Curve banti
     hai -> zyada accurate fit
"""

print("\n" + "=" * 60)
print("    POLYNOMIAL FEATURES - Popcorn Dataset")
print("=" * 60)

from sklearn.preprocessing import PolynomialFeatures

# Popcorn Data (apke notes se - Packets -> Earnings)
np.random.seed(2)
n = 100
packets = np.random.uniform(1, 20, n)
earnings = 2 * packets**2 - 5 * packets + 10 + np.random.normal(0, 20, n)

df_pop = pd.DataFrame({'Packets': packets, 'Earnings': earnings})
print("\nPopcorn Data Sample:")
print(df_pop.head())
print(f"Shape: {df_pop.shape}")
df_pop.info()

X_pop = df_pop[['Packets']]
y_pop = df_pop[['Earnings']]

x_tr_p, x_te_p, y_tr_p, y_te_p = train_test_split(
    X_pop, y_pop, test_size=0.2, random_state=2
)

# --- Method 1: Simple Linear Regression ---
lr_simple = LinearRegression()
lr_simple.fit(x_tr_p, y_tr_p)
y_pred_simple = lr_simple.predict(x_te_p)
r2_simple = r2_score(y_te_p, y_pred_simple)

# --- Method 2: Polynomial Features + Linear Regression ---
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_pop)

x_tr_poly, x_te_poly, y_tr_poly, y_te_poly = train_test_split(
    X_poly, y_pop, test_size=0.2, random_state=2
)

poly_lr = LinearRegression()
poly_lr.fit(x_tr_poly, y_tr_poly)
y_pred_poly = poly_lr.predict(x_te_poly)
r2_poly = r2_score(y_te_poly, y_pred_poly)

print(f"\nComparison:")
print(f"  Simple Linear Regression R2:  {r2_simple:.4f}")
print(f"  Polynomial Features R2:        {r2_poly:.4f}")
print(f"\n  Polynomial ne {(r2_poly-r2_simple)*100:.1f}% zyada accuracy di!")

print("\nActual vs Polynomial Predicted (first 5):")
print("  Actual  | Poly Pred")
for i in range(5):
    print(f"  {y_te_poly.iloc[i,0]:.2f}   | {y_pred_poly[i,0]:.2f}")


# ===========================================================================
# SECTION 8: RIDGE & LASSO REGRESSION KYA HAIN?
# ===========================================================================
"""
================================================================================
                    RIDGE & LASSO REGRESSION
================================================================================

OVERFITTING KYA HAI?
----------------------
  Jab Model training data par bohat acha kaam kare lekin naye data par
  bohat bura kaam kare -> yeh Overfitting hai

  Misaal: Aap ne 12 sawalon ko rata lagaya
    -> Woh 12 sawal aayen -> Pass
    -> Naye sawal aayen -> Fail

  Overfitting kyu hota hai?
    -> Model ne training data ke chhote details bhi yaad kar liye
    -> Coefficients bohat bade ho jate hain

--------------------------------------------------------------------------
RIDGE REGRESSION (L2 Regularization) KYA HAI?
--------------------------------------------------------------------------
  Definition: Ridge overfitting rokta hai
  Loss Function: Error + alpha x (sum of coefficients^2)

  Yeh tamam coefficients ko chhota karta hai lekin zero nahi karta

  Alpha kya hai?
    -> Ridge/Lasso ka ek parameter
    -> Bara alpha -> zyada regularization -> sada model
    -> Chhota alpha -> kam regularization -> complex model
    -> Best alpha dhoondne ke liye loop chalate hain

--------------------------------------------------------------------------
LASSO REGRESSION (L1 Regularization) KYA HAI?
--------------------------------------------------------------------------
  Definition: Lasso features ko khud hi muntakhib karta hai
  Loss Function: Error + alpha x (sum of |coefficients|)

  Yeh ghair zaroori features ke coefficients ko zero kar deta hai
  (Feature Selection khud-ba-khud ho jati hai)

FARQ (Ridge vs Lasso):
  Ridge -> Sab coefficients chhote karta hai (zero nahi)
  Lasso -> Ghair zaroori coefficients bilkul zero kar deta hai
"""

print("\n" + "=" * 60)
print("    RIDGE & LASSO REGRESSION - California Housing")
print("=" * 60)

from sklearn.linear_model import Ridge, Lasso

# Synthetic Housing Data (California Housing jaisa)
np.random.seed(42)
n_h = 2000
df_h = pd.DataFrame({
    'MedInc':     np.random.uniform(1, 15, n_h),
    'HouseAge':   np.random.uniform(1, 52, n_h),
    'AveRooms':   np.random.uniform(2, 10, n_h),
    'AveBedrms':  np.random.uniform(1, 3, n_h),
    'Population': np.random.uniform(100, 3000, n_h),
    'AveOccup':   np.random.uniform(1, 6, n_h),
    'Latitude':   np.random.uniform(32, 42, n_h),
    'Longitude':  np.random.uniform(-124, -114, n_h),
})
df_h['Price'] = (0.5 * df_h['MedInc'] - 0.01 * df_h['HouseAge']
                 + 0.1 * df_h['AveRooms'] + np.random.normal(0, 0.5, n_h))
print(f"\nHousing Data Shape: {df_h.shape}")
print(df_h.head(3))

# X aur y
X_h = df_h.drop('Price', axis=1)
y_h = df_h['Price']

# Train/Test Split
x_tr_h, x_te_h, y_tr_h, y_te_h = train_test_split(
    X_h, y_h, test_size=0.2, random_state=42
)

# Scaling (zaroori Ridge/Lasso ke liye)
scaler_h = StandardScaler()
x_tr_scaled_h = scaler_h.fit_transform(x_tr_h)
x_te_scaled_h  = scaler_h.transform(x_te_h)

# --- Linear Regression (baseline) ---
lr_h = LinearRegression()
lr_h.fit(x_tr_scaled_h, y_tr_h)
y_pred_lr_h = lr_h.predict(x_te_scaled_h)
print(f"\nLinear Regression - Train R2: {lr_h.score(x_tr_scaled_h, y_tr_h):.4f}")
print(f"Linear Regression - Test  R2: {r2_score(y_te_h, y_pred_lr_h):.4f}")

# --- Ridge Regression ---
ridge = Ridge()
ridge.fit(x_tr_scaled_h, y_tr_h)
y_pred_ridge = ridge.predict(x_te_scaled_h)
print(f"\nRidge Regression  - Train R2: {ridge.score(x_tr_scaled_h, y_tr_h):.4f}")
print(f"Ridge Regression  - Test  R2: {r2_score(y_te_h, y_pred_ridge):.4f}")

# --- Best Alpha dhoondna (Loop) ---
print("\nBest Alpha dhoond rahe hain (Ridge)...")
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
ridge_results = []
for alpha in alphas:
    ridge_temp = Ridge(alpha=alpha)
    ridge_temp.fit(x_tr_scaled_h, y_tr_h)
    train_r2 = ridge_temp.score(x_tr_scaled_h, y_tr_h)
    test_r2  = r2_score(y_te_h, ridge_temp.predict(x_te_scaled_h))
    ridge_results.append((alpha, train_r2, test_r2))

print(f"{'Alpha':>10} | {'Train R2':>10} | {'Test R2':>10}")
print("-" * 35)
for alpha, tr, te in ridge_results:
    print(f"{alpha:>10} | {tr:>10.4f} | {te:>10.4f}")

# --- Lasso Regression ---
lasso = Lasso()
lasso.fit(x_tr_scaled_h, y_tr_h)
y_pred_lasso = lasso.predict(x_te_scaled_h)
print(f"\nLasso Regression  - Train R2: {lasso.score(x_tr_scaled_h, y_tr_h):.4f}")
print(f"Lasso Regression  - Test  R2: {r2_score(y_te_h, y_pred_lasso):.4f}")

# Lasso ka feature selection dikhayen
print("\nLasso ne ghair zaroori features ko zero kiya:")
for feat, coef in zip(X_h.columns, lasso.coef_):
    status = "rakha" if coef != 0 else "zero kiya"
    print(f"  {feat:20s}: {coef:8.4f}  {status}")

# --- Best Alpha (Lasso) ---
print("\nBest Alpha dhoond rahe hain (Lasso)...")
lasso_results = []
for alpha in alphas:
    lasso_temp = Lasso(alpha=alpha, max_iter=10000)
    lasso_temp.fit(x_tr_scaled_h, y_tr_h)
    train_r2 = lasso_temp.score(x_tr_scaled_h, y_tr_h)
    test_r2  = r2_score(y_te_h, lasso_temp.predict(x_te_scaled_h))
    lasso_results.append((alpha, train_r2, test_r2))

print(f"{'Alpha':>10} | {'Train R2':>10} | {'Test R2':>10}")
print("-" * 35)
for alpha, tr, te in lasso_results:
    print(f"{alpha:>10} | {tr:>10.4f} | {te:>10.4f}")


# ===========================================================================
# SECTION 9: LOGISTIC REGRESSION KYA HAI?
# ===========================================================================
"""
================================================================================
                        LOGISTIC REGRESSION
================================================================================

LOGISTIC REGRESSION KYA HAI?
-------------------------------
Logistic Regression ek Classification algorithm hai jo 0 ya 1 (Yes/No,
Approved/Rejected) predict karta hai.

LINEAR REGRESSION vs LOGISTIC REGRESSION
-------------------------------------------
  Linear:   Continuous number predict karti hai (50, 100, 300...)
  Logistic: Category predict karti hai (0 ya 1)

SIGMOID FUNCTION KYA HAI?
----------------------------
  Formula: f(x) = 1 / (1 + e^(-x))

  Yeh koi bhi number ko 0 aur 1 ke darmiyan le aata hai

  Misaal:
    Linear Regression kahe: output = -50, 100, 300
    Sigmoid lagane ke baad: sab 0 aur 1 ke darmiyan aa jate hain
    Yeh probability hai (masalan 0.7 = 70% chance approved)

  Kaam:
    -> 0.5 se zyada -> class 1 (Yes/Approved)
    -> 0.5 se kam -> class 0 (No/Rejected)
"""

print("\n" + "=" * 60)
print("    LOGISTIC REGRESSION - Loan Dataset")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# Loan Dataset banana (apke notes ke mutabiq)
np.random.seed(42)
n = 614
loan_data = pd.DataFrame({
    'Gender':           np.random.choice(['Male', 'Female'], n),
    'Married':          np.random.choice(['Yes', 'No'], n),
    'Dependents':       np.random.choice(['0', '1', '2', '3+'], n),
    'Self_Employed':    np.random.choice(['Yes', 'No'], n),
    'LoanAmount':       np.random.randint(50, 700, n),
    'Loan_Amount_Term': np.random.choice([120, 180, 240, 360], n),
    'Credit_History':   np.random.choice([0, 1], n, p=[0.2, 0.8]),
    'Loan_Status':      np.random.choice(['Y', 'N'], n, p=[0.7, 0.3])
})

print("\nLoan Data Sample:")
print(loan_data.head())
print(f"\nShape: {loan_data.shape}")
print("\nLoan Status distribution:")
print(loan_data['Loan_Status'].value_counts())

# --- Data Cleaning (apke notes ke mutabiq) ---
# Dependents ki '3+' ko '3' se badlen
loan_data['Dependents'] = loan_data['Dependents'].replace({'3+': '3'})
loan_data['Dependents'] = loan_data['Dependents'].astype(int)

# Loan_Status ko 0/1 mein badlen (N=0, Y=1)
loan_data['Loan_Status'] = loan_data['Loan_Status'].replace({'N': 0, 'Y': 1})

# Null values fill karen
loan_data['Gender']           = loan_data['Gender'].fillna(loan_data['Gender'].mode()[0])
loan_data['Married']          = loan_data['Married'].fillna(loan_data['Married'].mode()[0])
loan_data['Self_Employed']    = loan_data['Self_Employed'].fillna(loan_data['Self_Employed'].mode()[0])
loan_data['LoanAmount']       = loan_data['LoanAmount'].fillna(loan_data['LoanAmount'].median())
loan_data['Loan_Amount_Term'] = loan_data['Loan_Amount_Term'].fillna(loan_data['Loan_Amount_Term'].median())
loan_data['Credit_History']   = loan_data['Credit_History'].fillna(loan_data['Credit_History'].mode()[0])

# --- Encoding (apke notes ke mutabiq) ---
# Gender encode karen
loan_data['Gender'] = loan_data['Gender'].replace({'Male': 1, 'Female': 0})
loan_data['Married'] = loan_data['Married'].replace({'Yes': 1, 'No': 0})
loan_data['Self_Employed'] = loan_data['Self_Employed'].replace({'Yes': 1, 'No': 0})

print("\nData after encoding:")
print(loan_data.head(3))

# --- X aur y ---
X_loan = loan_data.drop(['Loan_Status'], axis=1)
y_loan = loan_data['Loan_Status'].astype(int)

# Train/Test Split
x_tr_l, x_te_l, y_tr_l, y_te_l = train_test_split(
    X_loan, y_loan, test_size=0.2, random_state=42
)

# --- Scaling ---
scaler_l = StandardScaler()
x_tr_scaled_l = scaler_l.fit_transform(x_tr_l)
x_te_scaled_l  = scaler_l.transform(x_te_l)

# --- Model ---
model_loan = LogisticRegression(max_iter=500)
model_loan.fit(x_tr_scaled_l, y_tr_l)

# --- Predictions ---
y_pred_loan = model_loan.predict(x_te_scaled_l)
y_prob_loan = model_loan.predict_proba(x_te_scaled_l)  # Sigmoid output

print("\nFirst 5 Predictions:")
print(f"  {'Actual':>8} | {'Predicted':>10} | {'Probability (Yes)':>18}")
for i in range(5):
    print(f"  {y_te_l.iloc[i]:>8} | {y_pred_loan[i]:>10} | {y_prob_loan[i,1]:>18.4f}")

# --- Evaluation ---
acc = accuracy_score(y_te_l, y_pred_loan)
print(f"\nLogistic Regression Accuracy: {acc:.4f} ({acc*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_te_l, y_pred_loan,
                             target_names=['Rejected (0)', 'Approved (1)']))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_te_l, y_pred_loan)
print(cm)
print("  [True Negative | False Positive]")
print("  [False Negative | True Positive]")


# ===========================================================================
# SECTION 10: SUMMARY - KHULASA
# ===========================================================================
"""
================================================================================
                          KHULASA (SUMMARY)
================================================================================

Algorithm       | Kab istemal karen           | Output Type
--------------------------------------------------------------
Linear Reg.    | Number predict karna        | Continuous
Polynomial     | Curve shaped data           | Continuous
Ridge          | Overfitting + sab features  | Continuous
Lasso          | Overfitting + feature sel   | Continuous
Logistic Reg.  | 0/1 Classification          | Binary

Encoding:
  Label Encoding   -> Tarteeb wale categories
  One-Hot Encoding -> Baghair tarteeb wale categories

Metrics:
  MAE   -> Ausat ghalti (absolute)
  MSE   -> Ausat murabba ghalti
  RMSE  -> MSE ka square root
  R2    -> Model ki taaqat (0-1, zyada behtar)
"""

print("\n" + "=" * 60)
print("    Mukammal ML code chal gaya!")
print("=" * 60)
print("""
Topics Covered:
  - Machine Learning kya hai
  - 4 aqsaam (Supervised, Unsupervised, Semi, RL)
  - Linear Regression (Jobs Dataset)
  - Evaluation Metrics (MAE, MSE, RMSE, R2)
  - SST, SSE, SSR
  - Label Encoding vs One-Hot Encoding
  - Multiple Regression + Min-Max Scaling
  - Polynomial Features (Popcorn Dataset)
  - Ridge & Lasso Regression (California Housing)
  - Logistic Regression + Sigmoid (Loan Dataset)

Files:
  -> linear_regression_plot.png (graph)
""")