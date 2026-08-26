# =============================================================================
#   MACHINE LEARNING - مکمل نوٹس اور کوڈ
#   آپ کی نوٹ بک سے بنایا گیا
# =============================================================================

# ===========================================================================
# SECTION 1: MACHINE LEARNING کیا ہے؟
# ===========================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        MACHINE LEARNING (ML)                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

تعریف (Definition):
--------------------
Machine Learning، Artificial Intelligence (AI) کا ایک حصہ ہے جس میں کمپیوٹر کو
اس طرح تیار کیا جاتا ہے کہ وہ data دیکھ کر خود سیکھ سکے اور بغیر صراحتاً
پروگرام کیے، تجربے سے بہتر ہوتا جائے۔

سادہ الفاظ میں:
  جیسے بچہ بار بار گِر کر چلنا سیکھتا ہے، ویسے ہی ML میں
  کمپیوٹر data دیکھ کر patterns پہچاننا سیکھتا ہے۔

ML کے 3 اہم حصے:
  1. DATA     → کمپیوٹر کو سکھانے کا مواد
  2. ALGORITHM → ریاضی کے اصول جو data سے سیکھنے میں مدد کرتے ہیں
  3. MODEL    → وہ نتیجہ جو training کے بعد predictions کرتا ہے

ML کیوں استعمال کریں؟
  ✓ بڑے data سے patterns ڈھونڈنے کے لیے
  ✓ انسانی غلطیاں کم کرنے کے لیے
  ✓ خودکار فیصلے کرنے کے لیے (spam detection, loan approval وغیرہ)
"""

# ===========================================================================
# SECTION 2: TYPES OF MACHINE LEARNING - اقسام
# ===========================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     ML کی 4 اہم اقسام                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. SUPERVISED LEARNING (نگرانی والی سیکھ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   تعریف: کمپیوٹر کو labeled data دیا جاتا ہے یعنی سوال اور جواب دونوں دیے
           جاتے ہیں۔ کمپیوٹر اس سے pattern سیکھ کر نئے data کا جواب دیتا ہے۔

   مثال: 
     → آپ کے پاس گھروں کی قیمتیں اور ان کی سائز کا data ہے
     → Model سیکھتا ہے: بڑا گھر = زیادہ قیمت
     → پھر نئے گھر کی قیمت predict کرتا ہے

   اقسام:
     a) REGRESSION  → number predict کرے (مثلاً گھر کی قیمت، تنخواہ)
     b) CLASSIFICATION → category predict کرے (spam/not spam, loan approved/rejected)

   حقیقی مثالیں:
     ✓ Email spam detection
     ✓ House price prediction
     ✓ Loan approval/rejection

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. UNSUPERVISED LEARNING (بغیر نگرانی کے سیکھ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   تعریف: کمپیوٹر کو صرف data دیا جاتا ہے، کوئی label (جواب) نہیں ہوتا۔
           کمپیوٹر خود ہی data میں similarities اور patterns ڈھونڈتا ہے۔

   مثال:
     → آپ کے پاس customers کا خریداری data ہے
     → Model خود مختلف groups بنا لیتا ہے (سستے پسند کرنے والے، مہنگے پسند کرنے والے)
     → آپ نے کوئی label نہیں دیا، model نے خود سیکھا

   اقسام:
     a) CLUSTERING   → data کو گروپس میں تقسیم کرنا
     b) ASSOCIATION  → relationships ڈھونڈنا (جو X خریدے وہ Y بھی خریدتا ہے)

   حقیقی مثالیں:
     ✓ Netflix کی recommendations
     ✓ Customer segmentation
     ✓ Market basket analysis

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. SEMI-SUPERVISED LEARNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   تعریف: تھوڑا labeled data + بہت زیادہ unlabeled data۔
           پہلے labeled سے سیکھتا ہے، پھر unlabeled کو label کرتا ہے۔

   مثال: 100 تصویروں میں 10 labeled (بلی/کتا) باقی 90 unlabeled
         Model پہلے 10 سے سیکھتا ہے پھر 90 خود label کرتا ہے۔

   اقسام (آپ کے notes سے):
     a) Generative Models → unlabeled data سے 2 minutes کی samples بناتا ہے
     b) Self Training    → model خود labels کرتا ہے اور دوبارہ train ہوتا ہے

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. REINFORCEMENT LEARNING (انعام والی سیکھ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   تعریف: ایک Agent ہے جو Environment میں کام کرتا ہے۔
           صحیح کام پر Reward ملتا ہے، غلط پر Penalty۔
           Agent خود سیکھتا ہے کہ کیا کرنا چاہیے۔

   3 اہم حصے:
     1. Agent       → وہ جو سیکھ رہا ہے (جیسے robot یا انسان)
     2. Environment → وہ دنیا/ماحول جس میں کام کرتا ہے
     3. Reward/Penalty → صحیح کام پر انعام، غلط پر سزا

   مثال: Video game کھیلنا سیکھنا - ہر صحیح move پر points ملتے ہیں

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UNSUPERVISED کی تفصیل - Clustering & Association
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  CLUSTERING:
    → data کو مختلف گروپس میں تقسیم کرنا
    → Algorithm خود groups بناتا ہے data کی similarity کی بنیاد پر
    → مثال: Customers کو 3 groups میں بانٹنا

  ASSOCIATION:
    → data میں relationships ڈھونڈنا
    → مثال: جو روٹی خریدے وہ مکھن بھی خریدتا ہے
    → 100 میں سے کتنے customers نے یہ pattern follow کیا؟
"""

# ===========================================================================
# SECTION 3: LINEAR REGRESSION
# ===========================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        LINEAR REGRESSION                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

تعریف:
------
Linear Regression ایک Supervised ML algorithm ہے جو continuous/numerical
values predict کرتا ہے۔ یہ data میں ایک سیدھی لکیر (line) ڈھونڈتا ہے
جو best fit ہو۔

فارمولا:
  y = mx + c
  
  جہاں:
    y = output/target (جو predict کرنا ہے، مثلاً Package)
    x = input/feature (جو ہمارے پاس ہے، مثلاً CGPA)
    m = slope/coefficient (CGPA کتنا Package بڑھاتا ہے)
    c = intercept/constant (بنیادی رقم جب x=0 ہو)

کیوں استعمال کریں؟
  ✓ جب output ایک number ہو (قیمت، تنخواہ، نمبر)
  ✓ جب input اور output کا relationship linear ہو
  ✓ آسان اور سمجھنے میں سب سے آسان algorithm

مثال:
  CGPA 3.4 ہے → Package کیا ہوگا؟
  Model: y = 0.5839 * 3.4 + c
  
Steps یاد رکھیں:
  1. Libraries import کریں
  2. Data load کریں (CSV)
  3. X (input) اور y (output) الگ کریں
  4. Train/Test split کریں (80/20)
  5. Model بنائیں (LinearRegression)
  6. Model train کریں (fit)
  7. Predictions نکالیں (predict)
"""

# ===========================================================================
# SECTION 3A: LINEAR REGRESSION CODE - Jobs Dataset
# ===========================================================================
print("=" * 60)
print("    LINEAR REGRESSION - JOBS DATASET (CGPA → Package)")
print("=" * 60)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- Step 1: Data Load ---
# نوٹ: آپ کے پاس Jobs.csv ہونا چاہیے اسی folder میں
# jobs = pd.read_csv('Jobs.csv')
# jobs.info()  # data کی معلومات دیکھنے کے لیے

# Demo کے لیے synthetic data بناتے ہیں
np.random.seed(42)
n = 200
cgpa = np.random.uniform(2.0, 4.0, n)
package = 0.5839 * cgpa + 0.5 + np.random.normal(0, 0.3, n)
jobs = pd.DataFrame({'cgpa': cgpa, 'package': package})
print("\nData Sample:")
print(jobs.head())
print(f"\nData Shape: {jobs.shape}")

# --- Step 2: Features اور Target الگ کریں ---
X = jobs[['cgpa']]    # Input feature (2D array چاہیے)
y = jobs[['package']] # Output/Target

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

# --- Step 3: Train/Test Split (80% Train, 20% Test) ---
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {x_train.shape}")
print(f"Test size:  {x_test.shape}")

# --- Step 4: Model بنائیں ---
lr = LinearRegression()

# --- Step 5: Model Train کریں ---
lr.fit(x_train, y_train)
print("\nModel Successfully Trained!")

# --- Step 6: Predictions ---
y_pred = lr.predict(x_test)
print("\nFirst 5 Predictions vs Actual:")
for i in range(5):
    print(f"  CGPA: {x_test.iloc[i,0]:.2f} | Predicted: {y_pred[i,0]:.4f} | Actual: {y_test.iloc[i,0]:.2f}")

# --- Step 7: Coefficient اور Intercept ---
m = lr.coef_[0][0]   # slope
c = lr.intercept_[0] # intercept
print(f"\nFormula: y = {m:.4f}x + {c:.4f}")
print(f"یعنی: Package = {m:.4f} × CGPA + {c:.4f}")

# --- Step 8: نئی CGPA کے لیے Prediction ---
cgpa_input = 3.4
a = np.array([[cgpa_input]])
point = lr.predict(a)
print(f"\nCGPA {cgpa_input} کے لیے Package: {point[0][0]:.4f}")

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
# SECTION 4: EVALUATION METRICS - ماپ کے اوزار
# ===========================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        EVALUATION METRICS                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

تعریف:
------
Evaluation Metrics وہ ماپ ہیں جن سے ہم دیکھتے ہیں کہ ہمارا Model
کتنا اچھا یا برا کام کر رہا ہے۔

یہ y_test (actual) اور y_pred (predicted) کے درمیان فرق ناپتے ہیں۔

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. MAE - Mean Absolute Error (اوسط مطلق غلطی)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   فارمولا: MAE = Average of |actual - predicted|
   
   کیا کرتا ہے: ہر غلطی کا اوسط نکالتا ہے (بغیر square کیے)
   
   مثال: 3 predictions
     Actual:    2, 4, 6    → Mean = 4
     Predicted: 2.5, 3.5, 5.5
     Errors:    0.5, 0.5, 0.5
     MAE = 0.5
   
   کب استعمال: جب بڑی غلطیاں اور چھوٹی غلطیاں برابر اہم ہوں

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. MSE - Mean Squared Error (اوسط مربع غلطی)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   فارمولا: MSE = Average of (actual - predicted)²
   
   کیا کرتا ہے: غلطیوں کو square کر کے اوسط نکالتا ہے
   بڑی غلطیوں کو زیادہ penalty دیتا ہے
   
   کب استعمال: جب بڑی غلطیاں سنگین ہوں

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. RMSE - Root Mean Squared Error
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   فارمولا: RMSE = √MSE
   
   کیا کرتا ہے: MSE کا square root نکالتا ہے
   اصل unit میں واپس لاتا ہے (سمجھنا آسان)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. R² Score (R-Squared) - ماڈل کی طاقت
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Range: 0 سے 1 (یا کبھی منفی)
   1 کے قریب = بہترین model
   0 کے قریب = بیکار model

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SST, SSE, SSR کیا ہیں؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   SST (Total Sum of Squares):
     → y کا mean سے کل فرق
     → SST = Σ(y - ȳ)²
     → مثال: y = [2,4,6], ȳ=4
       SST = (2-4)² + (4-4)² + (6-4)² = 4+0+4 = 8

   SSE (Sum of Squared Errors):
     → actual اور predicted کا فرق
     → Model کتنا غلط ہے

   SSR (Sum of Squared Regression):
     → Model نے کتنی variation explain کی
     
   R² = 1 - (SSE/SST)   یا   R² = SSR/SST
"""

print("\n" + "=" * 60)
print("    EVALUATION METRICS")
print("=" * 60)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# SST مثال (آپ کے notes سے)
y_example = np.array([2, 4, 6])
y_mean = np.mean(y_example)
sst = np.sum((y_example - y_mean) ** 2)
print(f"\nSST مثال: y=[2,4,6], mean={y_mean}")
print(f"SST = (2-{y_mean})² + (4-{y_mean})² + (6-{y_mean})² = {sst}")

# Actual Metrics on our model
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\nLinear Regression Model کی Performance:")
print(f"  MAE  (Mean Absolute Error)  : {mae:.4f}")
print(f"  MSE  (Mean Squared Error)   : {mse:.4f}")
print(f"  RMSE (Root MSE)             : {rmse:.4f}")
print(f"  R²   (Accuracy/Score)       : {r2:.4f}")
print(f"\n  R²={r2:.2f} مطلب Model نے {r2*100:.1f}% variation explain کی")


# ===========================================================================
# SECTION 5: LABEL ENCODING vs ONE-HOT ENCODING
# ===========================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               LABEL ENCODING vs ONE-HOT ENCODING                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

مسئلہ: ML models صرف numbers سمجھتے ہیں، text نہیں
حل: Categorical data کو numbers میں بدلنا

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. LABEL ENCODING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ہر category کو ایک نمبر دو
   
   کب استعمال: جب order/ترتیب ہو (Ordinal data)
   مثال:
     Size: Small=0, Medium=1, Large=2  ← ترتیب موجود ہے ✓
     
   مثالیں (آپ کے notes سے):
     Customer Satisfaction:
       Unsatisfied=0, Neutral=1, Satisfied=2, Very Satisfied=3
     
     Education Level:
       Primary=0, High School=1, Bachelor's=2, Master's=3, PhD=4
     
     Temperature:
       Low=-1, Medium=-2, High=3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. ONE-HOT ENCODING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ہر category کے لیے نئی column بناؤ (0 یا 1)
   
   کب استعمال: جب کوئی order نہ ہو (Nominal data)
   مثال:
     Color: Red, Green, Blue ← کوئی ترتیب نہیں
     
     Color | Red | Green | Blue
     Red   |  1  |   0   |  0
     Green |  0  |   1   |  0
     Blue  |  0  |   0   |  1

   فرق:
     Label: ترتیب والے data کے لیے (کم columns)
     One-Hot: بغیر ترتیب والے data کے لیے (زیادہ columns، لیکن غلط order نہیں)
"""

print("\n" + "=" * 60)
print("    LABEL ENCODING vs ONE-HOT ENCODING")
print("=" * 60)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# --- Label Encoding مثال ---
print("\n1. LABEL ENCODING:")
satisfaction = ['Unsatisfied', 'Neutral', 'Satisfied', 'Very Satisfied',
                'Satisfied', 'Unsatisfied']
le = LabelEncoder()
encoded = le.fit_transform(satisfaction)
print("   Original:", satisfaction[:4])
print("   Encoded: ", encoded[:4].tolist())

# Manual Mapping (آپ کے notes کے مطابق)
satisfaction_map = {'Unsatisfied': 0, 'Neutral': 1,
                    'Satisfied': 2, 'Very Satisfied': 3}
education_map = {'Primary': 0, 'High School': 1, 'Bachelor\'s': 2,
                 'Master\'s': 3, 'PhD': 4}
print("\n   Customer Satisfaction Mapping:", satisfaction_map)
print("   Education Level Mapping:", education_map)

# --- One-Hot Encoding مثال ---
print("\n2. ONE-HOT ENCODING:")
colors = ['Red', 'Green', 'Blue', 'Red', 'Green']
df_color = pd.DataFrame({'Color': colors})
ohe_result = pd.get_dummies(df_color, columns=['Color'])
print(df_color.head())
print("   After One-Hot Encoding:")
print(ohe_result)

# Weight-Height dataset مثال (آپ کے notes سے)
print("\n3. PRACTICAL EXAMPLE - Weight-Height Dataset:")
wh_data = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Height': [175, 162, 180, 155, 170],
    'Weight': [70, 55, 85, 50, 75]
})
print("Original:")
print(wh_data)

# Gender encode کریں (Male=1, Female=0)
wh_data['Gender'] = wh_data['Gender'].replace({'Male': 1, 'Female': 0})
print("\nAfter Label Encoding:")
print(wh_data)


# ===========================================================================
# SECTION 6: MULTIPLE LINEAR REGRESSION + MIN-MAX SCALING
# ===========================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             MULTIPLE LINEAR REGRESSION & MIN-MAX SCALING                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Multiple Linear Regression:
  ایک سے زیادہ inputs (features) سے output predict کرنا
  y = m1*x1 + m2*x2 + ... + c

Min-Max Scaling (Normalization):
  تعریف: تمام features کو 0 سے 1 کی range میں لانا
  فارمولا: scaled = (x - min) / (max - min)
  
  کیوں ضروری؟
    ✓ مختلف scale والے features کو برابر بنانا
    ✓ Model زیادہ accurate ہوتا ہے
    ✓ Training تیز ہوتی ہے

  مثال: Height (150-200) اور Weight (40-100)
    → Height اور Weight کی range مختلف ہے
    → Scaling کے بعد دونوں 0-1 میں آ جاتے ہیں
"""

print("\n" + "=" * 60)
print("    MULTIPLE LINEAR REGRESSION - Weight-Height Dataset")
print("=" * 60)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Data تیار کریں
np.random.seed(42)
n = 1000
gender = np.random.choice([0, 1], n)
height = np.random.uniform(150, 195, n)
weight = 0.5 * height + gender * 10 - 50 + np.random.normal(0, 5, n)

wh = pd.DataFrame({'Gender': gender, 'Height': height, 'Weight': weight})
print("\nData Sample:")
print(wh.head())

# X اور y
X_wh = wh[['Gender', 'Height']]
y_wh = wh['Weight']

# Train/Test Split
x_tr, x_te, y_tr, y_te = train_test_split(
    X_wh, y_wh, test_size=0.2, random_state=2
)

# Min-Max Scaling
print("\nMin-Max Scaling لگانے سے پہلے:")
print(f"  Height range: {x_tr['Height'].min():.1f} - {x_tr['Height'].max():.1f}")

scaler = MinMaxScaler()
x_tr_scaled = scaler.fit_transform(x_tr)
x_te_scaled = scaler.transform(x_te)

print("Min-Max Scaling لگانے کے بعد:")
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
print(f"  R²:   {r2:.4f}")


# ===========================================================================
# SECTION 7: POLYNOMIAL FEATURES
# ===========================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        POLYNOMIAL FEATURES                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Linear Regression vs Polynomial Features:
  Linear:     y = mx + c           → سیدھی لکیر
  Polynomial: y = ax² + bx + c     → curve

کب استعمال کریں Polynomial؟
  1. جب simple linear regression اچھا fit نہ دے
  2. جب data curve shape میں ہو (اصل relationship non-linear ہو)

Polynomial Features کیسے کام کرتا ہے؟
  اگر X = [Packets] تو:
  Polynomial degree=2 → X_poly = [Packets, Packets²]
  یعنی ایک نیا column بن جاتا ہے X²

نوٹ (آپ کے notes سے):
  1. Linear Regression براہ راست data پر train ہوتی ہے
     مگر صرف سیدھی line بنا سکتی ہے
  2. Polynomial Features میں Poly.fit_transform(X) سے
     نئے features بنتے ہیں، پھر Linear Regression apply ہوتی ہے
     → Curve بنتی ہے → زیادہ accurate fit
"""

print("\n" + "=" * 60)
print("    POLYNOMIAL FEATURES - Popcorn Dataset")
print("=" * 60)

from sklearn.preprocessing import PolynomialFeatures

# Popcorn Data (آپ کے notes سے - Packets → Earnings)
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
print(f"  Simple Linear Regression R²:  {r2_simple:.4f}")
print(f"  Polynomial Features R²:        {r2_poly:.4f}")
print(f"\n  Polynomial نے {(r2_poly-r2_simple)*100:.1f}% زیادہ accuracy دی!")

print("\nActual vs Polynomial Predicted (first 5):")
print("  Actual  | Poly Pred")
for i in range(5):
    print(f"  {y_te_poly.iloc[i,0]:.2f}   | {y_pred_poly[i,0]:.2f}")


# ===========================================================================
# SECTION 8: RIDGE & LASSO REGRESSION
# ===========================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RIDGE & LASSO REGRESSION                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

پہلے سمجھیں - OVERFITTING کیا ہے؟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  جب Model training data پر بہت اچھا کام کرے لیکن
  نئے data پر بہت برا کام کرے → یہ Overfitting ہے

  مثال: آپ نے 12 سوالوں کو رٹا لگایا
    → وہ 12 سوال آئیں → پاس ✓
    → نئے سوال آئیں → فیل ✗

Overfitting کیوں ہوتا ہے؟
  → Model نے training data کے چھوٹے details بھی یاد کر لیے
  → Coefficients بہت بڑے ہو جاتے ہیں

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RIDGE REGRESSION (L2 Regularization)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  تعریف: Ridge overfitting روکتا ہے
  Loss Function: Error + α × (sum of coefficients²)
  
  یہ تمام coefficients کو چھوٹا کرتا ہے لیکن zero نہیں کرتا
  
  α (alpha) کیا ہے؟
    → Ridge/Lasso کا ایک parameter
    → بڑا α → زیادہ regularization → سادہ model
    → چھوٹا α → کم regularization → complex model
    → Best alpha ڈھونڈنے کے لیے loop چلاتے ہیں

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LASSO REGRESSION (L1 Regularization)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  تعریف: Lasso features کو خود ہی منتخب کرتا ہے
  Loss Function: Error + α × (sum of |coefficients|)
  
  یہ غیر ضروری features کے coefficients کو zero کر دیتا ہے
  (Feature Selection خودبخود ہو جاتی ہے)

فرق (Ridge vs Lasso):
  Ridge → سب coefficients چھوٹے کرتا ہے (zero نہیں)
  Lasso → غیر ضروری coefficients بالکل zero کر دیتا ہے
"""

print("\n" + "=" * 60)
print("    RIDGE & LASSO REGRESSION - California Housing")
print("=" * 60)

from sklearn.linear_model import Ridge, Lasso

# Synthetic Housing Data (California Housing جیسا)
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

# X اور y
X_h = df_h.drop('Price', axis=1)
y_h = df_h['Price']

# Train/Test Split
x_tr_h, x_te_h, y_tr_h, y_te_h = train_test_split(
    X_h, y_h, test_size=0.2, random_state=42
)

# Scaling (ضروری Ridge/Lasso کے لیے)
scaler_h = StandardScaler()
x_tr_scaled_h = scaler_h.fit_transform(x_tr_h)
x_te_scaled_h  = scaler_h.transform(x_te_h)

# --- Linear Regression (baseline) ---
lr_h = LinearRegression()
lr_h.fit(x_tr_scaled_h, y_tr_h)
y_pred_lr_h = lr_h.predict(x_te_scaled_h)
print(f"\nLinear Regression - Train R²: {lr_h.score(x_tr_scaled_h, y_tr_h):.4f}")
print(f"Linear Regression - Test  R²: {r2_score(y_te_h, y_pred_lr_h):.4f}")

# --- Ridge Regression ---
ridge = Ridge()
ridge.fit(x_tr_scaled_h, y_tr_h)
y_pred_ridge = ridge.predict(x_te_scaled_h)
print(f"\nRidge Regression  - Train R²: {ridge.score(x_tr_scaled_h, y_tr_h):.4f}")
print(f"Ridge Regression  - Test  R²: {r2_score(y_te_h, y_pred_ridge):.4f}")

# --- Best Alpha ڈھونڈنا (Loop) ---
print("\nBest Alpha ڈھونڈ رہے ہیں (Ridge)...")
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
ridge_results = []
for alpha in alphas:
    ridge_temp = Ridge(alpha=alpha)
    ridge_temp.fit(x_tr_scaled_h, y_tr_h)
    train_r2 = ridge_temp.score(x_tr_scaled_h, y_tr_h)
    test_r2  = r2_score(y_te_h, ridge_temp.predict(x_te_scaled_h))
    ridge_results.append((alpha, train_r2, test_r2))

print(f"{'Alpha':>10} | {'Train R²':>10} | {'Test R²':>10}")
print("-" * 35)
for alpha, tr, te in ridge_results:
    print(f"{alpha:>10} | {tr:>10.4f} | {te:>10.4f}")

# --- Lasso Regression ---
lasso = Lasso()
lasso.fit(x_tr_scaled_h, y_tr_h)
y_pred_lasso = lasso.predict(x_te_scaled_h)
print(f"\nLasso Regression  - Train R²: {lasso.score(x_tr_scaled_h, y_tr_h):.4f}")
print(f"Lasso Regression  - Test  R²: {r2_score(y_te_h, y_pred_lasso):.4f}")

# Lasso کا feature selection دکھائیں
print("\nLasso نے غیر ضروری features کو zero کیا:")
for feat, coef in zip(X_h.columns, lasso.coef_):
    status = "✓ رکھا" if coef != 0 else "✗ zero کیا"
    print(f"  {feat:20s}: {coef:8.4f}  {status}")

# --- Best Alpha (Lasso) ---
print("\nBest Alpha ڈھونڈ رہے ہیں (Lasso)...")
lasso_results = []
for alpha in alphas:
    lasso_temp = Lasso(alpha=alpha, max_iter=10000)
    lasso_temp.fit(x_tr_scaled_h, y_tr_h)
    train_r2 = lasso_temp.score(x_tr_scaled_h, y_tr_h)
    test_r2  = r2_score(y_te_h, lasso_temp.predict(x_te_scaled_h))
    lasso_results.append((alpha, train_r2, test_r2))

print(f"{'Alpha':>10} | {'Train R²':>10} | {'Test R²':>10}")
print("-" * 35)
for alpha, tr, te in lasso_results:
    print(f"{alpha:>10} | {tr:>10.4f} | {te:>10.4f}")


# ===========================================================================
# SECTION 9: LOGISTIC REGRESSION
# ===========================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        LOGISTIC REGRESSION                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

تعریف:
------
Logistic Regression ایک Classification algorithm ہے جو
0 یا 1 (Yes/No, Approved/Rejected) predict کرتا ہے۔

Linear Regression vs Logistic Regression:
  Linear:   continuous number predict کرتی ہے (50, 100, 300...)
  Logistic: category predict کرتی ہے (0 یا 1)

SIGMOID FUNCTION کیا ہے؟
━━━━━━━━━━━━━━━━━━━━━━━━━
  فارمولا: f(x) = 1 / (1 + e^(-x))
  
  یہ کوئی بھی number کو 0 اور 1 کے درمیان لے آتا ہے
  
  مثال:
    Linear Regression کہے: output = -50, 100, 300
    Sigmoid لگانے کے بعد: سب 0 اور 1 کے درمیان آ جاتے ہیں
    یہ probability ہے (مثلاً 0.7 = 70% chance approved)
  
  کام:
    → 0.5 سے زیادہ → class 1 (Yes/Approved)
    → 0.5 سے کم → class 0 (No/Rejected)
"""

print("\n" + "=" * 60)
print("    LOGISTIC REGRESSION - Loan Dataset")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# Loan Dataset بنانا (آپ کے notes کے مطابق)
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

# --- Data Cleaning (آپ کے notes کے مطابق) ---
# Dependents کی '3+' کو '3' سے بدلیں
loan_data['Dependents'] = loan_data['Dependents'].replace({'3+': '3'})
loan_data['Dependents'] = loan_data['Dependents'].astype(int)

# Loan_Status کو 0/1 میں بدلیں (N=0, Y=1)
loan_data['Loan_Status'] = loan_data['Loan_Status'].replace({'N': 0, 'Y': 1})

# Null values fill کریں
loan_data['Gender']           = loan_data['Gender'].fillna(loan_data['Gender'].mode()[0])
loan_data['Married']          = loan_data['Married'].fillna(loan_data['Married'].mode()[0])
loan_data['Self_Employed']    = loan_data['Self_Employed'].fillna(loan_data['Self_Employed'].mode()[0])
loan_data['LoanAmount']       = loan_data['LoanAmount'].fillna(loan_data['LoanAmount'].median())
loan_data['Loan_Amount_Term'] = loan_data['Loan_Amount_Term'].fillna(loan_data['Loan_Amount_Term'].median())
loan_data['Credit_History']   = loan_data['Credit_History'].fillna(loan_data['Credit_History'].mode()[0])

# --- Encoding (آپ کے notes کے مطابق) ---
# Gender encode کریں
loan_data['Gender'] = loan_data['Gender'].replace({'Male': 1, 'Female': 0})
loan_data['Married'] = loan_data['Married'].replace({'Yes': 1, 'No': 0})
loan_data['Self_Employed'] = loan_data['Self_Employed'].replace({'Yes': 1, 'No': 0})

print("\nData after encoding:")
print(loan_data.head(3))

# --- X اور y ---
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
# SECTION 10: SUMMARY - خلاصہ
# ===========================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          خلاصہ (SUMMARY)                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Algorithm       | کب استعمال کریں          | Output Type
─────────────────────────────────────────────────────────────
Linear Reg.    | Number predict کرنا      | Continuous
Polynomial     | Curve shaped data        | Continuous
Ridge          | Overfitting + سب features| Continuous
Lasso          | Overfitting + feature sel| Continuous
Logistic Reg.  | 0/1 Classification       | Binary

Encoding:
  Label Encoding  → ترتیب والے categories
  One-Hot Encoding→ بغیر ترتیب والے categories

Metrics:
  MAE   → اوسط غلطی (absolute)
  MSE   → اوسط مربع غلطی
  RMSE  → MSE کا square root
  R²    → Model کی طاقت (0-1, زیادہ بہتر)
"""

print("\n" + "=" * 60)
print("    مکمل ML کوڈ چل گیا! ✓")
print("=" * 60)
print("""
Topics Covered:
  ✓ Machine Learning کیا ہے
  ✓ 4 اقسام (Supervised, Unsupervised, Semi, RL)
  ✓ Linear Regression (Jobs Dataset)
  ✓ Evaluation Metrics (MAE, MSE, RMSE, R²)
  ✓ SST, SSE, SSR
  ✓ Label Encoding vs One-Hot Encoding
  ✓ Multiple Regression + Min-Max Scaling
  ✓ Polynomial Features (Popcorn Dataset)
  ✓ Ridge & Lasso Regression (California Housing)
  ✓ Logistic Regression + Sigmoid (Loan Dataset)

فائلیں:
  → linear_regression_plot.png (graph)
""")