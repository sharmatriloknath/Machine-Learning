# Column Transformer
 - To Solve the issue that we were facing with Encoding Multiple Column of DataFrame with different Encoding.
 - It was very difficult to merge all transformed dataframes.
 - Very Powerful concept.
 - Check Below Example

```python

Firstly we see the problem We have without Column Transformer
train_null_free_fever
# We have null values in fever, deal with them with SimpleImputer
si = SimpleImputer()

train_null_free_fever = si.fit_transform(X_train[['fever']])
test_null_free_fever = si.transform(X_test[['fever']])
train_null_free_fever.shape
(80, 1)
# Cough is Ordinal Column We will use OrdinalEncoder
oe = OrdinalEncoder(categories=[['Mild','Strong']])
X_train_transformed_cough
X_train_transformed_cough = oe.fit_transform(X_train[['cough']])
X_test_transformed_cough = oe.transform(X_test[['cough']])
X_train_transformed_cough.shape
(80, 1)
X_train_trans_gender_city
# gender and city are nominal columns we will use OneHotEncoder
ohe = OneHotEncoder(drop='first',sparse=False)

X_train_trans_gender_city = ohe.fit_transform(X_train[['gender','city']])
X_test_trans_gender_city = ohe.fit_transform(X_test[['gender','city']])

X_train_trans_gender_city.shape
(80, 4)
X_test_age
# Extract Age Column
X_train_age = X_train[['age']].values
X_test_age = X_test[['age']].age.values
X_train_age.shape
(80, 1)
# Now the final Transformed data is
X_train_transformed = np.concatenate((X_train_age,train_null_free_fever, X_train_transformed_cough,X_train_trans_gender_city),axis=1)
# X_test_transformed = np.concatenate((X_test_age,test_null_free_fever, X_test_transformed_cough,X_test_trans_gender_city),axis=1)
.shape
X_train_transformed.shape
(80, 7)
# X_train_transformed
How to Solve Above Difficult Task with ColumnTransformer Within One Line
# Use Column Transformer
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([
    ("imputer", SimpleImputer(),['fever']),
    ("odinalencoder", OrdinalEncoder(categories=[['Mild','Strong']]),['cough']),
     ("onehotencoder",OneHotEncoder(drop='first',sparse=False),['gender','city'])],remainder='passthrough')
X_train_transformed
X_train_transformed = ct.fit_transform(X_train)
shape
X_train_transformed.shape
(80, 7)

```