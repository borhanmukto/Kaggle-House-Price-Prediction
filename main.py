import pandas as pd
import pickle
import streamlit as st

st.title("House Sale Price Prediction System")
st.write("Enter the specifications of the house in the following table")

voting_regressor = pickle.load(open("voting_regressor.pkl", "rb"))

test_df_dict = pickle.load(open("test_df.pkl", "rb"))

test_df = pd.DataFrame(test_df_dict)

columns = pickle.load(open("columns.pkl", "rb"))

X_train_dict = pickle.load(open("X_train.pkl", "rb"))

X = pd.DataFrame(X_train_dict)

st.sidebar.header('Specify Input Parameters')

test_df.loc[0, "1stFlrSF"] = st.sidebar.slider('First Floor Area in SqFt', X["1stFlrSF"].min(), X["1stFlrSF"].max(), 2000)
test_df.loc[0, "2ndFlrSF"] = st.sidebar.slider('Second Floor Area in SqFt', X["2ndFlrSF"].min(), X["2ndFlrSF"].max(), 1000)
test_df.loc[0, "BedroomAbvGr"] = st.sidebar.slider('Bedroom Above Grade', X["BedroomAbvGr"].min(), X["BedroomAbvGr"].max(), 4)
test_df.loc[0, "BsmtFinSF1"] = st.sidebar.slider('Basement Finished SqFt', X["BsmtFinSF1"].min(), X["BsmtFinSF1"].max(), 2700, 1)
test_df.loc[0, "BsmtFullBath"] = st.sidebar.slider('Bsmt Full Bathroom', X["BsmtFullBath"].min(), X["BsmtFullBath"].max(), 2, 1)
test_df.loc[0, "BsmtUnfSF"] = st.sidebar.slider('Basement Unfinished SqFt', X["BsmtUnfSF"].min(), X["BsmtUnfSF"].max(), 1100)
test_df.loc[0, "EnclosedPorch"] = st.sidebar.slider('Enclosed Porch Area', X["EnclosedPorch"].min(), X["EnclosedPorch"].max(), 265, 1)
test_df.loc[0, "Fireplaces"] = st.sidebar.slider('Number of Fireplaces', X["Fireplaces"].min(), X["Fireplaces"].max(), 1)
test_df.loc[0, "FullBath"] = st.sidebar.slider('Full Bathrooms', X["FullBath"].min(), X["FullBath"].max(), 1)
test_df.loc[0, "GarageArea"] = st.sidebar.slider('Garage Area SqFt', X["GarageArea"].min(), X["GarageArea"].max(), 657, 1)
test_df.loc[0, "GarageCars"] = st.sidebar.slider('Garage Cars', X["GarageCars"].min(), X["GarageCars"].max(), 1, 1)
test_df.loc[0, "GarageYrBlt"] = st.sidebar.slider('Garage Year Built', X["GarageYrBlt"].min(), X["GarageYrBlt"].max(), 1995, 1)
test_df.loc[0, "GrLivArea"] = st.sidebar.slider('GrLivArea', X["GrLivArea"].min(), X["GrLivArea"].max(), int(X["GrLivArea"].median()), 1)
test_df.loc[0, "HalfBath"] = st.sidebar.slider('HalfBath', X["HalfBath"].min(), X["HalfBath"].max(), 1, 1)
test_df.loc[0, "KitchenAbvGr"] = st.sidebar.slider('KitchenAbvGr', X["KitchenAbvGr"].min(), X["KitchenAbvGr"].max(), int(X["KitchenAbvGr"].median()), 1)
test_df.loc[0, "LotArea"] = st.sidebar.slider('LotArea', X["LotArea"].min(), X["LotArea"].max(), int(X["LotArea"].median()), 1)
test_df.loc[0, "LotFrontage"] = st.sidebar.slider('LotFrontage', X["LotFrontage"].min(), X["LotFrontage"].max(), int(X["LotFrontage"].median()), 1)
test_df.loc[0, "MasVnrArea"] = st.sidebar.slider('MasVnrArea', X["MasVnrArea"].min(), X["MasVnrArea"].max(), 800, 1)
test_df.loc[0, "OpenPorchSF"] = st.sidebar.slider('OpenPorchSF', X["OpenPorchSF"].min(), X["OpenPorchSF"].max(), 347, 1)
test_df.loc[0, "OverallQual"] = st.sidebar.slider('OverallQual', X["OverallQual"].min(), X["OverallQual"].max(), int(X["OverallQual"].median()), 1)
test_df.loc[0, "TotRmsAbvGrd"] = st.sidebar.slider('TotRmsAbvGrd', X["TotRmsAbvGrd"].min(), X["TotRmsAbvGrd"].max(), int(X["OverallQual"].median()), 1)
test_df.loc[0, "TotalBsmtSF"] = st.sidebar.slider('TotalBsmtSF', X["TotalBsmtSF"].min(), X["TotalBsmtSF"].max(), int(X["TotalBsmtSF"].median()), 1)
test_df.loc[0, "WoodDeckSF"] = st.sidebar.slider('WoodDeckSF', X["WoodDeckSF"].min(), X["WoodDeckSF"].max(), 500, 1)
test_df.loc[0, "YearBuilt"] = st.sidebar.slider('YearBuilt', X["YearBuilt"].min(), X["YearBuilt"].max(), int(X["YearBuilt"].median()), 1)
test_df.loc[0, "YearRemodAdd"] = st.sidebar.slider('YearRemodAdd', X["YearRemodAdd"].min(), X["YearRemodAdd"].max(), int(X["YearRemodAdd"].median()), 1)

st.dataframe(test_df)

predict = st.button("Predict Saleprice", type="primary")

if predict is True:
    y_pred = voting_regressor.predict(test_df)
    # Round the first value to 3 decimal places
    rounded_value = round(y_pred[0], 3)

    # Convert the rounded value to a string
    rounded_value_string = str(rounded_value)

    # Add a $ sign to the beginning of the string
    rounded_value_with_dollar_sign = "$" + rounded_value_string

    # Print the rounded value with the dollar sign
    st.subheader(rounded_value_with_dollar_sign)
    st.caption('This model is trained on Random Forest, Adaptive Boost and XGBoost Regressor on Boston House Price Dataset')
