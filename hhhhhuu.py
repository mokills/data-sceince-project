import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

def BarChart_Borough(df):
    df = pd.DataFrame(df)
    temp = df.groupby(['BORO', 'TENURE']).size().unstack(fill_value=0)
    temp.plot(kind='bar')
    plt.title('Renters vs Owners by Borough')
    plt.xlabel('BORO')
    plt.ylabel('Number of Units')
    plt.tight_layout()
    plt.show()

def HistChart(df):
    df = pd.DataFrame(df)
    renters = df[df['TENURE'] == 1]['HHINC_REC1']
    owners = df[df['TENURE'] == 2]['HHINC_REC1']
    plt.hist(renters, bins=30, alpha=0.2, label='Renter')
    plt.hist(owners, bins=30, alpha=0.2, label='Owner')
    plt.title('Income Distribution: Renters vs Owners')
    plt.xlabel('Income')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

def BoxPlots(df):
    df = pd.DataFrame(df)
    sns.boxplot(data=df, x='borough', y='rent')
    plt.title('Rent Prices Across Boroughs')
    plt.tight_layout()
    plt.show()
    sns.boxplot(data=df, x='borough', y='home_price')
    plt.title('Home Prices Across Boroughs')
    plt.tight_layout()
    plt.show()

def NYC_Borough_Map(df, save_path='nyc_borough_renter_map.html'):
    df = pd.DataFrame(df)
    df = df.dropna(subset=['TENURE', 'BORO'])

    temp = df.groupby(['BORO', 'TENURE']).size().unstack(fill_value=0)
    if 1 not in temp.columns:
        print("TENURE == 1 (renter) not found in data.")
        return

    temp['renter_share'] = temp[1] / temp.sum(axis=1)

    boro_map = {
        1: "Manhattan",
        2: "Bronx",
        3: "Brooklyn",
        4: "Queens",
        5: "Staten Island"
    }
    temp['borough'] = temp.index.map(boro_map)
    temp = temp.dropna(subset=['borough'])

    borough_geojson = "https://data.dathere.com/dataset/1610c71e-b33e-481b-8a3d-96f69d8e5a7f/resource/6f1e4cfc-132b-413c-b1e9-48d8e884346a/download/42c737fd496f4d6683bba25fb0e86e1dnycboroughboundaries.geojson"

    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)

    folium.Choropleth(
        geo_data=borough_geojson,
        data=temp,
        columns=['borough', 'renter_share'],
        key_on='feature.properties.borough',
        fill_color='YlOrRd',
        fill_opacity=0.8,
        line_opacity=0.4,
        legend_name='Renter Share by Borough'
    ).add_to(m)

    m.save(save_path)
    print("Saved map file:", save_path)

def FeatureImportanceChart(model, feature_names, title):
    importances = model.feature_importances_
    temp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    temp.plot(kind='bar')
    plt.title(title)
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.show()

def ConfMatrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Owner', 'Pred Renter'],
                yticklabels=['Actual Owner', 'Actual Renter'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def ROC_Curve(model, X_test, y_test, title):
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()

def RunDecisionTree(df):
    df = pd.DataFrame(df)
    df = df.dropna(subset=['TENURE', 'HHINC_REC1', 'BORO'])
    df['BORO_CODE'] = pd.Categorical(df['BORO']).codes
    features = ['HHINC_REC1', 'BORO_CODE']
    X = df[features]
    y = (df['TENURE'] == 1).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Decision Tree accuracy:", (y_pred == y_test).mean())
    FeatureImportanceChart(model, features, 'Decision Tree Feature Importance')
    ConfMatrix(y_test, y_pred, 'Confusion Matrix (Decision Tree)')
    ROC_Curve(model, X_test, y_test, 'ROC Curve (Decision Tree)')

def RunRandomForest(df):
    df = pd.DataFrame(df)
    df = df.dropna(subset=['TENURE', 'HHINC_REC1', 'BORO'])
    df['BORO_CODE'] = pd.Categorical(df['BORO']).codes
    features = ['HHINC_REC1', 'BORO_CODE']
    X = df[features]
    y = (df['TENURE'] == 1).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Random Forest accuracy:", (y_pred == y_test).mean())
    FeatureImportanceChart(model, features, 'Random Forest Feature Importance')
    ConfMatrix(y_test, y_pred, 'Confusion Matrix (Random Forest)')
    ROC_Curve(model, X_test, y_test, 'ROC Curve (Random Forest)')

def main():
    csv_pathO = r"C:\Users\student\Desktop\FP\occupied.csv"
    csv_pathA = r"C:\Users\student\Desktop\FP\allunits_puf.csv"

    dfO = pd.read_csv(csv_pathO)
    dfA = pd.read_csv(csv_pathA)
    df = pd.merge(dfO, dfA, how="left")

    #HistChart(df)
    #BarChart_Borough(df)
    NYC_Borough_Map(df)
    #RunDecisionTree(df)
    #RunRandomForest(df)
    #BoxPlots(df)

    print("Current working directory:", os.getcwd())

if __name__ == '__main__':
    main()
