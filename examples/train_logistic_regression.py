"""Demonstrate logistic regression on a synthetic clinical CSV."""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',required=True)
    parser.add_argument('--schema',required=True)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    frame=pd.read_csv(args.csv)
    schema=pd.read_csv(args.schema).fillna('')
    X,y=frame.iloc[:,:-1],frame.iloc[:,-1]
    if y.value_counts().min()<10 or y.nunique()!=2:
        parser.error('Generate a larger cohort: at least 10 examples of each class are required.')
    if 'Field_ID' in schema:
        categorical=[str(row.Field_ID) for row in schema.itertuples() if str(row.Recommended_Allowed_Values)]
    elif 'native_name' in schema:
        categorical=schema.loc[schema['type']=='categorical','native_name'].tolist()
    else:
        categorical=schema.loc[schema['type']=='categorical','field_id'].astype(str).tolist()
    categorical=[c for c in categorical if c in X]
    numeric=[c for c in X if c not in categorical]
    train,holdout,y_train,y_holdout=train_test_split(X,y,test_size=.4,stratify=y,random_state=args.seed)
    val,test,y_val,y_test=train_test_split(holdout,y_holdout,test_size=.5,stratify=y_holdout,random_state=args.seed)
    best=None
    for C in [.01,.1,1,10]:
        pre=ColumnTransformer([
            ('numeric',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),numeric),
            ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),('encode',OneHotEncoder(handle_unknown='ignore'))]),categorical)])
        model=Pipeline([('preprocess',pre),('classifier',LogisticRegression(C=C,max_iter=2000))])
        model.fit(train,y_train)
        scores=model.predict_proba(val)[:,1]
        auc=roc_auc_score(y_val,scores)
        if best is None or auc>best[0]:best=(auc,C,model,scores)
    _,C,model,val_scores=best
    thresholds=np.linspace(.01,.99,99)
    threshold=float(max(thresholds,key=lambda t:f1_score(y_val,val_scores>=t,zero_division=0)))
    scores=model.predict_proba(test)[:,1]
    print(json.dumps({'evaluation':'held-out synthetic demonstration, not real-cohort validation',
        'training_rows':len(train),'validation_rows':len(val),'test_rows':len(test),
        'C':C,'threshold':threshold,'test_auc':roc_auc_score(y_test,scores),
        'test_f1':f1_score(y_test,scores>=threshold,zero_division=0)},indent=2))

if __name__=='__main__':
    main()
