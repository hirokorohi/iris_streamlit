# 基本ライブラリ
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# データセット読み込み
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 目標値
df['target'] = iris.target

# 目標値を数字から花の名前に変換
df['target'] = df['target'].astype(str)# 'target'列を文字列型に変換
df.loc[df['target'] == 0, 'target'] = 'setosa'# df['target']列が0の行の'target'を'setosa'に変換
df.loc[df['target'] == 1, 'target'] = 'varicolor'
df.loc[df['target'] == 2, 'target'] = 'virginica'

# 予測モデルの構築
x = iris.data[:, [0, 2]]# 全ての行の0列目(sepal length)と2列目(petal length)を取得
y = iris.target

# ロジスティック回帰
clf = LogisticRegression()
clf.fit(x, y)

# サイドバーの入力画面
st.sidebar.header("Input features")

sepalValue = st.sidebar.slider('sepal length (cm)', min_value=0.0, max_value=10.0, step=0.1)
petalValue = st.sidebar.slider('petal length (cm)', min_value=0.0, max_value=10.0, step=0.1)

# メインパネル
st.title('Iris Classifier')
st.write('## Input value')

# インプットデータ（1行のデータフレーム）
value_df = pd.DataFrame(columns=['type', 'sepal length (cm)', 'petal length (cm)'])
record = pd.Series(['data', sepalValue, petalValue], index=value_df.columns)
value_df = pd.concat([value_df, record.to_frame().T], ignore_index=True)

# インデックスを設定して表示
value_df.set_index('type', inplace=True)
st.write(value_df)

# 予測値のデータフレーム
pred_probs = clf.predict_proba(value_df)
pred_df = pd.DataFrame(pred_probs,columns=['setosa','versicolor','virginica'],index=['probability'])

st.write('## Prediction')
st.write(pred_df)

# 予測結果の出力
name = pred_df.idxmax(axis=1).tolist()
st.write('## Result')
st.write('このアイリスはきっと',str(name[0]),'です!')