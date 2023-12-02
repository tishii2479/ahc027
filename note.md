# 考察メモ

- 解が出る方針ではなく、良い解が出る方針を探す

# 12/2

## 観察

- なるべく等間隔に訪れるのが一番良い

- 間隔の配分
    ```
    minimize a1 * (T / d1) ^ 2 + a2 * (T / d2) ^ 2
    r = pow(d1 / d2, 1 / 3) / (1 + pow(d1 / d2, 1 / 3)) : 1 - r となるrが最善
    ```
- 3変数の場合
    ```
    r1 = (r1 + r3) * np.power(a1 / a3, 1/3) / (1 + np.power(a1 / a3, 1/3))
    r2 = (r2 + r3) * np.power(a2 / a3, 1/3) / (1 + np.power(a2 / a3, 1/3))
    r3 = r3

    r1 * (1 + np.power(a1 / a3, 1/3)) = (r1 + r3) * np.power(a1 / a3, 1/3)
    r1 * (1 + np.power(a1 / a3, 1/3) - np.power(a1 / a3, 1/3)) = r3 * np.power(a1 / a3, 1/3)

    r1 = r3 * np.power(a1 / a3, 1/3)
    r2 = r3 * np.power(a2 / a3, 1/3)
    r3 = r3

    r1 + r2 + r3 = 1
    r3 * (\sum_i (a_i / a_3) ^ 1/3) = 1
    r3 = 1 / (\sum_i (a_i / a_3) ^ 1/3)
    ```
- n変数の場合
    ```
    minimize \sum_i a_i / r_i ^ 2
    r_n = 1 / (\sum_i (a_i / a_n) ^ 1/3)
    r_i = r_n * (a_i / a_n) ^ 1/3
    ```
- なるべく回数がrの比率に近い経路を求めることが最善になる
```
r  = 0.2, 0.1, 0.05 * 14
t = 1/0.2, 1/0.1, 1/0.05
t = 5, 10, 20
L = 100 = 20 + 10 + 5 * 14
```

## 方針案

- 初期解がんばる
- 焼きなましで頑張る
- 良い評価関数のビームサーチで頑張る

## メモ

- 各マスを訪れる時刻を保存する？
- 割と高速に差分計算はできそう
    - なら一旦焼いてみるのもありそう
    - 局所解から抜けづらそうなので、
        - 根本的に違う解法（マップを作る、など）
        - 初期解が重要な場合
    - を常に意識する
- 完全情報なので、ちゃんと局所探索できるならビームサーチより焼きなましの方が強そう
    - パスに後から挿入することはできる
- 良い評価関数を組めば貪欲でもある程度戦えそう
- 遅れる時のペナルティはどうやって計算する？
- スタート、繋げる時が難しい気がする

## 貪欲

- 間隔が空いているところを掃除しに行きたい
- 現在の間隔をt、理想的な間隔をt'、距離をdとして、a^(t+d-t')が大きいところを掃除しに行く
    - タイブレークも同じ指標を使用
- 探索は毎回距離を計算するの、大変だね