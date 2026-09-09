# Sa_displacedge — 蜃気楼エッジ (Mirage Edge)

輪郭を液状ガラスのように渦を巻かせて屈折させる、YukkuriMovieMaker4(YMM4)用の映像エフェクトプラグインです。

輪郭検出そのものは[Sa_aohue](https://github.com/sabiasagimp4-ai/Sa_aohue)のYMM4実装を参考にしていますが、狙いは正反対です。Sa_aohueは輪郭に沿って**色**(彩度・輝度・色相)を変えるのに対し、Sa_displacedgeは輪郭に沿って**画素の位置**を発散ゼロの渦ノイズで変位させ、色収差(プリズム縁取り)と薄膜干渉風の虹色ハイライトを重ねます。ブラー・グロー・単純な色収差・既存の変位マップ(UVをそのままずらすだけのもの)とは異なる、「輪郭ガイド付きの液体屈折」という新しい組み合わせです。

YMM4版の詳細(UI・アルゴリズム・ビルド方法)は[`ymm/README.md`](ymm/README.md)を参照してください。GitHub Actionsの`YMM4 build`で、最新の公式YMM4 Liteを取得してDLLと配布ZIPまで生成します。

## 見た目

添付画像に既定パラメータを適用した例です(他のパラメータ例は[`docs/PIPELINE.md`](docs/PIPELINE.md)を参照)。

| 元画像 | 適用後(既定値) |
|---|---|
| ![](docs/pipeline_examples/00_input.png) | ![](docs/pipeline_examples/01_composite_default.png) |

## アルゴリズムの要約

1. OKLab LへのScharr勾配で輪郭方向・強さを検出(Sa_aohueと同じOKLab Lベース)。
2. 生の勾配場をガウシアンでぼかし、輪郭からの距離に応じてなめらかに減衰する場を作る(Sa_aohueのRadiusぼかしと同じ三重ボックスブラー近似を`prototype/preview.py`側で共有)。
3. 回転オクターブ付き値ノイズfBmから発散ゼロ(渦のみ、湧き出し・吸い込みなし)のカール・ノイズを作り、輪郭の外向き方向とユーザー指定の比率で混ぜて流れベクトルを得る。
4. 流れベクトルに沿って元画像をR/G/Bそれぞれ異なる距離でサンプリングし(色収差)、輪郭の法線と光源角度から求めたリムライトを虹色パレットで着色して加算する。

詳しい設計判断とパラメータ別の見た目比較は[`docs/PIPELINE.md`](docs/PIPELINE.md)、UIと実装ファイルの対応は[`ymm/README.md`](ymm/README.md)にあります。

## リポジトリ構成

```
prototype/preview.py     NumPy参照実装。GPU無しでアルゴリズムを事前確認できる(Sa_aohueのprototype/と同じ役割)
tests/test_prototype.py  preview.pyの数値契約テスト(カール・ノイズの発散ゼロ性など)
ymm/                     YMM4プラグイン本体(C# + HLSL)。詳細はymm/README.md
docs/PIPELINE.md         パイプライン解説とパラメータ比較画像
docs/pipeline_examples/  上記で使う実画像の処理結果
.github/workflows/       YMM4ビルドCI(Windows runner、公式YMM4 Lite + fxcを取得してビルド)
```

## クイックスタート(プレビュー)

```bash
pip install numpy pillow
python prototype/preview.py input.png output.png                 # 既定パラメータで合成
python prototype/preview.py input.png flow.png --mode flow        # 流れベクトルを可視化
python prototype/preview.py input.png mask.png --mode mask        # 変位の強さマスクを可視化
python -m unittest tests.test_prototype -v                        # 数値契約テスト
```

YMM4への導入は[`ymm/README.md`](ymm/README.md)の「ビルド・インストール」章を参照してください。
