# Overnight 论文基础框架

这是独立的英文 LaTeX 初稿，不覆盖旧版 `paper/manuscript.tex`。内容针对当前 Overnight / Anchored，不延续旧版 Hard-Concrete、BIC 或离线 teacher 描述。

## 文件与状态

- `manuscript.tex`：问题、方法、训练、实验协议、待填结果、局限及结论；暂不放图。
- `references.bib`：五个对照论文及三个真实数据来源的基础书目。
- [结果审计](../../docs/overnight_results_audit_20260921.md)：本次服务器结果的数字、来源、失败和解释边界。

使用通用 `article` 类，不依赖 `elsarticle`。这不是 CAD 最终排版或投稿合规声明；内容稳定后再转入期刊当时提供的模板。作者、基金、声明、最终实验表和学习领域相关文献仍需补充。

框架明确区分：

1. **已有训练记录**：局部注意力 Proposal、动态 beta / probability-mass TopK、KeepMask 条件化存活几何、参数与节点一起更新、标准 LS、在线反事实 teacher。
2. **新可选实验，尚无完整成绩**：`proposal_refinement_layers`、`selection_refinement_layers`、`survivor_refinement_layers`（底层默认均0，新增块零门控初始化）；逐点峰值/上尾损失；M16/M32两个通用容量模型。当前两模型实验默认各加2层并启用峰值损失，不按数据集另训专用模型。不得写成已经证实提高精度。
3. **没有理论或实验证实的承诺**：阈值必达、全局最少节点、所有对照均更慢、连续曲线最大误差有界。

Keep embedding 的准确描述：由 KeepMask 决定的邻距、rank、数量形成 geometry embedding，并屏蔽被删除节点的 K/V；不是凭空存在的独立二值 embedding 查表。

两模型运行协议见 [M16/M32](../../docs/overnight_m16_m32.md)。默认M16的有标签合成源K4..16，M32为K4..24，因此不是纯容量消融；后者需显式使用共同K4..16训练范围。两者测试同一五来源，六方法均采用当前模型的容量上限，分表报告；不按测试误差逐条挑选模型，也不将中断的来源专用训练结果改标为通用模型。

## 编译

在装有 TeX Live / MiKTeX、BibTeX 的环境中：

```bash
cd paper/overnight
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

本机检查未找到 `pdflatex` 或 `latexmk`，本次只交付源文件并做静态检查，**未编译、未进行 PDF 页面排版验收**，没有生成最终投稿PDF。

## 书目核对：2026-09-21

基本信息使用原始发布机构/作者机构及 DOI 注册记录核对；部分全文受限，不能据此声称完整复现。适配范围以 [复现说明](../../docs/published_knot_methods_reproduction.md) 为准。

| 条目 | 核对来源 | 备注 |
|---|---|---|
| Park & Lee 2007 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010448507000024)、[作者机构ETRI](https://ksp.etri.re.kr/ksp/article/read?id=42770) | 39(6):439–451；DOI 10.1016/j.cad.2006.12.006 |
| Liang et al. 2017 | [作者苏州大学论文目录](https://jdxy.suda.edu.cn/df/0d/c30328a515853/page.htm)、[Crossref出版者注册元数据](https://api.crossref.org/works/10.1088/1361-6501/aa6a05) | 28(6):065015；IOP页面不能直接读取，未声称核完全文 |
| Dung & Tjahjowidodo 2017 | [PLOS原文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0173857) | 12(3):e0173857；DOI 10.1371/journal.pone.0173857 |
| Kang et al. 2015 | [ScienceDirect DOI页](https://doi.org/10.1016%2Fj.cad.2014.08.022) | 58:179–188；Knot calculation for spline fitting via sparse optimization |
| Luo et al. 2022 | [期刊官网](https://computmath.cjoe.ac.cn/jcm/EN/lexeme/showArticleByLexeme.do?articleID=52047)、[出版社目录](https://www.global-sci.com/JCM/issue/view/1120) | 40(4):589–606；采用目录2022，某下载PDF首页年份不一致；不是另一篇DNN论文 |
| UJI Version 2 | [UCI官方页](https://archive.ics.uci.edu/dataset/177/uji%2Bpen%2Bcharacters%2Bversion%2B2) | 官方数据DOI 10.24432/C5FG8S |
| Natural Earth | [1:10m coastline官方页](https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/) | 1:10m是地图比例尺，不是10米分辨率；实际版本以manifest为准 |
| USGS | [官方Contours服务](https://carto.nationalmap.gov/arcgis/rest/services/contours/MapServer)、[USGS下载说明](https://www.usgs.gov/the-national-map-data-delivery/gis-data-download) | 提取区域、scale、split需附manifest |

工业等距线是程序生成的 CAD 风格曲线，没有虚构外部数据文献，也不能写成真实测量工业数据。
