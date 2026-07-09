# Active Learning Models Evaluation

This repository contains the evaluation results of YOLO, Faster R-CNN, and RT-DETR models across 5 active learning cycles (0-4), as well as a MegaDetector baseline.

## 1. Detection Level Metrics (mAP)

| Model             | Cycle   | Dataset   |    mAP50 |   mAP50-95 |   Other_Amphibian_AP50 |   Small_Mammal_AP50 |   WLT_AP50 |
|:------------------|:--------|:----------|---------:|-----------:|-----------------------:|--------------------:|-----------:|
| faster_rcnn_clahe | cycle_0 | test_full | 0.771178 |  0.572232  |               0.685704 |           0.388967  |   0.642024 |
| faster_rcnn_clahe | cycle_1 | test_full | 0.838621 |  0.606298  |               0.692896 |           0.410644  |   0.715355 |
| faster_rcnn_clahe | cycle_2 | test_full | 0.900046 |  0.605735  |               0.687162 |           0.488119  |   0.641923 |
| faster_rcnn_clahe | cycle_3 | test_full | 0.915007 |  0.697241  |               0.741204 |           0.652475  |   0.698044 |
| faster_rcnn_clahe | cycle_4 | test_full | 0.862818 |  0.595786  |               0.738893 |           0.345875  |   0.70259  |
| megadetector      | nan     | test_full | 0        |  0         |              -1        |           0         |   0        |
| rtdetr_clahe      | cycle_0 | test_full | 0.530341 |  0.297089  |               0.176274 |           0.213201  |   0.501793 |
| rtdetr_clahe      | cycle_1 | test_full | 0.399833 |  0.265491  |               0.220234 |           0.0308911 |   0.545349 |
| rtdetr_clahe      | cycle_2 | test_full | 0.541863 |  0.374005  |               0.444236 |           0.115842  |   0.561936 |
| rtdetr_clahe      | cycle_3 | test_full | 0.821229 |  0.466094  |               0.41427  |           0.438119  |   0.545893 |
| rtdetr_clahe      | cycle_4 | test_full | 0.696163 |  0.394386  |               0.250731 |           0.457921  |   0.474507 |
| yolo_clahe        | cycle_0 | test_full | 0.703014 |  0.463786  |               0.636746 |           0.215842  |   0.538771 |
| yolo_clahe        | cycle_1 | test_full | 0.680777 |  0.451284  |               0.585116 |           0.278218  |   0.490518 |
| yolo_clahe        | cycle_2 | test_full | 0.24488  |  0.179293  |               0.274342 |           0         |   0.263537 |
| yolo_clahe        | cycle_3 | test_full | 0.591221 |  0.386805  |               0.575981 |           0.0343234 |   0.550112 |
| yolo_clahe        | cycle_4 | test_full | 0.666344 |  0.352461  |               0.341332 |           0.333333  |   0.382717 |
| faster_rcnn_clahe | cycle_0 | val_full  | 0.822762 |  0.641328  |               0.655384 |           0.8       |   0.468599 |
| faster_rcnn_clahe | cycle_1 | val_full  | 0.877309 |  0.629124  |               0.710938 |           0.60198   |   0.574455 |
| faster_rcnn_clahe | cycle_2 | val_full  | 0.827635 |  0.632413  |               0.681148 |           0.750495  |   0.465596 |
| faster_rcnn_clahe | cycle_3 | val_full  | 0.890266 |  0.630955  |               0.738286 |           0.552475  |   0.602105 |
| faster_rcnn_clahe | cycle_4 | val_full  | 0.888272 |  0.627549  |               0.754981 |           0.552475  |   0.575189 |
| megadetector      | nan     | val_full  | 0        |  0         |              -1        |           0         |   0        |
| rtdetr_clahe      | cycle_0 | val_full  | 0.428161 |  0.186889  |               0.104918 |           0.0504951 |   0.405253 |
| rtdetr_clahe      | cycle_1 | val_full  | 0.642789 |  0.29649   |               0.156955 |           0.325248  |   0.407268 |
| rtdetr_clahe      | cycle_2 | val_full  | 0.578263 |  0.319181  |               0.341936 |           0.151485  |   0.464123 |
| rtdetr_clahe      | cycle_3 | val_full  | 0.576918 |  0.311711  |               0.295731 |           0.151485  |   0.487917 |
| rtdetr_clahe      | cycle_4 | val_full  | 0.485495 |  0.275866  |               0.222199 |           0.166794  |   0.438606 |
| yolo_clahe        | cycle_0 | val_full  | 0.625297 |  0.456549  |               0.542132 |           0.353465  |   0.474051 |
| yolo_clahe        | cycle_1 | val_full  | 0.583121 |  0.395136  |               0.498825 |           0.252475  |   0.434109 |
| yolo_clahe        | cycle_2 | val_full  | 0.12025  |  0.0869341 |               0.159491 |           0         |   0.101312 |
| yolo_clahe        | cycle_3 | val_full  | 0.497692 |  0.327227  |               0.582051 |           0         |   0.39963  |
| yolo_clahe        | cycle_4 | val_full  | 0.399445 |  0.265478  |               0.192933 |           0.252475  |   0.351027 |

## 2. Image Level Metrics (WLT vs Background)

| Model             | Cycle   | Dataset   |      AUC |    Best_F1 |   Best_Threshold |   Precision |    Recall |   TP |   FP |   FN |     TN |
|:------------------|:--------|:----------|---------:|-----------:|-----------------:|------------:|----------:|-----:|-----:|-----:|-------:|
| faster_rcnn_clahe | cycle_0 | test_full | 0.990128 | 0.404959   |         0.924969 |  0.304348   | 0.604938  |   49 |  112 |   32 | 162093 |
| faster_rcnn_clahe | cycle_1 | test_full | 0.985624 | 0.465517   |         0.923234 |  0.357616   | 0.666667  |   54 |   97 |   27 | 162108 |
| faster_rcnn_clahe | cycle_2 | test_full | 0.983184 | 0.413793   |         0.912006 |  0.3        | 0.666667  |   54 |  126 |   27 | 162079 |
| faster_rcnn_clahe | cycle_3 | test_full | 0.985653 | 0.488      |         0.730649 |  0.360947   | 0.753086  |   61 |  108 |   20 | 162097 |
| faster_rcnn_clahe | cycle_4 | test_full | 0.998476 | 0.516393   |         0.818993 |  0.386503   | 0.777778  |   63 |  100 |   18 | 162105 |
| megadetector      | nan     | test_full | 0.552901 | 0.00550206 |         0.482096 |  0.00309598 | 0.0246914 |    2 |  644 |   79 | 161561 |
| rtdetr_clahe      | cycle_0 | test_full | 0.991578 | 0.449541   |         0.689004 |  0.357664   | 0.604938  |   49 |   88 |   32 | 162117 |
| rtdetr_clahe      | cycle_1 | test_full | 0.969284 | 0.423729   |         0.40139  |  0.322581   | 0.617284  |   50 |  105 |   31 | 162100 |
| rtdetr_clahe      | cycle_2 | test_full | 0.982766 | 0.415385   |         0.893198 |  0.301676   | 0.666667  |   54 |  125 |   27 | 162080 |
| rtdetr_clahe      | cycle_3 | test_full | 0.966677 | 0.427948   |         0.612178 |  0.331081   | 0.604938  |   49 |   99 |   32 | 162106 |
| rtdetr_clahe      | cycle_4 | test_full | 0.956084 | 0.348148   |         0.343832 |  0.248677   | 0.580247  |   47 |  142 |   34 | 162063 |
| yolo_clahe        | cycle_0 | test_full | 0.936256 | 0.364821   |         0.55399  |  0.247788   | 0.691358  |   56 |  170 |   25 | 162035 |
| yolo_clahe        | cycle_1 | test_full | 0.948633 | 0.333333   |         0.42614  |  0.223629   | 0.654321  |   53 |  184 |   28 | 162021 |
| yolo_clahe        | cycle_2 | test_full | 0.840543 | 0.4        |         0.626969 |  0.5        | 0.333333  |   27 |   27 |   54 | 162178 |
| yolo_clahe        | cycle_3 | test_full | 0.919249 | 0.524017   |         0.320939 |  0.405405   | 0.740741  |   60 |   88 |   21 | 162117 |
| yolo_clahe        | cycle_4 | test_full | 0.894019 | 0.352941   |         0.726405 |  0.375      | 0.333333  |   27 |   45 |   54 | 162160 |
| faster_rcnn_clahe | cycle_0 | val_full  | 0.948771 | 0.545455   |         0.94405  |  0.913043   | 0.388889  |   21 |    2 |   33 | 163697 |
| faster_rcnn_clahe | cycle_1 | val_full  | 0.969489 | 0.613861   |         0.865304 |  0.659574   | 0.574074  |   31 |   16 |   23 | 163683 |
| faster_rcnn_clahe | cycle_2 | val_full  | 0.899348 | 0.56       |         0.950068 |  1          | 0.388889  |   21 |    0 |   33 | 163699 |
| faster_rcnn_clahe | cycle_3 | val_full  | 0.969931 | 0.651163   |         0.841644 |  0.875      | 0.518519  |   28 |    4 |   26 | 163695 |
| faster_rcnn_clahe | cycle_4 | val_full  | 0.979088 | 0.618557   |         0.761764 |  0.697674   | 0.555556  |   30 |   13 |   24 | 163686 |
| megadetector      | nan     | val_full  | 0.558846 | 0.0606061  |         0.820732 |  0.166667   | 0.037037  |    2 |   10 |   52 | 163689 |
| rtdetr_clahe      | cycle_0 | val_full  | 0.917053 | 0.574713   |         0.691824 |  0.757576   | 0.462963  |   25 |    8 |   29 | 163691 |
| rtdetr_clahe      | cycle_1 | val_full  | 0.923203 | 0.55814    |         0.63069  |  0.75       | 0.444444  |   24 |    8 |   30 | 163691 |
| rtdetr_clahe      | cycle_2 | val_full  | 0.940541 | 0.593407   |         0.887772 |  0.72973    | 0.5       |   27 |   10 |   27 | 163689 |
| rtdetr_clahe      | cycle_3 | val_full  | 0.940178 | 0.673684   |         0.674893 |  0.780488   | 0.592593  |   32 |    9 |   22 | 163690 |
| rtdetr_clahe      | cycle_4 | val_full  | 0.899559 | 0.626263   |         0.595115 |  0.688889   | 0.574074  |   31 |   14 |   23 | 163685 |
| yolo_clahe        | cycle_0 | val_full  | 0.895388 | 0.506329   |         0.703601 |  0.8        | 0.37037   |   20 |    5 |   34 | 163694 |
| yolo_clahe        | cycle_1 | val_full  | 0.904863 | 0.378378   |         0.619643 |  0.7        | 0.259259  |   14 |    6 |   40 | 163693 |
| yolo_clahe        | cycle_2 | val_full  | 0.640693 | 0.097561   |         0.551874 |  0.142857   | 0.0740741 |    4 |   24 |   50 | 163675 |
| yolo_clahe        | cycle_3 | val_full  | 0.823616 | 0.634146   |         0.502444 |  0.928571   | 0.481481  |   26 |    2 |   28 | 163697 |
| yolo_clahe        | cycle_4 | val_full  | 0.7771   | 0.5        |         0.324311 |  0.605263   | 0.425926  |   23 |   15 |   31 | 163684 |

## 3. Image Level Metrics (Class Agnostic vs Background)

| Model             | Cycle   | Dataset   |      AUC |    Best_F1 |   Best_Threshold |   Precision |    Recall |   TP |     FP |   FN |     TN |
|:------------------|:--------|:----------|---------:|-----------:|-----------------:|------------:|----------:|-----:|-------:|-----:|-------:|
| faster_rcnn_clahe | cycle_0 | test_full | 0.985969 | 0.534997   |        0.896631  |  0.542135   | 0.528044  |  386 |    326 |  345 | 161229 |
| faster_rcnn_clahe | cycle_1 | test_full | 0.970362 | 0.593076   |        0.811183  |  0.5675     | 0.621067  |  454 |    346 |  277 | 161209 |
| faster_rcnn_clahe | cycle_2 | test_full | 0.973712 | 0.481781   |        0.737454  |  0.475366   | 0.488372  |  357 |    394 |  374 | 161161 |
| faster_rcnn_clahe | cycle_3 | test_full | 0.988639 | 0.688635   |        0.774633  |  0.677249   | 0.70041   |  512 |    244 |  219 | 161311 |
| faster_rcnn_clahe | cycle_4 | test_full | 0.976454 | 0.637561   |        0.721117  |  0.646067   | 0.629275  |  460 |    252 |  271 | 161303 |
| megadetector      | nan     | test_full | 0.498305 | 0.00896839 |        0         |  0.00450439 | 1         |  731 | 161555 |    0 |      0 |
| rtdetr_clahe      | cycle_0 | test_full | 0.767186 | 0.29004    |        0.332719  |  0.347328   | 0.248974  |  182 |    342 |  549 | 161213 |
| rtdetr_clahe      | cycle_1 | test_full | 0.837589 | 0.326908   |        0.211179  |  0.374558   | 0.290014  |  212 |    354 |  519 | 161201 |
| rtdetr_clahe      | cycle_2 | test_full | 0.900897 | 0.463219   |        0.758989  |  0.495327   | 0.435021  |  318 |    324 |  413 | 161231 |
| rtdetr_clahe      | cycle_3 | test_full | 0.873367 | 0.514529   |        0.261188  |  0.533824   | 0.49658   |  363 |    317 |  368 | 161238 |
| rtdetr_clahe      | cycle_4 | test_full | 0.767404 | 0.290797   |        0.236643  |  0.33452    | 0.257182  |  188 |    374 |  543 | 161181 |
| yolo_clahe        | cycle_0 | test_full | 0.958723 | 0.621567   |        0.503119  |  0.608924   | 0.634747  |  464 |    298 |  267 | 161257 |
| yolo_clahe        | cycle_1 | test_full | 0.935422 | 0.581357   |        0.165593  |  0.487061   | 0.72093   |  527 |    555 |  204 | 161000 |
| yolo_clahe        | cycle_2 | test_full | 0.793103 | 0.39008    |        0.190479  |  0.382392   | 0.398085  |  291 |    470 |  440 | 161085 |
| yolo_clahe        | cycle_3 | test_full | 0.953939 | 0.749848   |        0.252637  |  0.675439   | 0.842681  |  616 |    296 |  115 | 161259 |
| yolo_clahe        | cycle_4 | test_full | 0.781946 | 0.471879   |        0.0796599 |  0.473177   | 0.470588  |  344 |    383 |  387 | 161172 |
| faster_rcnn_clahe | cycle_0 | val_full  | 0.956049 | 0.491265   |        0.766118  |  0.487907   | 0.49467   |  464 |    487 |  474 | 162328 |
| faster_rcnn_clahe | cycle_1 | val_full  | 0.967361 | 0.639009   |        0.662098  |  0.645969   | 0.632196  |  593 |    325 |  345 | 162490 |
| faster_rcnn_clahe | cycle_2 | val_full  | 0.956019 | 0.463463   |        0.545034  |  0.436792   | 0.493603  |  463 |    597 |  475 | 162218 |
| faster_rcnn_clahe | cycle_3 | val_full  | 0.984604 | 0.654565   |        0.547621  |  0.655615   | 0.653518  |  613 |    322 |  325 | 162493 |
| faster_rcnn_clahe | cycle_4 | val_full  | 0.983602 | 0.639779   |        0.631754  |  0.663991   | 0.617271  |  579 |    293 |  359 | 162522 |
| megadetector      | nan     | val_full  | 0.500256 | 0.0146374  |        0.356239  |  0.019469   | 0.0117271 |   11 |    554 |  927 | 162261 |
| rtdetr_clahe      | cycle_0 | val_full  | 0.713577 | 0.168745   |        0.323062  |  0.270588   | 0.122601  |  115 |    310 |  823 | 162505 |
| rtdetr_clahe      | cycle_1 | val_full  | 0.802445 | 0.265889   |        0.139775  |  0.293436   | 0.24307   |  228 |    549 |  710 | 162266 |
| rtdetr_clahe      | cycle_2 | val_full  | 0.854956 | 0.371747   |        0.642138  |  0.614251   | 0.266525  |  250 |    157 |  688 | 162658 |
| rtdetr_clahe      | cycle_3 | val_full  | 0.853294 | 0.418662   |        0.231841  |  0.493488   | 0.363539  |  341 |    350 |  597 | 162465 |
| rtdetr_clahe      | cycle_4 | val_full  | 0.721964 | 0.192049   |        0.209678  |  0.225251   | 0.167377  |  157 |    540 |  781 | 162275 |
| yolo_clahe        | cycle_0 | val_full  | 0.891242 | 0.480558   |        0.330644  |  0.451311   | 0.513859  |  482 |    586 |  456 | 162229 |
| yolo_clahe        | cycle_1 | val_full  | 0.870249 | 0.423564   |        0.144119  |  0.389785   | 0.463753  |  435 |    681 |  503 | 162134 |
| yolo_clahe        | cycle_2 | val_full  | 0.657601 | 0.197338   |        0.114271  |  0.173247   | 0.229211  |  215 |   1026 |  723 | 161789 |
| yolo_clahe        | cycle_3 | val_full  | 0.970754 | 0.781988   |        0.451737  |  0.806342   | 0.759062  |  712 |    171 |  226 | 162644 |
| yolo_clahe        | cycle_4 | val_full  | 0.66183  | 0.333828   |        0.085775  |  0.54878    | 0.239872  |  225 |    185 |  713 | 162630 |

## 4. Plots

- ROC and PR curves are saved as PDFs in `results/plots/`.
- Confusion matrices are also saved as PDFs in the same directory.
